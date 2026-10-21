"""50 Hz hardware system identification for Feetech STS3215 servos on SO-ARM101.

Logs 50 Hz step responses and static gravity-load droop on the physical follower arm,
then fits MuJoCo actuator parameters (stiffness kp, damping kd, frictionloss,
armature, effort_limit) by minimizing trajectory L2 error against mujoco.mj_step.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from mjlab.entity import Entity

from so101_mjlab.so101.so101_constants import (
  ARMATURE,
  DAMPING,
  FRICTIONLOSS,
  HOME_KEYFRAME,
  STIFFNESS,
  get_so101_robot_cfg,
)
from so101_mjlab.sts3215_driver import (
  command_joint_positions,
  interpolate_to_pose,
  open_follower_serial,
  read_joint_positions,
  relax_servos,
)

JOINT_NAMES = (
  "shoulder_pan",
  "shoulder_lift",
  "elbow_flex",
  "wrist_flex",
  "wrist_roll",
  "gripper",
)


def collect_step_responses(port: str | None, dt: float = 0.02) -> dict:
  q_home = np.array(
    [HOME_KEYFRAME.joint_pos[name] for name in JOINT_NAMES], dtype=np.float32
  )
  # Safe step amplitudes (rad) around HOME_KEYFRAME for each joint
  step_deltas = [0.15, 0.15, 0.15, 0.18, 0.25, 0.25]
  steps_per_phase = 30  # 0.60 s per step phase at 50 Hz

  ser = open_follower_serial(port)
  recorded: dict[str, list[dict[str, float]]] = {}
  gravity_probes: list[dict[str, list[float]]] = []

  try:
    print("[sysid] Interpolating to HOME_KEYFRAME...")
    interpolate_to_pose(ser, q_home, duration_s=1.5, rate_hz=50.0)
    time.sleep(0.3)

    # 1. Measure static gravity droop at HOME and EXTENDED reach postures
    probe_targets = [
      q_home.copy(),
      np.array([0.0, 0.10, 0.45, 0.65, -1.5708, 0.38], dtype=np.float32),
      np.array([0.0, 0.28, 0.35, 0.55, -1.5708, 0.38], dtype=np.float32),
    ]
    for idx, q_targ in enumerate(probe_targets):
      interpolate_to_pose(ser, q_targ, duration_s=1.0, rate_hz=50.0)
      command_joint_positions(ser, q_targ)
      time.sleep(0.5)
      samples = [read_joint_positions(ser) for _ in range(10)]
      q_mean = np.mean(samples, axis=0)
      gravity_probes.append(
        {
          "q_cmd": q_targ.tolist(),
          "q_meas": q_mean.tolist(),
        }
      )
      print(f"[sysid] Gravity probe {idx}: err_rad={np.round(q_targ - q_mean, 4)}")

    interpolate_to_pose(ser, q_home, duration_s=1.0, rate_hz=50.0)
    time.sleep(0.3)

    # 2. Measure 50 Hz step responses (+delta -> -delta -> 0) per joint
    for j_idx, j_name in enumerate(JOINT_NAMES):
      delta = step_deltas[j_idx]
      seq_Offsets = [0.0, +delta, -delta, 0.0]
      traj: list[dict[str, float]] = []
      print(
        f"[sysid] Recording 50 Hz step response for {j_name} (delta={delta:.2f} rad)..."
      )
      for offset in seq_Offsets:
        q_cmd = q_home.copy()
        q_cmd[j_idx] += offset
        for _ in range(steps_per_phase):
          t0 = time.perf_counter()
          command_joint_positions(ser, q_cmd)
          q_meas = read_joint_positions(ser)
          traj.append(
            {
              "q_cmd": float(q_cmd[j_idx]),
              "q_meas": float(q_meas[j_idx]),
              "q_all_cmd": q_cmd.tolist(),
              "q_all_meas": q_meas.tolist(),
            }
          )
          elapsed = time.perf_counter() - t0
          if elapsed < dt:
            time.sleep(dt - elapsed)
      recorded[j_name] = traj
      interpolate_to_pose(ser, q_home, duration_s=0.6, rate_hz=50.0)
      time.sleep(0.2)
  finally:
    print("[sysid] Relaxing servos...")
    relax_servos(ser)
    ser.close()

  return {"gravity_probes": gravity_probes, "step_responses": recorded}


def fit_mujoco_actuators(data: dict, dt: float = 0.02) -> dict:
  robot = Entity(get_so101_robot_cfg())
  mj_model = robot.spec.compile()
  mj_model.opt.timestep = 0.005
  mj_data = mujoco.MjData(mj_model)

  # 1. Estimate static stiffness kp from gravity probes on shoulder_lift, elbow_flex, wrist_flex
  kp_estimates: dict[str, list[float]] = {name: [] for name in JOINT_NAMES}
  for probe in data["gravity_probes"]:
    q_cmd = np.array(probe["q_cmd"], dtype=np.float64)
    q_meas = np.array(probe["q_meas"], dtype=np.float64)
    mj_data.qpos[:6] = q_meas
    mj_data.qvel[:6] = 0.0
    mj_data.qacc[:6] = 0.0
    mujoco.mj_forward(mj_model, mj_data)
    tau_grav = mj_data.qfrc_bias[:6].copy()
    err = q_cmd - q_meas
    for j_idx, name in enumerate(JOINT_NAMES):
      if abs(err[j_idx]) > 0.008 and abs(tau_grav[j_idx]) > 0.03:
        kp_val = float(tau_grav[j_idx] / err[j_idx])
        if kp_val > 0:
          kp_estimates[name].append(kp_val)
      print(
        f"  probe {name:14s}: tau_grav={tau_grav[j_idx]:+.4f} Nm, "
        f"err={err[j_idx]:+.4f} rad"
      )

  # 2. Grid search over (kp, kd, frictionloss, armature) against 50 Hz step trajectories
  fitted: dict[str, dict[str, float]] = {}
  for j_idx, name in enumerate(JOINT_NAMES):
    traj = data["step_responses"][name]
    q_cmds = np.array([pt["q_all_cmd"] for pt in traj], dtype=np.float64)
    q_meas = np.array([pt["q_all_meas"] for pt in traj], dtype=np.float64)

    best_mse = float("inf")
    best_params: dict[str, float] = {}

    for kp in (12.0, 16.0, 20.0, 25.0, 32.0, 40.0, 50.0):
      for kd in (0.4, 0.7, 1.0, 1.5, 2.2, 3.0):
        for arm in (0.010, 0.018, 0.028, 0.045):
          for floss in (0.02, 0.06, 0.12, 0.20):
            mj_model.actuator_gainprm[j_idx, 0] = kp
            mj_model.actuator_biasprm[j_idx, 1] = -kp
            mj_model.actuator_biasprm[j_idx, 2] = -kd
            mj_model.dof_armature[j_idx] = arm
            mj_model.dof_frictionloss[j_idx] = floss
            mj_model.dof_damping[j_idx] = 0.0

            mj_data.qpos[:6] = q_meas[0]
            mj_data.qvel[:6] = 0.0
            mujoco.mj_forward(mj_model, mj_data)

            sim_q = []
            for step_i in range(len(traj)):
              mj_data.ctrl[:6] = q_cmds[step_i]
              mujoco.mj_step(mj_model, mj_data, nstep=4)
              sim_q.append(float(mj_data.qpos[j_idx]))
            mse = float(np.mean(np.square(np.array(sim_q) - q_meas[:, j_idx])))
            if mse < best_mse:
              best_mse = mse
              best_params = {
                "kp": kp,
                "kd": kd,
                "armature": arm,
                "frictionloss": floss,
                "rmse_rad": float(np.sqrt(mse)),
              }
    if kp_estimates[name]:
      best_params["gravity_kp_mean"] = float(np.mean(kp_estimates[name]))
    fitted[name] = best_params
    print(f"[sysid] Fitted {name:14s}: {best_params}")

  return fitted


def main() -> None:
  parser = argparse.ArgumentParser(description="SO-101 50 Hz servo system ID.")
  parser.add_argument("--port", type=str, default=None)
  parser.add_argument(
    "--out", type=Path, default=Path("checkpoints/sysid_sts3215.json")
  )
  parser.add_argument(
    "--fit-only", type=Path, default=None, help="Fit from existing JSON file."
  )
  args = parser.parse_args()

  if args.fit_only is not None:
    payload = json.loads(args.fit_only.read_text())
  else:
    payload = collect_step_responses(args.port)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))

  fitted = fit_mujoco_actuators(payload)
  payload["fitted_params"] = fitted
  payload["shared_constants_in_so101_constants_py"] = {
    "note": "Hand-tuned shared constants starting from fitted_params and reflected rotor armature",
    "kp": STIFFNESS,
    "kd": DAMPING,
    "armature": ARMATURE,
    "frictionloss": FRICTIONLOSS,
  }
  args.out.write_text(json.dumps(payload, indent=2))
  print(f"[sysid] Saved system ID results to {args.out}")


if __name__ == "__main__":
  main()
