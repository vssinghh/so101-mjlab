"""Closed-loop 50 Hz ONNX policy deployment for the physical SO-ARM101 follower arm."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
from mjlab.entity import Entity

from so101_mjlab.camera_recorder import LiveCameraRecorder
from so101_mjlab.so101.so101_constants import (
  HOME_KEYFRAME,
  SO101_ACTION_SCALE,
  get_so101_robot_cfg,
)
from so101_mjlab.sts3215_driver import (
  RAD_PER_TICK,
  ZERO_OFFSETS,
  command_joint_positions,
  interpolate_to_pose,
  open_follower_serial,
  rad_to_ticks,
  read_joint_positions,
  relax_servos,
  ticks_to_rad,
)

JOINT_NAMES = (
  "shoulder_pan",
  "shoulder_lift",
  "elbow_flex",
  "wrist_flex",
  "wrist_roll",
  "gripper",
)

DEFAULT_JOINT_POS = np.array(
  [HOME_KEYFRAME.joint_pos[name] for name in JOINT_NAMES], dtype=np.float32
)
ACTION_SCALE = np.array(
  [SO101_ACTION_SCALE[name] for name in JOINT_NAMES], dtype=np.float32
)


def build_state_observation(
  mj_model: mujoco.MjModel,
  mj_data: mujoco.MjData,
  site_id: int,
  q_rad: np.ndarray,
  dq_rad: np.ndarray,
  cube_pos_b: np.ndarray,
  goal_pos_b: np.ndarray,
  last_action: np.ndarray,
) -> np.ndarray:
  """Construct the 24-D actor observation vector matching Mjlab-Lift-Cube-SO101."""
  mj_data.qpos[:6] = q_rad
  mujoco.mj_kinematics(mj_model, mj_data)
  ee_pos_b = mj_data.site_xpos[site_id].astype(np.float32)

  obs = np.concatenate(
    [
      q_rad - DEFAULT_JOINT_POS,
      dq_rad,
      cube_pos_b - ee_pos_b,
      goal_pos_b - cube_pos_b,
      last_action,
    ],
    axis=0,
  ).astype(np.float32)
  return obs.reshape(1, -1)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Deploy an mjlab ONNX manipulation policy to the physical SO-101 arm."
  )
  parser.add_argument("--onnx", type=Path, default=None, help="Exported ONNX policy.")
  parser.add_argument("--port", type=str, default=None, help="Serial port override.")
  parser.add_argument(
    "--cube-xyz", type=float, nargs=3, default=[0.235, 0.018, -0.034]
  )
  parser.add_argument("--goal-xyz", type=float, nargs=3, default=[0.235, 0.018, 0.115])
  parser.add_argument("--steps", type=int, default=75)
  parser.add_argument("--max-delta-rad", type=float, default=0.08)
  parser.add_argument("--gripper-open-tick", type=int, default=2560)
  parser.add_argument("--gripper-clamp-tick", type=int, default=1800)
  parser.add_argument("--capture-dir", type=Path, default=None)
  parser.add_argument("--no-place-back", action="store_true")
  parser.add_argument("--relax", action="store_true")
  parser.add_argument("--dry-run", action="store_true")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  robot = Entity(get_so101_robot_cfg())
  mj_model = robot.spec.compile()
  mj_model.opt.timestep = 0.005
  mj_data = mujoco.MjData(mj_model)
  site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "pinch_site")
  jnt_low = mj_model.jnt_range[:6, 0].astype(np.float32)
  jnt_high = mj_model.jnt_range[:6, 1].astype(np.float32)
  q_clamp_min = float((args.gripper_clamp_tick - ZERO_OFFSETS[5]) * RAD_PER_TICK)
  q_open_max = float((args.gripper_open_tick - ZERO_OFFSETS[5]) * RAD_PER_TICK)

  ort_session = None
  input_name = None
  if args.onnx is not None:
    import onnxruntime as ort

    ort_session = ort.InferenceSession(
      str(args.onnx), providers=["CPUExecutionProvider"]
    )
    input_name = ort_session.get_inputs()[0].name

  cube_pos_b = np.asarray(args.cube_xyz, dtype=np.float32)
  goal_pos_b = np.asarray(args.goal_xyz, dtype=np.float32)
  last_action = np.zeros(6, dtype=np.float32)
  grasped = False
  dt = 0.02

  if args.dry_run:
    mj_data.qpos[:6] = DEFAULT_JOINT_POS
    mj_data.qvel[:6] = 0.0
    mujoco.mj_forward(mj_model, mj_data)
    for step in range(args.steps):
      q_cur = mj_data.qpos[:6].astype(np.float32).copy()
      dq_cur = mj_data.qvel[:6].astype(np.float32).copy()
      ee_pos = mj_data.site_xpos[site_id].astype(np.float32).copy()
      if not grasped and np.linalg.norm(ee_pos - cube_pos_b) < 0.025 and q_cur[5] < 0.28:
        grasped = True
      if grasped:
        cube_pos_b = ee_pos.copy()

      obs = build_state_observation(
        mj_model, mj_data, site_id, q_cur, dq_cur, cube_pos_b, goal_pos_b, last_action
      )
      raw_action = (
        np.asarray(ort_session.run(None, {input_name: obs})[0]).reshape(-1)[:6]
        if ort_session is not None and input_name is not None
        else np.zeros(6, dtype=np.float32)
      )
      mj_data.ctrl[:6] = DEFAULT_JOINT_POS + ACTION_SCALE * raw_action
      mujoco.mj_step(mj_model, mj_data, nstep=4)
      if grasped and mj_data.qpos[5] < 0.175:
        mj_data.qpos[5] = 0.175
        mj_data.qvel[5] = 0.0
        mujoco.mj_forward(mj_model, mj_data)
      last_action = raw_action
      if step % 15 == 0 or step == args.steps - 1:
        print(
          f"[step {step:03d}] pinch_site={np.round(ee_pos, 4)} "
          f"cube_z={cube_pos_b[2]:.4f} grasped={grasped} q_grip={q_cur[5]:.3f}"
        )
    return

  recorder = LiveCameraRecorder(args.capture_dir, fps=15.0) if args.capture_dir else None
  if recorder is not None:
    recorder.start()

  ser = open_follower_serial(args.port)
  try:
    q_home = DEFAULT_JOINT_POS.copy()
    q_home[5] = q_open_max
    interpolate_to_pose(ser, q_home, duration_s=1.5, rate_hz=50.0)
    dq_est = np.zeros(6, dtype=np.float32)
    q_grasp_pose: np.ndarray | None = None

    for step in range(args.steps):
      t0 = time.perf_counter()
      q_cur = read_joint_positions(ser)
      q_obs = q_cur.copy()
      if grasped and q_obs[5] < 0.175:
        q_obs[5] = 0.175

      mj_data.qpos[:6] = q_obs
      mj_data.qvel[:6] = dq_est
      mujoco.mj_forward(mj_model, mj_data)
      ee_real = mj_data.site_xpos[site_id].astype(np.float32).copy()

      if not grasped and np.linalg.norm(ee_real - cube_pos_b) < 0.030 and last_action[5] < -0.5:
        grasped = True
        q_grasp_pose = q_cur.copy()
        if recorder is not None:
          recorder.snapshot("policy_grasp_clamped")
      if grasped:
        cube_pos_b = ee_real.copy()

      obs = build_state_observation(
        mj_model, mj_data, site_id, q_obs, dq_est, cube_pos_b, goal_pos_b, last_action
      )
      raw_action = (
        np.asarray(ort_session.run(None, {input_name: obs})[0]).reshape(-1)[:6]
        if ort_session is not None and input_name is not None
        else np.zeros(6, dtype=np.float32)
      )

      q_target = DEFAULT_JOINT_POS + ACTION_SCALE * raw_action
      mj_data.ctrl[:6] = q_target
      mujoco.mj_step(mj_model, mj_data, nstep=4)
      q_next = mj_data.qpos[:6].astype(np.float32).copy()
      dq_est = mj_data.qvel[:6].astype(np.float32).copy()
      if grasped and dq_est[5] < 0.0:
        dq_est[5] = 0.0

      q_cmd = np.clip(np.clip(q_next, jnt_low, jnt_high), q_cur - args.max_delta_rad, q_cur + args.max_delta_rad)
      q_cmd[5] = max(q_clamp_min, float(q_target[5])) if (raw_action[5] < -0.5 or grasped) else min(q_open_max, float(q_cmd[5] + 0.22))
      command_joint_positions(ser, q_cmd)
      last_action = raw_action

      if step % 15 == 0 or step == args.steps - 1:
        print(f"[step {step:03d}] real_ee={np.round(ee_real, 4)} grasped={grasped} ticks={rad_to_ticks(q_cur)}")

      elapsed = time.perf_counter() - t0
      if elapsed < dt:
        time.sleep(dt - elapsed)

    time.sleep(0.45)
    if recorder is not None:
      recorder.snapshot("policy_peak_lift")

    if grasped and not args.no_place_back and q_grasp_pose is not None:
      interpolate_to_pose(ser, q_grasp_pose, duration_s=1.2, rate_hz=50.0)
      q_release = q_grasp_pose.copy()
      q_release[5] = q_open_max
      interpolate_to_pose(ser, q_release, duration_s=0.45, rate_hz=50.0)

    interpolate_to_pose(ser, q_home, duration_s=1.2, rate_hz=50.0)
  finally:
    if recorder is not None:
      recorder.stop_and_save()
    if args.relax:
      relax_servos(ser)
    ser.close()


if __name__ == "__main__":
  main()
