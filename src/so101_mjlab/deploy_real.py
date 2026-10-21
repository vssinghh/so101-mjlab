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
  command_joint_positions,
  interpolate_to_pose,
  open_follower_serial,
  rad_to_ticks,
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
  parser.add_argument("--cube-xyz", type=float, nargs=3, default=[0.215, 0.002, -0.036])
  parser.add_argument("--goal-xyz", type=float, nargs=3, default=[0.215, 0.002, 0.115])
  parser.add_argument("--steps", type=int, default=150)
  parser.add_argument("--max-delta-rad", type=float, default=0.025)
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
  g_fixed = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_pad")
  g_moving = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "moving_jaw_pad")
  jnt_low = mj_model.jnt_range[:6, 0].astype(np.float32)
  jnt_high = mj_model.jnt_range[:6, 1].astype(np.float32)

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
      pad_mid = (
        0.5 * (mj_data.geom_xpos[g_fixed] + mj_data.geom_xpos[g_moving])
      ).astype(np.float32)
      if (
        not grasped and np.linalg.norm(pad_mid - cube_pos_b) < 0.025 and q_cur[5] < 0.38
      ):
        grasped = True
      if grasped:
        cube_pos_b = pad_mid.copy()

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
      if grasped and mj_data.qpos[5] < 0.22:
        mj_data.qpos[5] = 0.22
        mj_data.qvel[5] = 0.0
        mujoco.mj_forward(mj_model, mj_data)
      last_action = raw_action
      if step % 15 == 0 or step == args.steps - 1:
        print(
          f"[step {step:03d}] pinch_site={np.round(ee_pos, 4)} "
          f"cube_z={cube_pos_b[2]:.4f} grasped={grasped} q_grip={q_cur[5]:.3f}"
        )
    return

  recorder = (
    LiveCameraRecorder(args.capture_dir, fps=15.0) if args.capture_dir else None
  )
  if recorder is not None:
    recorder.start()

  ser = open_follower_serial(args.port)
  try:
    q_home = DEFAULT_JOINT_POS.copy()
    interpolate_to_pose(ser, q_home, duration_s=2.5, rate_hz=50.0)
    time.sleep(0.4)
    q_prev = read_joint_positions(ser)
    q_cmd = q_prev.copy()
    dq_est = np.zeros(6, dtype=np.float32)
    act_ema = np.zeros(6, dtype=np.float32)
    q_grasp_pose: np.ndarray | None = None

    for step in range(args.steps):
      t0 = time.perf_counter()
      q_cur = read_joint_positions(ser)
      dq_est = 0.75 * dq_est + 0.25 * ((q_cur - q_prev) / dt).astype(np.float32)
      q_prev = q_cur.copy()

      mj_data.qpos[:6] = q_cur
      mujoco.mj_kinematics(mj_model, mj_data)
      ee_real = mj_data.site_xpos[site_id].astype(np.float32).copy()
      pad_mid = (
        0.5 * (mj_data.geom_xpos[g_fixed] + mj_data.geom_xpos[g_moving])
      ).astype(np.float32)

      dist_to_cube = float(np.linalg.norm(pad_mid - cube_pos_b))
      if (
        not grasped
        and dist_to_cube < 0.025
        and pad_mid[2] < -0.020
        and 0.20 < q_cur[5] < 0.40
        and abs(dq_est[5]) < 0.40
        and last_action[5] < -0.25
      ):
        grasped = True
        q_grasp_pose = q_cur.copy()
        if recorder is not None:
          recorder.snapshot("policy_grasp_clamped")
      if grasped:
        cube_pos_b = pad_mid.copy()

      obs = build_state_observation(
        mj_model, mj_data, site_id, q_cur, dq_est, cube_pos_b, goal_pos_b, last_action
      )
      raw_action = (
        np.asarray(ort_session.run(None, {input_name: obs})[0]).reshape(-1)[:6]
        if ort_session is not None and input_name is not None
        else np.zeros(6, dtype=np.float32)
      )
      act_ema = (0.55 * act_ema + 0.45 * raw_action).astype(np.float32)

      q_target = np.clip(DEFAULT_JOINT_POS + ACTION_SCALE * act_ema, jnt_low, jnt_high)
      arm_delta = args.max_delta_rad * min(1.0, (step + 1) / 10.0)
      delta_vec = np.array([arm_delta] * 5 + [0.08], dtype=np.float32)
      track_vec = np.array([0.10] * 5 + [0.25], dtype=np.float32)
      q_cmd = np.clip(q_target, q_cmd - delta_vec, q_cmd + delta_vec)
      q_cmd = np.clip(q_cmd, q_cur - track_vec, q_cur + track_vec)
      command_joint_positions(ser, q_cmd)
      last_action = raw_action

      if step % 5 == 0 or step == args.steps - 1:
        print(
          f"[step {step:03d}] real_ee={np.round(ee_real, 4)} "
          f"dist={dist_to_cube:.4f} grasped={grasped} ticks={rad_to_ticks(q_cur)}"
        )

      elapsed = time.perf_counter() - t0
      if elapsed < dt:
        time.sleep(dt - elapsed)

    time.sleep(0.45)
    if recorder is not None:
      recorder.snapshot("policy_peak_lift")

    if grasped and not args.no_place_back and q_grasp_pose is not None:
      interpolate_to_pose(ser, q_grasp_pose, duration_s=1.2, rate_hz=50.0)
      q_release = q_grasp_pose.copy()
      q_release[5] = DEFAULT_JOINT_POS[5]
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
