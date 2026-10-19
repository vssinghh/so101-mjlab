"""Closed-loop 50 Hz ONNX policy deployment for the physical SO-ARM101 follower arm."""

from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import mujoco
import numpy as np
from mjlab.entity import Entity

from so101_mjlab.so101.so101_constants import (
  HOME_KEYFRAME,
  SO101_ACTION_SCALE,
  get_so101_robot_cfg,
)

DEFAULT_PORT = "/dev/tty.usbmodem5B790143431"
KNOWN_LEADER_SUBSTR = "5B8E116931"
BAUD = 1_000_000
ZERO_OFFSETS = np.array(
  [2027.5, 1851.0, 2035.0, 1930.0, 1997.0, 2048.0], dtype=np.float64
)
RAD_PER_TICK = (2.0 * np.pi) / 4096.0

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
BASE_POS_OFFSET = np.array(HOME_KEYFRAME.pos, dtype=np.float32)
ACTION_SCALE = np.array(
  [SO101_ACTION_SCALE[name] for name in JOINT_NAMES], dtype=np.float32
)


def _checksum(packet_body: list[int]) -> int:
  return (~sum(packet_body)) & 0xFF


def read_u16(ser, motor_id: int, reg: int) -> int:
  body = [motor_id, 4, 2, reg, 2]
  ser.reset_input_buffer()
  ser.write(bytes([0xFF, 0xFF, *body, _checksum(body)]))
  resp = ser.read(8)
  if len(resp) == 8 and resp[0] == 0xFF and resp[1] == 0xFF:
    return int(resp[5] | (resp[6] << 8))
  return -1


def write_u8(ser, motor_id: int, reg: int, val: int) -> None:
  body = [motor_id, 4, 3, reg, val & 0xFF]
  ser.write(bytes([0xFF, 0xFF, *body, _checksum(body)]))
  ser.read(6)


def write_u16(ser, motor_id: int, reg: int, val: int) -> None:
  lo = val & 0xFF
  hi = (val >> 8) & 0xFF
  body = [motor_id, 5, 3, reg, lo, hi]
  ser.write(bytes([0xFF, 0xFF, *body, _checksum(body)]))
  ser.read(6)


def open_follower_serial(port: str | None = None):
  import serial

  if port is not None:
    return serial.Serial(port, BAUD, timeout=0.02)

  all_usb = sorted(glob.glob("/dev/tty.usbmodem*"))
  candidates: list[str] = []
  if Path(DEFAULT_PORT).exists():
    candidates.append(DEFAULT_PORT)
  for p in all_usb:
    if p not in candidates and KNOWN_LEADER_SUBSTR not in p:
      candidates.append(p)

  if not candidates:
    raise RuntimeError(
      "No follower /dev/tty.usbmodem* port detected. Check USB-C connection."
    )

  for candidate in candidates:
    ser = serial.Serial(candidate, BAUD, timeout=0.02)
    if read_u16(ser, 1, 56) >= 0:
      return ser
    ser.close()

  raise RuntimeError(
    f"Follower serial port found ({candidates}), but STS3215 servos did not respond. "
    "Verify that 12V power is switched on."
  )


def ticks_to_rad(ticks: list[int]) -> np.ndarray:
  return ((np.asarray(ticks, dtype=np.float64) - ZERO_OFFSETS) * RAD_PER_TICK).astype(
    np.float32
  )


def rad_to_ticks(q_rad: np.ndarray) -> list[int]:
  raw = np.round(ZERO_OFFSETS + (q_rad.astype(np.float64) / RAD_PER_TICK)).astype(int)
  return [int(np.clip(v, 200, 3900)) for v in raw]


def read_joint_positions(ser) -> np.ndarray:
  ticks: list[int] = []
  for m in range(1, 7):
    val = read_u16(ser, m, 56)
    if val < 0:
      val = read_u16(ser, m, 56)
    if val < 0:
      raise RuntimeError(f"Lost communication with STS3215 servo ID {m}.")
    ticks.append(val)
  return ticks_to_rad(ticks)


def command_joint_positions(ser, q_target_rad: np.ndarray) -> None:
  target_ticks = rad_to_ticks(q_target_rad)
  for m in range(1, 7):
    write_u16(ser, m, 42, target_ticks[m - 1])


def interpolate_to_pose(
  ser, q_target: np.ndarray, duration_s: float = 2.0, rate_hz: float = 50.0
) -> None:
  q_start = read_joint_positions(ser)
  steps = max(1, int(duration_s * rate_hz))
  dt = 1.0 / rate_hz
  for m in range(1, 7):
    write_u8(ser, m, 40, 1)
  for step in range(1, steps + 1):
    alpha = 0.5 * (1.0 - np.cos(np.pi * (step / steps)))
    q_cmd = (1.0 - alpha) * q_start + alpha * q_target
    command_joint_positions(ser, q_cmd)
    time.sleep(dt)


def relax_servos(ser) -> None:
  for m in range(1, 7):
    write_u8(ser, m, 40, 0)


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

  joint_pos_rel = q_rad - DEFAULT_JOINT_POS
  joint_vel_rel = dq_rad
  ee_to_cube = cube_pos_b - ee_pos_b
  cube_to_goal = goal_pos_b - cube_pos_b

  obs = np.concatenate(
    [joint_pos_rel, joint_vel_rel, ee_to_cube, cube_to_goal, last_action],
    axis=0,
  ).astype(np.float32)
  return obs.reshape(1, -1)


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Deploy an mjlab ONNX manipulation policy to the physical SO-101 arm."
  )
  parser.add_argument(
    "--onnx",
    type=Path,
    default=None,
    help="Path to exported ONNX policy file (if omitted, runs zero-action holding test).",
  )
  parser.add_argument(
    "--port",
    type=str,
    default=None,
    help="Serial port override (auto-detects follower /dev/tty.usbmodem* by default).",
  )
  parser.add_argument(
    "--cube-xyz",
    type=float,
    nargs=3,
    default=[0.22, 0.0, 0.015],
    help="Initial cube position [x, y, z] in base frame (meters).",
  )
  parser.add_argument(
    "--goal-xyz",
    type=float,
    nargs=3,
    default=[0.22, 0.0, 0.16],
    help="Target lift goal [x, y, z] in base frame (meters).",
  )
  parser.add_argument(
    "--steps",
    type=int,
    default=250,
    help="Number of 50 Hz control steps to execute (default: 250 = 5.0 seconds).",
  )
  parser.add_argument(
    "--max-delta-rad",
    type=float,
    default=0.15,
    help="Per-step joint command safety clamp relative to current pose (radians).",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Run kinematic & ONNX inference loop in simulation without opening USB serial.",
  )
  args = parser.parse_args()

  robot = Entity(get_so101_robot_cfg())
  mj_model = robot.spec.compile()
  mj_data = mujoco.MjData(mj_model)
  site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "pinch_site")
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
  q_prev = DEFAULT_JOINT_POS.copy()
  grasped = False
  dt = 0.02  # 50 Hz control rate

  if args.dry_run:
    print("Running in --dry-run mode (no hardware serial required).")
    mj_data.qpos[:6] = DEFAULT_JOINT_POS
    mj_data.qvel[:6] = 0.0
    mujoco.mj_forward(mj_model, mj_data)
    for step in range(args.steps):
      q_cur = mj_data.qpos[:6].astype(np.float32).copy()
      dq_cur = mj_data.qvel[:6].astype(np.float32).copy()
      ee_pos = mj_data.site_xpos[site_id].astype(np.float32).copy()
      if (
        not grasped and np.linalg.norm(ee_pos - cube_pos_b) < 0.025 and q_cur[5] < 0.24
      ):
        grasped = True
      if grasped:
        cube_pos_b = ee_pos.copy()

      obs = build_state_observation(
        mj_model,
        mj_data,
        site_id,
        q_cur,
        dq_cur,
        cube_pos_b,
        goal_pos_b,
        last_action,
      )
      if ort_session is not None and input_name is not None:
        raw_action = np.asarray(ort_session.run(None, {input_name: obs})[0]).reshape(
          -1
        )[:6]
      else:
        raw_action = np.zeros(6, dtype=np.float32)
      q_target = DEFAULT_JOINT_POS + ACTION_SCALE * raw_action
      mj_data.ctrl[:6] = q_target
      mujoco.mj_step(mj_model, mj_data, nstep=4)
      if grasped and mj_data.qpos[5] < 0.175:
        mj_data.qpos[5] = 0.175
        mj_data.qvel[5] = 0.0
        mujoco.mj_forward(mj_model, mj_data)
      last_action = raw_action
      if step % 25 == 0 or step == args.steps - 1:
        print(
          f"[step {step:03d}] pinch_site={np.round(ee_pos, 4)} "
          f"cube_z={cube_pos_b[2]:.4f} grasped={grasped} q_grip={q_cur[5]:.3f}"
        )
    return

  ser = open_follower_serial(args.port)
  try:
    print("Moving SO-101 smoothly to HOME_KEYFRAME...")
    interpolate_to_pose(ser, DEFAULT_JOINT_POS, duration_s=2.0, rate_hz=50.0)
    q_prev = read_joint_positions(ser)

    print(f"Running 50 Hz closed-loop policy for {args.steps} steps...")
    for step in range(args.steps):
      t0 = time.perf_counter()
      q_cur = read_joint_positions(ser)
      dq_cur = (q_cur - q_prev) / dt
      q_prev = q_cur.copy()

      mj_data.qpos[:6] = q_cur
      mujoco.mj_kinematics(mj_model, mj_data)
      ee_pos = mj_data.site_xpos[site_id].astype(np.float32)
      if (
        not grasped and np.linalg.norm(ee_pos - cube_pos_b) < 0.025 and q_cur[5] < 0.24
      ):
        grasped = True
      if grasped:
        cube_pos_b = ee_pos.copy()

      obs = build_state_observation(
        mj_model,
        mj_data,
        site_id,
        q_cur,
        dq_cur,
        cube_pos_b,
        goal_pos_b,
        last_action,
      )
      if ort_session is not None and input_name is not None:
        raw_action = np.asarray(ort_session.run(None, {input_name: obs})[0]).reshape(
          -1
        )[:6]
      else:
        raw_action = np.zeros(6, dtype=np.float32)

      q_des = np.clip(DEFAULT_JOINT_POS + ACTION_SCALE * raw_action, jnt_low, jnt_high)
      q_cmd = np.clip(q_des, q_cur - args.max_delta_rad, q_cur + args.max_delta_rad)
      command_joint_positions(ser, q_cmd)
      last_action = raw_action
      if step % 25 == 0 or step == args.steps - 1:
        print(
          f"[step {step:03d}] pinch_site={np.round(ee_pos, 4)} "
          f"cube_z={cube_pos_b[2]:.4f} grasped={grasped} q_cmd={np.round(q_cmd, 3)}"
        )

      elapsed = time.perf_counter() - t0
      if elapsed < dt:
        time.sleep(dt - elapsed)

    print("Returning SO-101 smoothly to HOME_KEYFRAME...")
    interpolate_to_pose(ser, DEFAULT_JOINT_POS, duration_s=1.5, rate_hz=50.0)
  finally:
    relax_servos(ser)
    ser.close()


if __name__ == "__main__":
  main()
