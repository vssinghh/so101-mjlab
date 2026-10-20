"""Feetech STS3215 1 Mbps USB serial driver and calibration helpers for SO-ARM101."""

from __future__ import annotations

import glob
import json
import os
import time
from pathlib import Path

import numpy as np

DEFAULT_PORT = os.environ.get("SO101_PORT", "/dev/tty.usbmodem5B790143431")
KNOWN_LEADER_SUBSTR = os.environ.get("SO101_LEADER_SUBSTR", "5B8E116931")
DEFAULT_LEROBOT_CALIB = (
  Path.home()
  / ".cache/huggingface/lerobot/calibration/robots/so_follower/so_101_follower.json"
)
BAUD = 1_000_000
DEFAULT_ZERO_OFFSETS = np.array(
  [2027.5, 1851.0, 2035.0, 1930.0, 1997.0, 2048.0], dtype=np.float64
)
RAD_PER_TICK = (2.0 * np.pi) / 4096.0


def load_zero_offsets(calibration_path: Path | None = None) -> np.ndarray:
  """Load zero-radian tick offsets from a LeRobot follower calibration JSON file."""
  offsets = DEFAULT_ZERO_OFFSETS.copy()
  cand = calibration_path or Path(
    os.environ.get("SO101_CALIBRATION_FILE", str(DEFAULT_LEROBOT_CALIB))
  )
  if not cand.exists():
    return offsets
  try:
    data = json.loads(cand.read_text())
    joint_names = (
      "shoulder_pan",
      "shoulder_lift",
      "elbow_flex",
      "wrist_flex",
      "wrist_roll",
      "gripper",
    )
    for idx, name in enumerate(joint_names):
      entry = data.get(name, {})
      if "zero_offset" in entry:
        offsets[idx] = float(entry["zero_offset"])
      elif idx < 4 and "range_min" in entry and "range_max" in entry:
        offsets[idx] = 0.5 * (float(entry["range_min"]) + float(entry["range_max"]))
  except Exception:
    pass
  return offsets


ZERO_OFFSETS = load_zero_offsets()


def _checksum(packet_body: list[int]) -> int:
  return (~sum(packet_body)) & 0xFF


def read_u16(ser, motor_id: int, reg: int) -> int:
  body = [motor_id, 4, 2, reg, 2]
  ser.reset_input_buffer()
  ser.write(bytes([0xFF, 0xFF, *body, _checksum(body)]))
  resp = ser.read(8)
  if (
    len(resp) == 8
    and resp[0] == 0xFF
    and resp[1] == 0xFF
    and resp[2] == motor_id
    and _checksum(list(resp[2:7])) == resp[7]
  ):
    val = int(resp[5] | (resp[6] << 8))
    if reg != 56 or 0 <= val < 4096:
      return val
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


def configure_servo_torque(ser) -> None:
  """Sync goal position to present position, set smooth acc/speed profiles, and enable torque."""
  for m in range(1, 7):
    cur = read_u16(ser, m, 56)
    if cur < 0:
      cur = read_u16(ser, m, 56)
    if 0 <= cur < 4096:
      write_u16(ser, m, 42, cur)
    write_u8(ser, m, 41, 35)
    write_u16(ser, m, 46, 1100)
    write_u16(ser, m, 48, 500 if m < 6 else 400)
    write_u8(ser, m, 40, 1)
  time.sleep(0.02)
  ser.reset_input_buffer()


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
  configure_servo_torque(ser)
  q_start = read_joint_positions(ser)
  steps = max(1, int(duration_s * rate_hz))
  dt = 1.0 / rate_hz
  for step in range(1, steps + 1):
    alpha = 0.5 * (1.0 - np.cos(np.pi * (step / steps)))
    q_cmd = (1.0 - alpha) * q_start + alpha * q_target
    command_joint_positions(ser, q_cmd)
    time.sleep(dt)


def relax_servos(ser) -> None:
  for m in range(1, 7):
    write_u8(ser, m, 40, 0)
