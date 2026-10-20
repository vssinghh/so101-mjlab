"""Background multi-camera OpenCV video and GIF recorder for SO-ARM101 rollouts."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np


class LiveCameraRecorder:
  """Background OpenCV multi-camera recorder for non-blocking 50 Hz rollouts."""

  def __init__(self, out_dir: Path, fps: float = 15.0) -> None:
    self.out_dir = out_dir
    self.fps = fps
    self.out_dir.mkdir(parents=True, exist_ok=True)
    self._caps: dict[int, object] = {}
    self._latest: dict[int, np.ndarray] = {}
    self._frames_side: list[np.ndarray] = []
    self._frames_dual: list[np.ndarray] = []
    self._lock = threading.Lock()
    self._running = False
    self._thread: threading.Thread | None = None

  def start(self) -> None:
    try:
      import cv2
    except ImportError:
      return
    for idx in (1, 2):
      cap = cv2.VideoCapture(idx)
      if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        for _ in range(6):
          cap.read()
        ok, frame = cap.read()
        if ok and frame is not None:
          self._caps[idx] = cap
          self._latest[idx] = frame
        else:
          cap.release()
    if not self._caps:
      return
    self._running = True
    self._thread = threading.Thread(target=self._loop, daemon=True)
    self._thread.start()

  def _loop(self) -> None:
    import cv2

    dt = 1.0 / self.fps
    while self._running:
      t0 = time.perf_counter()
      current: dict[int, np.ndarray] = {}
      for idx, cap in self._caps.items():
        ok, frame = cap.read()  # type: ignore[attr-defined]
        if ok and frame is not None:
          current[idx] = frame
      with self._lock:
        self._latest.update(current)
        f1 = self._latest.get(1)
        f2 = self._latest.get(2)
        if f1 is not None:
          self._frames_side.append(f1.copy())
          if f2 is not None:
            h_target = 360
            w1 = int(f1.shape[1] * (h_target / f1.shape[0]))
            w2 = int(f2.shape[1] * (h_target / f2.shape[0]))
            r1 = cv2.resize(f1, (w1, h_target), interpolation=cv2.INTER_AREA)
            r2 = cv2.resize(f2, (w2, h_target), interpolation=cv2.INTER_AREA)
            self._frames_dual.append(np.hstack([r1, r2]))
      elapsed = time.perf_counter() - t0
      if elapsed < dt:
        time.sleep(dt - elapsed)

  def snapshot(self, prefix: str) -> list[str]:
    import cv2

    saved: list[str] = []
    with self._lock:
      frames = {k: v.copy() for k, v in self._latest.items()}
    for idx, frame in sorted(frames.items()):
      path = self.out_dir / f"{prefix}_cam_{idx}.png"
      cv2.imwrite(str(path), frame)
      saved.append(str(path))
    return saved

  def stop_and_save(self) -> list[str]:
    self._running = False
    if self._thread is not None:
      self._thread.join(timeout=2.0)
    for cap in self._caps.values():
      cap.release()  # type: ignore[attr-defined]
    self._caps.clear()

    saved_files: list[str] = []
    try:
      import cv2
      from PIL import Image
    except ImportError:
      return saved_files

    if self._frames_side:
      h, w = self._frames_side[0].shape[:2]
      h_gif = 360
      w_gif = int(w * (h_gif / h))
      mp4_side = self.out_dir / "so101_real_lift_side.mp4"
      writer = cv2.VideoWriter(
        str(mp4_side),
        cv2.VideoWriter_fourcc(*"mp4v"),
        self.fps,
        (w_gif, h_gif),
      )
      pil_side: list[Image.Image] = []
      for f in self._frames_side:
        resized = cv2.resize(f, (w_gif, h_gif), interpolation=cv2.INTER_AREA)
        writer.write(resized)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_side.append(Image.fromarray(rgb))
      writer.release()
      saved_files.append(str(mp4_side))

      gif_side = self.out_dir / "so101_real_lift_side.gif"
      pil_side[0].save(
        gif_side,
        save_all=True,
        append_images=pil_side[1:],
        duration=int(1000.0 / self.fps),
        loop=0,
        optimize=True,
      )
      saved_files.append(str(gif_side))

    if self._frames_dual:
      h_d, w_d = self._frames_dual[0].shape[:2]
      mp4_dual = self.out_dir / "so101_sim2real_lift.mp4"
      writer = cv2.VideoWriter(
        str(mp4_dual),
        cv2.VideoWriter_fourcc(*"mp4v"),
        self.fps,
        (w_d, h_d),
      )
      pil_dual: list[Image.Image] = []
      for f in self._frames_dual:
        writer.write(f)
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        pil_dual.append(Image.fromarray(rgb))
      writer.release()
      saved_files.append(str(mp4_dual))

      gif_dual = self.out_dir / "so101_sim2real_lift.gif"
      pil_dual[0].save(
        gif_dual,
        save_all=True,
        append_images=pil_dual[1:],
        duration=int(1000.0 / self.fps),
        loop=0,
        optimize=True,
      )
      saved_files.append(str(gif_dual))

    return saved_files
