"""Deterministic synthetic video helpers for tests.

These utilities are intentionally small and dependency-light (OpenCV + NumPy)
so tests don't depend on external assets or flaky codec behavior.

The default container/codec is AVI+MJPG because it is broadly supported by
OpenCV builds across platforms.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SyntheticVideoSpec:
    """Specification for a deterministic synthetic video."""

    width: int = 160
    height: int = 120
    fps: int = 10
    num_frames: int = 20


def write_synthetic_video_avi(
    path: Path, spec: SyntheticVideoSpec | None = None
) -> None:
    """Write a deterministic AVI video to disk.

    The content is a simple deterministic RGB pattern that changes each frame.

    Args:
        path: Output file path (should typically end with .avi)
        spec: Video specification

    Raises:
        RuntimeError: If the video writer cannot be opened.
    """

    import cv2

    spec = spec or SyntheticVideoSpec()
    path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(
        str(path), fourcc, float(spec.fps), (spec.width, spec.height)
    )
    if not writer.isOpened():
        raise RuntimeError("Failed to open cv2.VideoWriter for synthetic video")

    try:
        for frame_idx in range(spec.num_frames):
            frame = np.zeros((spec.height, spec.width, 3), dtype=np.uint8)
            # Deterministic per-frame pattern (no randomness)
            frame[:, :, 0] = (frame_idx * 7) % 255
            frame[:, :, 1] = (
                np.arange(spec.width, dtype=np.uint8)[None, :] + frame_idx * 3
            ) % 255
            frame[:, :, 2] = (
                np.arange(spec.height, dtype=np.uint8)[:, None] + frame_idx * 5
            ) % 255
            writer.write(frame)
    finally:
        writer.release()


def synthetic_video_bytes_avi(spec: SyntheticVideoSpec | None = None) -> bytes:
    """Create a deterministic AVI video and return its bytes."""

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        write_synthetic_video_avi(tmp_path, spec=spec)
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)
