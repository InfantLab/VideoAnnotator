"""Reading the frames a VLM sample sends (spec 027 R13).

Used by the pipeline, `/vlm/preview` and Ask AI alike, so a try, the frames
shown in the window and a run all see the same frames: same decoder, same
frame rounding.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import sampling


@dataclass(frozen=True)
class SampleFrame:
    offset: int
    frame_number: int
    time: float
    jpeg: bytes


def video_metadata(video_path: str) -> dict[str, Any]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()
    return {
        "video_id": Path(video_path).stem,
        "filepath": video_path,
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
        "total_frames": total_frames,
    }


def seek_frame_jpeg(
    cap: Any, fps: float, timestamp_sec: float
) -> tuple[int, bytes] | None:
    import cv2

    frame_number = max(0, round(timestamp_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        return None
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    return frame_number, buf.tobytes()


def read_sample(
    cap: Any,
    fps: float,
    duration: float,
    anchor: float,
    offsets: list[int] | None,
    spacing: float,
) -> list[SampleFrame]:
    """The readable frames of one sample: the anchor alone when `offsets` is
    None, else each burst frame inside the video."""
    times = (
        [(0, anchor)]
        if offsets is None
        else sampling.burst_times(anchor, offsets, spacing, duration)
    )
    out = []
    for off, t in times:
        result = seek_frame_jpeg(cap, fps, t)
        if result is None:
            continue
        frame_number, jpeg = result
        frame_time = round(frame_number / fps, 3) if fps else t
        out.append(SampleFrame(off, frame_number, frame_time, jpeg))
    return out


def read_clip(
    cap: Any,
    fps: float,
    duration: float,
    anchor: float,
    length: float,
    clip_fps: float,
) -> list[SampleFrame]:
    """The readable frames of a clip around `anchor` (spec 033 R12), each
    with its index in the clip as `offset`. Frames that round to the same
    video frame are read once."""
    out: list[SampleFrame] = []
    seen: set[int] = set()
    for i, t in enumerate(sampling.clip_times(anchor, length, clip_fps, duration)):
        result = seek_frame_jpeg(cap, fps, t)
        if result is None or result[0] in seen:
            continue
        frame_number, jpeg = result
        seen.add(frame_number)
        frame_time = round(frame_number / fps, 3) if fps else t
        out.append(SampleFrame(i, frame_number, frame_time, jpeg))
    return out


def read_sample_from(
    video_path: str,
    anchor: float,
    offsets: list[int] | None,
    spacing: float,
) -> tuple[list[SampleFrame], dict[str, Any]]:
    """`read_sample` on a file, opening and closing it."""
    import cv2

    meta = video_metadata(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    try:
        return (
            read_sample(cap, meta["fps"], meta["duration"], anchor, offsets, spacing),
            meta,
        )
    finally:
        cap.release()


def thumbnail(jpeg: bytes, width: int = 320) -> str:
    """`jpeg` scaled down to `width` px wide, base64-encoded."""
    import cv2
    import numpy as np

    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if frame is not None and frame.shape[1] > width:
        height = round(frame.shape[0] * width / frame.shape[1])
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            jpeg = buf.tobytes()
    return base64.b64encode(jpeg).decode("ascii")
