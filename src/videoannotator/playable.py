"""A copy of a video that any browser can play (viewer design D7, UAT #54).

Browsers without HEVC (Firefox, Chromium builds) can't show H.265 videos. On
request, the server makes an H.264 copy with ffmpeg, keeps it under the
storage root and streams it in place of the original. Pipelines still read the
original; the copy keeps every frame's timing, so times and frame numbers match.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

from .storage.config import get_storage_root

logger = logging.getLogger(__name__)

FOLDER = "playable"
NONE, MAKING, READY, FAILED, UNAVAILABLE = (
    "none",
    "making",
    "ready",
    "failed",
    "unavailable",
)
TIMEOUT_SECONDS = 60 * 60


@dataclass
class CopyState:
    state: str
    error: str | None = None


_lock = threading.Lock()
_making: dict[Path, threading.Thread] = {}
_failed: dict[Path, str] = {}


def ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def copy_path(video: Path) -> Path:
    """Where `video`'s copy goes: named by its path, size and change time, so an
    edited or replaced file gets a new copy."""
    st = video.stat()
    key = f"{video.resolve()}\0{st.st_size}\0{st.st_mtime_ns}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:24]
    return get_storage_root() / FOLDER / f"{digest}.mp4"


def state(video: Path) -> CopyState:
    target = copy_path(video)
    if target.is_file():
        return CopyState(READY)
    with _lock:
        if target in _making:
            return CopyState(MAKING)
        if target in _failed:
            return CopyState(FAILED, _failed[target])
    if ffmpeg() is None:
        return CopyState(
            UNAVAILABLE,
            "ffmpeg isn't installed on the server, so it can't make a copy.",
        )
    return CopyState(NONE)


def start(video: Path) -> CopyState:
    """Start making `video`'s copy in the background, unless it's made or being made."""
    found = state(video)
    if found.state in (READY, MAKING, UNAVAILABLE):
        return found
    target = copy_path(video)
    with _lock:
        if target in _making:
            return CopyState(MAKING)
        _failed.pop(target, None)
        thread = threading.Thread(
            target=_make, args=(video, target), name="playable-copy", daemon=True
        )
        _making[target] = thread
    thread.start()
    return CopyState(MAKING)


def command(source: Path, target: Path) -> list[str]:
    return [
        ffmpeg() or "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        # Every frame at its own time: frame numbers and times match the original.
        "-fps_mode",
        "passthrough",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(target),
    ]


def _make(source: Path, target: Path) -> None:
    partial = target.with_suffix(".partial.mp4")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            command(source, partial),
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip()[-500:] or "ffmpeg failed")
        partial.replace(target)
        logger.info("Made a playable copy of %s at %s", source, target)
    except Exception as e:
        logger.warning("Couldn't make a playable copy of %s: %s", source, e)
        partial.unlink(missing_ok=True)
        with _lock:
            _failed[target] = f"The copy couldn't be made: {e}"
    finally:
        with _lock:
            _making.pop(target, None)
