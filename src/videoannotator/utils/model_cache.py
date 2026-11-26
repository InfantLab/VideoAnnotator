"""Models kept loaded between videos (UAT #72).

A job builds its pipelines afresh for every video. For most pipelines that
costs little, but loading pyannote's diarization model, a Whisper model or
CLIP took 3-14 s per video, as long as the work itself on a short clip. The
pipelines that load those ask this cache instead: the first video loads the
model, later videos reuse it, and a model nobody has used for
``VIDEOANNOTATOR_MODEL_IDLE_SECONDS`` (default 300) is released so the GPU is
free for other work, such as Ask AI. Set it to 0 to load per video as before.

Jobs run in threads of one process, two at a time by default, so two videos
may want the same model at once: each entry has a lock that a pipeline holds
while it runs the model, and the two take turns.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from collections.abc import Callable, Hashable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_IDLE_SECONDS = 300.0
_SWEEP_SECONDS = 30.0


@dataclass
class Entry:
    """One loaded model, what loading it returned, and its turn-taking lock."""

    key: Hashable
    value: Any
    lock: threading.RLock = field(default_factory=threading.RLock)
    last_used: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.last_used = time.monotonic()


_entries: dict[Hashable, Entry] = {}
_loading: dict[Hashable, threading.Lock] = {}
_guard = threading.Lock()
_sweeper: threading.Thread | None = None


def idle_seconds() -> float:
    raw = os.environ.get("VIDEOANNOTATOR_MODEL_IDLE_SECONDS")
    if raw is None or raw.strip() == "":
        return DEFAULT_IDLE_SECONDS
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning(
            "VIDEOANNOTATOR_MODEL_IDLE_SECONDS=%r isn't a number; using %s",
            raw,
            DEFAULT_IDLE_SECONDS,
        )
        return DEFAULT_IDLE_SECONDS


def get(key: Hashable, load: Callable[[], Any]) -> Entry:
    """The entry for ``key``, calling ``load`` only if it isn't loaded.

    Two callers asking for the same key at once load it once. A failed load
    caches nothing, so the next video tries again.
    """
    if idle_seconds() == 0:
        return Entry(key, load())
    with _guard:
        entry = _entries.get(key)
        if entry is not None:
            entry.touch()
            return entry
        loading = _loading.setdefault(key, threading.Lock())
    with loading:
        with _guard:
            entry = _entries.get(key)
            if entry is not None:
                entry.touch()
                return entry
        value = load()
        entry = Entry(key, value)
        with _guard:
            _entries[key] = entry
            _loading.pop(key, None)
            _start_sweeper()
        logger.info("Model loaded and kept for later videos: %s", _label(key))
        return entry


def release(key: Hashable) -> None:
    """Forget ``key``'s model, e.g. after it failed in a way loading again fixes."""
    with _guard:
        entry = _entries.pop(key, None)
    if entry is not None:
        _free([entry])


def release_idle(now: float | None = None) -> int:
    """Release models unused for the idle time; returns how many."""
    limit = idle_seconds()
    now = time.monotonic() if now is None else now
    idle: list[Entry] = []
    with _guard:
        for key, entry in list(_entries.items()):
            if now - entry.last_used < limit:
                continue
            # A pipeline running the model holds its lock: not idle.
            if not entry.lock.acquire(blocking=False):
                continue
            entry.lock.release()
            idle.append(_entries.pop(key))
    _free(idle)
    return len(idle)


def clear() -> None:
    """Release every model (tests, shutdown)."""
    with _guard:
        entries = list(_entries.values())
        _entries.clear()
    _free(entries)


def loaded() -> list[Hashable]:
    with _guard:
        return list(_entries)


def _free(entries: list[Entry]) -> None:
    if not entries:
        return
    # The model is freed once the last pipeline holding the entry lets go of
    # it; a job between loading and using it keeps a working model.
    for entry in entries:
        logger.info("Released idle model: %s", _label(entry.key))
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _label(key: Hashable) -> str:
    return " ".join(str(part) for part in key) if isinstance(key, tuple) else str(key)


def _start_sweeper() -> None:
    global _sweeper
    if _sweeper is not None and _sweeper.is_alive():
        return
    _sweeper = threading.Thread(target=_sweep, name="model-cache-sweep", daemon=True)
    _sweeper.start()


def _sweep() -> None:
    while True:
        time.sleep(min(_SWEEP_SECONDS, max(1.0, idle_seconds() / 2)))
        try:
            release_idle()
        except Exception:  # never let the sweeper die
            logger.exception("Releasing idle models failed")


def turn(entry: Entry | None) -> AbstractContextManager[Any]:
    """Hold ``entry``'s model while running it; nothing to hold without one."""
    if entry is None:
        return nullcontext()
    entry.touch()
    return entry.lock
