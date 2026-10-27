"""One value per finished result, for the Videos table (UAT #10, viewer design P4).

A cell says what was found ("3 speakers", "412 words"), not that something ran.
Read from the result file with the export readers, so the table and the
export can't disagree; cached by file, since an OpenFace file can be tens of MB.
"""

from __future__ import annotations

import threading
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .exporters.readers import FileContents, ReaderError, classify
from .provenance import read_record

CACHE_SIZE = 5000


@dataclass(frozen=True)
class ResultValue:
    count: int
    text: str
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"count": self.count, "text": self.text, "detail": self.detail}


def _plural(n: int, one: str, many: str | None = None) -> str:
    return f"{n:,} {one if n == 1 else (many or one + 's')}"


def _most_in_a_frame(contents: FileContents) -> int:
    per_frame = Counter(
        e.frame if e.frame is not None else e.start for e in contents.events
    )
    return max(per_frame.values(), default=0)


def _tracks(contents: FileContents) -> int:
    return len({e.track for e in contents.events if e.track})


def _who(contents: FileContents, one: str, many: str | None = None) -> ResultValue:
    """People or faces: tracks where the file has them, else the most at once."""
    tracks = _tracks(contents)
    frames = len({e.frame if e.frame is not None else e.start for e in contents.events})
    seen = f"seen in {_plural(frames, 'frame')}"
    if tracks:
        return ResultValue(tracks, _plural(tracks, one, many), seen)
    most = _most_in_a_frame(contents)
    if not most:
        return ResultValue(0, f"no {many or one + 's'}")
    return ResultValue(most, _plural(most, one, many), f"most in one frame; {seen}")


def value_of(kind: str, contents: FileContents) -> ResultValue:
    """What a file of `kind` (a `FileKind.kind`) found, in one value."""
    events = contents.events
    if kind == "webvtt":
        words = sum(len((m.label or "").split()) for e in events for m in e.measures)
        if not words:
            return ResultValue(0, "no speech")
        return ResultValue(
            words, _plural(words, "word"), _plural(len(events), "segment")
        )
    if kind == "rttm":
        speakers = _tracks(contents)
        if not speakers:
            return ResultValue(0, "no speakers")
        return ResultValue(
            speakers, _plural(speakers, "speaker"), _plural(len(events), "turn")
        )
    if kind == "person":
        return _who(contents, "person", "people")
    if kind in ("face_deepface", "face_openface3"):
        return _who(contents, "face")
    if kind == "scene":
        return ResultValue(len(events), _plural(len(events), "scene"))
    if kind == "vlm":
        return ResultValue(len(events), _plural(len(events), "answer"))
    return ResultValue(len(events), _plural(len(events), "event"))


# A pipeline that writes several files is summed up by the one researchers ask
# about first: Whisper + pyannote by its words.
_PREFERRED = {"audio_processing": "webvtt"}

_lock = threading.Lock()
_cache: OrderedDict[tuple[str, int, int], ResultValue | None] = OrderedDict()


def _cached(path: Path, kind: Any) -> ResultValue | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    try:
        value: ResultValue | None = value_of(kind.kind, kind.reader(path))
    except (ReaderError, OSError, ValueError):
        value = None
    with _lock:
        _cache[key] = value
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    return value


def _record(path: Path) -> dict[str, Any] | None:
    try:
        record = read_record(path)
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def result_value(folder: Path, pipeline: str) -> ResultValue | None:
    """The value of `pipeline`'s result in a job's folder, or None if there's
    no file for it (or none that can be read)."""
    try:
        paths = sorted(p for p in folder.iterdir() if p.is_file())
    except OSError:
        return None
    found = []
    for path in paths:
        kind, _ = classify(path)
        if kind is None:
            continue
        recorded = (_record(path) or {}).get("pipeline") or {}
        name = recorded.get("name") if isinstance(recorded, dict) else None
        if (name or kind.default_pipeline) == pipeline:
            found.append((path, kind))
    if not found:
        return None
    preferred = _PREFERRED.get(pipeline)
    found.sort(key=lambda f: f[1].kind != preferred)
    return _cached(*found[0])


def clear_cache() -> None:
    with _lock:
        _cache.clear()
