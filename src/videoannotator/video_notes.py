"""Researchers' notes on a video (spec 028 R9, contracts/api.md "Notes").

A note is a moment (`end_sec` None) or a span on one video, keyed by the
video's resolved server path. Every signed-in user may read and change every
note (no admin friction); each keeps who wrote it and who last changed it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .database.models import VideoNote

MAX_TEXT = 2000
ANONYMOUS = "anonymous"


@dataclass
class NoteError(ValueError):
    """A note that can't be saved, with the API's error code."""

    code: str
    message: str

    def __str__(self) -> str:
        return self.message


class NoteNotFound(LookupError):
    """No note with that id."""


def author_of(user: dict[str, Any] | None) -> tuple[str, str]:
    """`(id, name)` of the signed-in user, as recorded on a note."""
    if not user:
        return ANONYMOUS, ANONYMOUS
    user_id = str(user.get("id") or user.get("user_id") or ANONYMOUS)
    name = user.get("full_name") or user.get("username") or user.get("email") or user_id
    return user_id[:64], str(name)[:200]


def _validated(
    start: float, end: float | None, text: str, duration: float | None
) -> tuple[float, float | None, str]:
    text = (text or "").strip()
    if not text:
        raise NoteError("NOTE_TEXT_REQUIRED", "Write something in the note.")
    if len(text) > MAX_TEXT:
        raise NoteError(
            "NOTE_TEXT_TOO_LONG",
            f"A note can be at most {MAX_TEXT:,} characters ({len(text):,} given).",
        )
    if end is not None and end < start:
        # A span marked end first (Shift+N at the later time first).
        start, end = end, start
    if start < 0:
        raise NoteError("NOTE_TIME_INVALID", "A note can't start before the video.")
    if end is not None and end <= start:
        raise NoteError("NOTE_TIME_INVALID", "A span note must end after it starts.")
    last = end if end is not None else start
    if duration is not None and last > duration + 1e-3:
        raise NoteError(
            "NOTE_TIME_INVALID",
            f"The video is {duration:.1f} s long; this note goes to {last:.1f} s.",
        )
    return start, end, text


def _resolved(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def create(
    db: Any,
    *,
    path: Path,
    start: float,
    end: float | None,
    text: str,
    user: dict[str, Any] | None,
    duration: float | None = None,
    sha256: str | None = None,
) -> VideoNote:
    start, end, text = _validated(start, end, text, duration)
    author_id, author_name = author_of(user)
    note = VideoNote(
        id=uuid.uuid4(),
        video_path=_resolved(path),
        video_name=Path(path).name,
        video_sha256=sha256,
        start_sec=start,
        end_sec=end,
        text=text,
        author_id=author_id,
        author_name=author_name,
        created_at=datetime.now(UTC),
    )
    db.add(note)
    db.commit()
    db.refresh(note)
    return note


def list_for(
    db: Any, path: str | Path, also: list[str] | None = None
) -> list[VideoNote]:
    """The video's notes, by start then creation; `also` are its other paths
    (spec 029 R1), so notes made on an upload stay with its reruns' copies."""
    paths = {_resolved(path), *(_resolved(p) for p in also or [])}
    return (
        db.query(VideoNote)
        .filter(VideoNote.video_path.in_(paths))
        .order_by(VideoNote.start_sec, VideoNote.created_at)
        .all()
    )


def get(db: Any, note_id: str) -> VideoNote:
    try:
        key = uuid.UUID(str(note_id))
    except ValueError as e:
        raise NoteNotFound(note_id) from e
    note = db.get(VideoNote, key)
    if note is None:
        raise NoteNotFound(note_id)
    return note


_UNSET: Any = object()


def update(
    db: Any,
    note_id: str,
    *,
    user: dict[str, Any] | None,
    text: Any = _UNSET,
    start: Any = _UNSET,
    end: Any = _UNSET,
    duration: float | None = None,
) -> VideoNote:
    """Change any of text, start and end (end None makes it a moment)."""
    note = get(db, note_id)
    new_start, new_end, new_text = _validated(
        note.start_sec if start is _UNSET else start,
        note.end_sec if end is _UNSET else end,
        note.text if text is _UNSET else text,
        duration,
    )
    note.start_sec, note.end_sec, note.text = new_start, new_end, new_text
    note.updated_at = datetime.now(UTC)
    note.updated_by = author_of(user)[0]
    db.commit()
    db.refresh(note)
    return note


def delete(db: Any, note_id: str) -> None:
    db.delete(get(db, note_id))
    db.commit()


def for_media_path(db: Any, path: str | Path | None) -> list[VideoNote]:
    """A video's notes for an export; none when the path is unknown."""
    if not path:
        return []
    try:
        return list_for(db, path)
    except OSError:
        return []


def _iso(when: datetime | None) -> str | None:
    if when is None:
        return None
    if when.tzinfo is None:  # SQLite hands back naive datetimes
        when = when.replace(tzinfo=UTC)
    return when.isoformat()


def to_dict(note: VideoNote) -> dict[str, Any]:
    return {
        "id": str(note.id),
        "video_path": note.video_path,
        "start_sec": note.start_sec,
        "end_sec": note.end_sec,
        "text": note.text,
        "author_id": note.author_id,
        "author_name": note.author_name,
        "created_at": _iso(note.created_at),
        "updated_at": _iso(note.updated_at),
        "updated_by": note.updated_by,
    }
