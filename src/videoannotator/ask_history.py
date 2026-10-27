"""Ask AI's history: every try and run, kept on the server (spec 027 R11).

An entry points at its question in the prompt library rather than copying
the text. A try's answers are saved here one by one as they land; a run's
are read from its results file. Nothing here is an image.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session, object_session

from . import prompt_library
from .database.models import AskEntry, User


def own_session() -> Session:
    """A session for a thread that outlives a request (a try's worker).

    Each session has its own connection (``database.make_engine``), so a
    worker writing while requests read is safe.
    """
    from .database import database as db_module

    return db_module.SessionLocal()


def _id(entry_id: str | uuid.UUID) -> uuid.UUID:
    return entry_id if isinstance(entry_id, uuid.UUID) else uuid.UUID(str(entry_id))


def create_entry(
    db: Session,
    *,
    kind: str,
    prompt: str,
    model: str,
    base_url: str,
    endpoint_class: str,
    sampling: dict[str, Any],
    think: bool,
    video_path: str,
    moments: list[float] | None = None,
    job_id: str | None = None,
    user_id: str | None = None,
    video_sha256: str | None = None,
    backend: str = "ollama",
    answer_shape: list[dict[str, Any]] | None = None,
) -> AskEntry:
    """Saved before any model call, so a try is never lost (SC-002). A shape
    is also kept on the library's question, so reopening it brings the shape."""
    if kind == "try":
        library = prompt_library.record_use(db, prompt, model, "try", user_id=user_id)
    else:
        library = prompt_library.ensure_prompt(db, prompt, user_id=user_id)
    if answer_shape is not None:
        library.answer_shape = answer_shape
    entry = AskEntry(
        id=uuid.uuid4(),
        kind=kind,
        status="done" if kind == "draft" else "running",
        prompt_sha256=prompt_library.prompt_sha256(prompt),
        model=model,
        base_url=base_url,
        endpoint_class=endpoint_class,
        sampling=sampling,
        think=think,
        video_path=video_path,
        video_name=Path(video_path).name,
        video_sha256=video_sha256,
        moments=moments,
        answers=[] if kind == "try" else None,
        job_id=job_id,
        user_id=user_id,
        backend=backend,
        answer_shape=answer_shape,
        starred=False,
        created_at=datetime.now(UTC),
        finished_at=datetime.now(UTC) if kind == "draft" else None,
    )
    db.add(entry)
    db.commit()
    return entry


def get_entry(db: Session, entry_id: str | uuid.UUID) -> AskEntry | None:
    try:
        return db.get(AskEntry, _id(entry_id))
    except ValueError:
        return None


def add_answer(db: Session, entry_id: str | uuid.UUID, answer: dict[str, Any]) -> None:
    entry = db.get(AskEntry, _id(entry_id))
    if entry is None:
        return
    # A new list, so SQLAlchemy sees the JSON column change.
    entry.answers = [*(entry.answers or []), answer]
    db.commit()


def finish(db: Session, entry_id: str | uuid.UUID, status: str) -> None:
    entry = db.get(AskEntry, _id(entry_id))
    if entry is None:
        return
    entry.status = status
    entry.finished_at = datetime.now(UTC)
    db.commit()


def stop_running_tries(db: Session) -> int:
    """Tries a restart cut off: kept, with the answers they got, as stopped."""
    running = (
        db.query(AskEntry)
        .filter(AskEntry.kind == "try", AskEntry.status == "running")
        .all()
    )
    now = datetime.now(UTC)
    for entry in running:
        entry.status = "stopped"
        entry.finished_at = now
    db.commit()
    return len(running)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.isoformat()


def _user_name(entry: Any) -> str | None:
    """Who asked, as people are named elsewhere, not their id (UAT #50)."""
    db = object_session(entry)
    if not entry.user_id or db is None:
        return None
    try:
        user = db.get(User, uuid.UUID(str(entry.user_id)))
    except ValueError:
        return None
    return (user.full_name or user.username) if user is not None else None


def to_dict(entry: AskEntry) -> dict[str, Any]:
    return {
        "id": str(entry.id),
        "kind": entry.kind,
        "status": entry.status,
        "prompt": entry.prompt.text if entry.prompt is not None else "",
        "prompt_sha256": entry.prompt_sha256,
        "model": entry.model,
        "base_url": entry.base_url,
        "backend": entry.backend or "ollama",
        "answer_shape": entry.answer_shape,
        "endpoint_class": entry.endpoint_class,
        "sampling": entry.sampling or {},
        "think": bool(entry.think),
        "video_path": entry.video_path,
        "video_name": entry.video_name,
        "video_sha256": entry.video_sha256,
        "moments": entry.moments,
        "answers": entry.answers,
        "job_id": entry.job_id,
        "user_id": entry.user_id,
        "user_name": _user_name(entry),
        "name": entry.name,
        "starred": bool(entry.starred),
        "created_at": _iso(entry.created_at),
        "finished_at": _iso(entry.finished_at),
    }
