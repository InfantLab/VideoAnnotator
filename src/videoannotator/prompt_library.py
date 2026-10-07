"""Every VLM prompt that runs, kept once per text (spec 020).

A prompt is identified by the SHA-256 of its exact text (no normalisation:
reproducibility needs the exact bytes), the same hash spec 017's provenance
records with each VLM output. Each run adds a use: the model, whether it was a
job or a preview, the job, who, and when.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from .database.models import Prompt, PromptUse

logger = logging.getLogger(__name__)


class PromptInUse(Exception):
    """A prompt jobs used can't be deleted: it is part of their provenance."""


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def record_use(
    db: Session,
    text: str,
    model: str,
    kind: str,
    job_id: str | None = None,
    user_id: str | None = None,
) -> Prompt:
    """Add one use of `text`, creating its library entry the first time."""
    now = datetime.now(UTC)
    sha = prompt_sha256(text)
    prompt = db.get(Prompt, sha)
    if prompt is None:
        prompt = Prompt(
            sha256=sha,
            text=text,
            tags=[],
            starred=False,
            hidden=False,
            first_used_at=now,
            last_used_at=now,
            first_user_id=user_id,
        )
        db.add(prompt)
    else:
        prompt.last_used_at = now
    db.add(
        PromptUse(
            prompt_sha256=sha,
            model=model,
            kind=kind,
            job_id=job_id,
            user_id=user_id,
            used_at=now,
        )
    )
    db.commit()
    return prompt


_tables_checked = False


def record_use_quietly(text: str, model: str, kind: str, **kwargs: Any) -> None:
    """record_use in its own session, never raising: a library hiccup must not
    fail the job or preview that used the prompt."""
    try:
        from .database.database import SessionLocal, engine

        global _tables_checked
        if not _tables_checked:
            # `videoannotator process` runs without the server, which is
            # what creates tables; an older database may not have these yet.
            Prompt.metadata.create_all(
                bind=engine, tables=[Prompt.__table__, PromptUse.__table__]
            )
            _tables_checked = True
        with SessionLocal() as db:
            record_use(db, text, model, kind, **kwargs)
    except Exception as e:
        logger.warning(f"Could not record the prompt in the library: {e}")


def summary(db: Session, prompt: Prompt) -> dict[str, Any]:
    """A prompt with what its uses say: models, jobs, use count."""
    uses = prompt.uses
    return {
        "sha256": prompt.sha256,
        "text": prompt.text,
        "name": prompt.name,
        "tags": prompt.tags or [],
        "starred": prompt.starred,
        "hidden": prompt.hidden,
        "first_used_at": prompt.first_used_at,
        "last_used_at": prompt.last_used_at,
        "first_user_id": prompt.first_user_id,
        "updated_at": prompt.updated_at,
        "updated_by": prompt.updated_by,
        "models": sorted({u.model for u in uses}),
        "job_ids": sorted({u.job_id for u in uses if u.job_id}),
        "use_count": len(uses),
    }


def search(
    db: Session,
    q: str | None = None,
    model: str | None = None,
    tag: str | None = None,
    include_hidden: bool = False,
    limit: int = 200,
) -> list[Prompt]:
    """Starred first, then most recently used."""
    query = db.query(Prompt)
    if not include_hidden:
        query = query.filter(Prompt.hidden.is_(False))
    if q:
        like = f"%{q.lower()}%"
        query = query.filter(
            or_(func.lower(Prompt.text).like(like), func.lower(Prompt.name).like(like))
        )
    if model:
        query = query.filter(
            Prompt.uses.any(func.lower(PromptUse.model) == model.lower())
        )
    prompts = query.order_by(Prompt.starred.desc(), Prompt.last_used_at.desc()).all()
    if tag:  # tags are a JSON list: filtered here, portably
        prompts = [p for p in prompts if tag in (p.tags or [])]
    return prompts[:limit]


def get(db: Session, sha: str) -> Prompt | None:
    """By full hash, or by an unambiguous prefix (for the CLI)."""
    prompt = db.get(Prompt, sha)
    if prompt is not None or len(sha) >= 64:
        return prompt
    matches = db.query(Prompt).filter(Prompt.sha256.like(f"{sha}%")).limit(2).all()
    return matches[0] if len(matches) == 1 else None


def update(
    db: Session,
    prompt: Prompt,
    user_id: str | None,
    name: str | None = None,
    tags: list[str] | None = None,
    starred: bool | None = None,
    hidden: bool | None = None,
) -> Prompt:
    """Change what people call or think of a prompt; never its text."""
    if name is not None:
        prompt.name = name.strip() or None
    if tags is not None:
        prompt.tags = sorted({t.strip() for t in tags if t.strip()})
    if starred is not None:
        prompt.starred = starred
    if hidden is not None:
        prompt.hidden = hidden
    prompt.updated_at = datetime.now(UTC)
    prompt.updated_by = user_id
    db.commit()
    return prompt


def delete(db: Session, prompt: Prompt) -> None:
    if any(u.job_id for u in prompt.uses):
        raise PromptInUse(prompt.sha256)
    db.delete(prompt)
    db.commit()
