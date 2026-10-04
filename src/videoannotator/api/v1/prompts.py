"""Prompt library endpoints (spec 020): every VLM prompt that ran, once each.

Shared by everyone on the server, like presets and datasets. Prompts are added
by running them (jobs, previews), never through this API; here they are
searched, named, starred, tagged, hidden, and (when no job used them) deleted.
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Path, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ... import prompt_library
from ...database.database import get_db
from ..errors import APIError
from ..middleware.auth import validate_api_key

router = APIRouter()


class PromptResponse(BaseModel):
    sha256: str
    text: str
    name: str | None = None
    tags: list[str] = []
    starred: bool = False
    hidden: bool = False
    first_used_at: datetime
    last_used_at: datetime
    first_user_id: str | None = None
    updated_at: datetime | None = None
    updated_by: str | None = None
    models: list[str] = []
    job_ids: list[str] = []
    use_count: int = 0


class PromptListResponse(BaseModel):
    prompts: list[PromptResponse]
    total: int


class PromptUpdateRequest(BaseModel):
    """Only provided fields change. The text never does: a different text is a
    different prompt."""

    name: str | None = None
    tags: list[str] | None = None
    starred: bool | None = None
    hidden: bool | None = None


def _get_or_404(db: Session, sha256: str) -> Any:
    prompt = prompt_library.get(db, sha256)
    if prompt is None:
        raise APIError(
            status_code=404,
            code="PROMPT_NOT_FOUND",
            message=f"No prompt with hash '{sha256}'",
            hint="GET /api/v1/prompts lists them; prompts are added by running them.",
        )
    return prompt


def _user_id(user: dict[str, Any] | None) -> str | None:
    return str(user["id"]) if user and user.get("id") else None


@router.get("", include_in_schema=False)
@router.get(
    "/",
    response_model=PromptListResponse,
    summary="Search the prompt library",
    description="Starred first, then most recently used. `q` matches text and name.",
)
async def list_prompts(
    q: str | None = Query(None, description="Words in the text or name"),
    model: str | None = Query(None, description="Only prompts run with this model"),
    tag: str | None = Query(None),
    include_hidden: bool = Query(False),
    db: Session = Depends(get_db),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> PromptListResponse:
    prompts = prompt_library.search(db, q, model, tag, include_hidden)
    return PromptListResponse(
        prompts=[PromptResponse(**prompt_library.summary(db, p)) for p in prompts],
        total=len(prompts),
    )


@router.get("/{sha256}", response_model=PromptResponse, summary="Get a prompt")
async def get_prompt(
    sha256: str = Path(..., description="SHA-256 of the text, or a unique prefix"),
    db: Session = Depends(get_db),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> PromptResponse:
    return PromptResponse(**prompt_library.summary(db, _get_or_404(db, sha256)))


@router.put(
    "/{sha256}",
    response_model=PromptResponse,
    summary="Name, tag, star or hide a prompt",
)
async def update_prompt(
    request: PromptUpdateRequest,
    sha256: str = Path(...),
    db: Session = Depends(get_db),
    user: dict[str, Any] | None = Depends(validate_api_key),
) -> PromptResponse:
    prompt = prompt_library.update(
        db,
        _get_or_404(db, sha256),
        _user_id(user),
        name=request.name,
        tags=request.tags,
        starred=request.starred,
        hidden=request.hidden,
    )
    return PromptResponse(**prompt_library.summary(db, prompt))


@router.delete(
    "/{sha256}",
    status_code=204,
    summary="Delete a prompt no job used",
    description="409 PROMPT_USED_BY_JOBS once any job ran it: it is part of their "
    "provenance. Hide it instead.",
)
async def delete_prompt(
    sha256: str = Path(...),
    db: Session = Depends(get_db),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> None:
    try:
        prompt_library.delete(db, _get_or_404(db, sha256))
    except prompt_library.PromptInUse as e:
        raise APIError(
            status_code=409,
            code="PROMPT_USED_BY_JOBS",
            message="Jobs used this prompt, so it stays in the library.",
            hint="Hide it instead (PUT with hidden=true).",
        ) from e
