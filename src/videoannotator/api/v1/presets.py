"""Saved pipeline-preset endpoints for VideoAnnotator API (spec 007).

Lets a researcher save a named pipeline selection + configuration (including
a multi-line VLM prompt, reproduced verbatim) once and reuse it on every
future job submission instead of re-selecting pipelines and re-typing
config each time. See specs/007-datasets-and-presets/spec.md.
"""

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Path
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ...database.crud import SavedPipelinePresetCRUD
from ...database.database import get_db
from ...registry.pipeline_loader import extras_available
from ...registry.pipeline_registry import get_registry
from ..errors import APIError
from ..middleware.auth import validate_api_key

logger = logging.getLogger("videoannotator.api")

router = APIRouter()


class PresetCreateRequest(BaseModel):
    """Body for `POST /presets`. Also accepts a previously-exported preset's
    `GET` response verbatim (FR-006) -- `id`/`owner_user_id`/timestamp/
    `unavailable_pipelines` fields present in an export are simply ignored
    here; a fresh id and the current caller as owner are always assigned."""

    name: str
    description: str | None = None
    selected_pipelines: list[str] = []
    config: dict[str, Any] = {}
    tags: dict[str, Any] = {}


class PresetUpdateRequest(BaseModel):
    """Body for `PUT /presets/{id}`. Only provided fields are changed."""

    name: str | None = None
    description: str | None = None
    selected_pipelines: list[str] | None = None
    config: dict[str, Any] | None = None
    tags: dict[str, Any] | None = None


class PresetResponse(BaseModel):
    """A saved preset. `selected_pipelines`/`config` mirror a job
    submission's own fields exactly, so applying a preset is a direct copy
    into a new job submission with no translation needed."""

    id: str
    name: str
    description: str | None = None
    owner_user_id: str
    selected_pipelines: list[str]
    config: dict[str, Any]
    tags: dict[str, Any]
    created_at: datetime
    updated_at: datetime | None = None
    last_used_at: datetime | None = None
    unavailable_pipelines: list[str] = []


class PresetListResponse(BaseModel):
    presets: list[PresetResponse]
    total: int


def _unavailable_pipelines(selected_pipelines: list[str]) -> list[str]:
    """Which of a preset's referenced pipelines are currently unavailable on
    this server (FR-009) -- unknown name, or a known one whose extras group
    isn't installed. Retrieval must still succeed either way (edge case)."""
    reg = get_registry()
    reg.load()  # idempotent
    unavailable: list[str] = []
    for name in selected_pipelines:
        meta = reg.get(name)
        if meta is None or not extras_available(meta.requires_extras):
            unavailable.append(name)
    return unavailable


def _to_response(preset: Any) -> PresetResponse:
    return PresetResponse(
        id=str(preset.id),
        name=preset.name,
        description=preset.description,
        owner_user_id=str(preset.owner_user_id),
        selected_pipelines=preset.selected_pipelines or [],
        config=preset.config or {},
        tags=preset.tags or {},
        created_at=preset.created_at,
        updated_at=preset.updated_at,
        last_used_at=preset.last_used_at,
        unavailable_pipelines=_unavailable_pipelines(preset.selected_pipelines or []),
    )


def _require_owner_identity(user: dict[str, Any] | None) -> str:
    """Presets always have a non-null owner (FR-003), unlike jobs which
    allow anonymous submission -- so a save/modify call needs a resolved
    identity even when the server's AUTH_REQUIRED toggle is off."""
    if user is None or not user.get("id"):
        raise APIError(
            status_code=400,
            code="OWNER_IDENTITY_REQUIRED",
            message="Saving or modifying a preset requires an authenticated identity.",
            hint="Provide a valid API key (Authorization: Bearer ...).",
        )
    return str(user["id"])


def _get_or_404(db: Session, preset_id: str):
    preset = SavedPipelinePresetCRUD.get_by_id(db, preset_id)
    if preset is None:
        raise APIError(
            status_code=404,
            code="PRESET_NOT_FOUND",
            message=f"Preset '{preset_id}' not found",
        )
    return preset


def _require_owner_or_admin(preset, owner_id: str, is_admin: bool) -> None:
    if str(preset.owner_user_id) != owner_id and not is_admin:
        raise APIError(
            status_code=403,
            code="NOT_PRESET_OWNER",
            message="Only the owner or an administrator may modify or delete this preset.",
        )


@router.get(
    "",
    include_in_schema=False,
)
@router.get(
    "/",
    response_model=PresetListResponse,
    summary="List saved pipeline presets",
    description=(
        "Shared-read: any authenticated user sees every saved preset on this "
        "server, matching existing job-listing visibility (FR-005)."
    ),
)
async def list_presets(
    db: Session = Depends(get_db),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> PresetListResponse:
    """List all saved pipeline presets."""
    presets = SavedPipelinePresetCRUD.list_all(db)
    return PresetListResponse(
        presets=[_to_response(p) for p in presets], total=len(presets)
    )


@router.post(
    "",
    include_in_schema=False,
)
@router.post(
    "/",
    response_model=PresetResponse,
    status_code=201,
    summary="Save a new pipeline preset (also used for import)",
    description=(
        "Creates a saved preset owned by the caller. Also serves as the "
        "import path for a previously-exported preset definition (FR-006) -- "
        "POST the exact body a prior GET returned; a fresh id and the current "
        "caller as owner are assigned. A name collision with one the caller "
        "already owns is rejected with 409, never silently overwritten (FR-007)."
    ),
)
async def create_preset(
    request: PresetCreateRequest,
    db: Session = Depends(get_db),
    user: dict[str, Any] | None = Depends(validate_api_key),
) -> PresetResponse:
    """Save a new pipeline preset."""
    owner_id = _require_owner_identity(user)
    preset = SavedPipelinePresetCRUD.create(
        db,
        owner_user_id=owner_id,
        name=request.name,
        description=request.description,
        selected_pipelines=request.selected_pipelines,
        config=request.config,
        tags=request.tags,
    )
    if preset is None:
        raise APIError(
            status_code=409,
            code="PRESET_NAME_CONFLICT",
            message=f"You already have a preset named '{request.name}'.",
            hint="Choose a different name, or update the existing preset instead.",
        )
    return _to_response(preset)


@router.get(
    "/{preset_id}",
    response_model=PresetResponse,
    summary="Get a saved pipeline preset",
    description=(
        "Retrieval always succeeds even if a referenced pipeline is no "
        "longer available on this server -- see `unavailable_pipelines` "
        "(FR-009)."
    ),
)
async def get_preset(
    preset_id: str = Path(..., description="The saved preset's id"),
    db: Session = Depends(get_db),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> PresetResponse:
    """Retrieve a single saved preset by id."""
    preset = _get_or_404(db, preset_id)
    return _to_response(preset)


@router.put(
    "/{preset_id}",
    response_model=PresetResponse,
    summary="Update a saved pipeline preset",
    description="Owner or administrator only.",
)
async def update_preset(
    request: PresetUpdateRequest,
    preset_id: str = Path(..., description="The saved preset's id"),
    db: Session = Depends(get_db),
    user: dict[str, Any] | None = Depends(validate_api_key),
) -> PresetResponse:
    """Update a saved preset's name, description, pipelines, config, and/or tags."""
    owner_id = _require_owner_identity(user)
    preset = _get_or_404(db, preset_id)
    _require_owner_or_admin(
        preset, owner_id, bool(user and user.get("is_admin", False))
    )
    updated = SavedPipelinePresetCRUD.update(
        db,
        preset_id,
        name=request.name,
        description=request.description,
        selected_pipelines=request.selected_pipelines,
        config=request.config,
        tags=request.tags,
    )
    if updated is None:
        raise APIError(
            status_code=409,
            code="PRESET_NAME_CONFLICT",
            message=f"You already have a preset named '{request.name}'.",
        )
    return _to_response(updated)


@router.delete(
    "/{preset_id}",
    status_code=204,
    summary="Delete a saved pipeline preset",
    description=(
        "Owner or administrator only. Does not affect the historical record "
        "of any job previously submitted using this preset (FR-008)."
    ),
)
async def delete_preset(
    preset_id: str = Path(..., description="The saved preset's id"),
    db: Session = Depends(get_db),
    user: dict[str, Any] | None = Depends(validate_api_key),
) -> None:
    """Delete a saved pipeline preset."""
    owner_id = _require_owner_identity(user)
    preset = _get_or_404(db, preset_id)
    _require_owner_or_admin(
        preset, owner_id, bool(user and user.get("is_admin", False))
    )
    SavedPipelinePresetCRUD.delete(db, preset_id)
