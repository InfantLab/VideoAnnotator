"""Saved-dataset endpoints for VideoAnnotator API (spec 007).

Server-side, DB-backed, metadata-only (filename+size manifest, never the
video files themselves) so a researcher can save a named set of videos once
and reuse it on every future job submission instead of re-picking files from
a browser dialog each time. See specs/007-datasets-and-presets/spec.md.
"""

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Path
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ...database.crud import SavedDatasetCRUD
from ...database.database import get_db
from ..errors import APIError
from ..middleware.auth import validate_api_key

logger = logging.getLogger("videoannotator.api")

router = APIRouter()


class VideoManifestEntry(BaseModel):
    """One remembered video's identity within a saved dataset (FR-002)."""

    filename: str
    size_bytes: int
    last_seen_at: datetime | None = None


class DatasetCreateRequest(BaseModel):
    """Body for `POST /datasets`. Also accepts a previously-exported
    dataset's `GET` response verbatim (FR-006) -- `id`/`owner_user_id`/
    timestamp fields present in an export are simply ignored here; a fresh
    id and the current caller as owner are always assigned."""

    name: str
    description: str | None = None
    video_manifest: list[VideoManifestEntry] = []


class DatasetUpdateRequest(BaseModel):
    """Body for `PUT /datasets/{id}`. Only provided fields are changed."""

    name: str | None = None
    description: str | None = None
    video_manifest: list[VideoManifestEntry] | None = None


class DatasetResponse(BaseModel):
    """A saved dataset, including its manifest -- this exact shape is also
    what export produces and import (`POST`) accepts (FR-006)."""

    id: str
    name: str
    description: str | None = None
    owner_user_id: str
    video_manifest: list[VideoManifestEntry]
    created_at: datetime
    updated_at: datetime | None = None
    last_used_at: datetime | None = None


class DatasetListResponse(BaseModel):
    datasets: list[DatasetResponse]
    total: int


def _to_response(dataset: Any) -> DatasetResponse:
    return DatasetResponse(
        id=str(dataset.id),
        name=dataset.name,
        description=dataset.description,
        owner_user_id=str(dataset.owner_user_id),
        video_manifest=dataset.video_manifest or [],
        created_at=dataset.created_at,
        updated_at=dataset.updated_at,
        last_used_at=dataset.last_used_at,
    )


def _require_owner_identity(user: dict[str, Any] | None) -> str:
    """Datasets always have a non-null owner (FR-001), unlike jobs which
    allow anonymous submission -- so a save/modify call needs a resolved
    identity even when the server's AUTH_REQUIRED toggle is off."""
    if user is None or not user.get("id"):
        raise APIError(
            status_code=400,
            code="OWNER_IDENTITY_REQUIRED",
            message="Saving or modifying a dataset requires an authenticated identity.",
            hint="Provide a valid API key (Authorization: Bearer ...).",
        )
    return str(user["id"])


def _get_or_404(db: Session, dataset_id: str):
    dataset = SavedDatasetCRUD.get_by_id(db, dataset_id)
    if dataset is None:
        raise APIError(
            status_code=404,
            code="DATASET_NOT_FOUND",
            message=f"Dataset '{dataset_id}' not found",
        )
    return dataset


def _require_owner_or_admin(dataset, owner_id: str, is_admin: bool) -> None:
    if str(dataset.owner_user_id) != owner_id and not is_admin:
        raise APIError(
            status_code=403,
            code="NOT_DATASET_OWNER",
            message="Only the owner or an administrator may modify or delete this dataset.",
        )


@router.get(
    "",
    include_in_schema=False,
)
@router.get(
    "/",
    response_model=DatasetListResponse,
    summary="List saved datasets",
    description=(
        "Shared-read: any authenticated user sees every saved dataset on this "
        "server, matching existing job-listing visibility (FR-005)."
    ),
)
async def list_datasets(
    db: Session = Depends(get_db),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> DatasetListResponse:
    """List all saved datasets."""
    datasets = SavedDatasetCRUD.list_all(db)
    return DatasetListResponse(
        datasets=[_to_response(d) for d in datasets], total=len(datasets)
    )


@router.post(
    "",
    include_in_schema=False,
)
@router.post(
    "/",
    response_model=DatasetResponse,
    status_code=201,
    summary="Save a new dataset (also used for import)",
    description=(
        "Creates a saved dataset owned by the caller. Also serves as the "
        "import path for a previously-exported dataset definition (FR-006) -- "
        "POST the exact body a prior GET returned; a fresh id and the current "
        "caller as owner are assigned. A name collision with one the caller "
        "already owns is rejected with 409, never silently overwritten (FR-007)."
    ),
)
async def create_dataset(
    request: DatasetCreateRequest,
    db: Session = Depends(get_db),
    user: dict[str, Any] | None = Depends(validate_api_key),
) -> DatasetResponse:
    """Save a new dataset."""
    owner_id = _require_owner_identity(user)
    dataset = SavedDatasetCRUD.create(
        db,
        owner_user_id=owner_id,
        name=request.name,
        description=request.description,
        video_manifest=[e.model_dump(mode="json") for e in request.video_manifest],
    )
    if dataset is None:
        raise APIError(
            status_code=409,
            code="DATASET_NAME_CONFLICT",
            message=f"You already have a dataset named '{request.name}'.",
            hint="Choose a different name, or update the existing dataset instead.",
        )
    return _to_response(dataset)


@router.get(
    "/{dataset_id}",
    response_model=DatasetResponse,
    summary="Get a saved dataset",
)
async def get_dataset(
    dataset_id: str = Path(..., description="The saved dataset's id"),
    db: Session = Depends(get_db),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> DatasetResponse:
    """Retrieve a single saved dataset by id."""
    dataset = _get_or_404(db, dataset_id)
    return _to_response(dataset)


@router.put(
    "/{dataset_id}",
    response_model=DatasetResponse,
    summary="Update a saved dataset",
    description="Owner or administrator only.",
)
async def update_dataset(
    request: DatasetUpdateRequest,
    dataset_id: str = Path(..., description="The saved dataset's id"),
    db: Session = Depends(get_db),
    user: dict[str, Any] | None = Depends(validate_api_key),
) -> DatasetResponse:
    """Update a saved dataset's name, description, and/or manifest."""
    owner_id = _require_owner_identity(user)
    dataset = _get_or_404(db, dataset_id)
    _require_owner_or_admin(
        dataset, owner_id, bool(user and user.get("is_admin", False))
    )
    updated = SavedDatasetCRUD.update(
        db,
        dataset_id,
        name=request.name,
        description=request.description,
        video_manifest=(
            [e.model_dump(mode="json") for e in request.video_manifest]
            if request.video_manifest is not None
            else None
        ),
    )
    if updated is None:
        raise APIError(
            status_code=409,
            code="DATASET_NAME_CONFLICT",
            message=f"You already have a dataset named '{request.name}'.",
        )
    return _to_response(updated)


@router.delete(
    "/{dataset_id}",
    status_code=204,
    summary="Delete a saved dataset",
    description=(
        "Owner or administrator only. Does not affect the historical record "
        "of any job previously submitted using this dataset (FR-008)."
    ),
)
async def delete_dataset(
    dataset_id: str = Path(..., description="The saved dataset's id"),
    db: Session = Depends(get_db),
    user: dict[str, Any] | None = Depends(validate_api_key),
) -> None:
    """Delete a saved dataset."""
    owner_id = _require_owner_identity(user)
    dataset = _get_or_404(db, dataset_id)
    _require_owner_or_admin(
        dataset, owner_id, bool(user and user.get("is_admin", False))
    )
    SavedDatasetCRUD.delete(db, dataset_id)
