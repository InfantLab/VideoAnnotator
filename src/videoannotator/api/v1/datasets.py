"""Saved-dataset endpoints for VideoAnnotator API (spec 007).

Server-side, DB-backed, metadata-only (filename+size manifest, never the
video files themselves) so a researcher can save a named set of videos once
and reuse it on every future job submission instead of re-picking files from
a browser dialog each time. See specs/007-datasets-and-presets/spec.md.

A dataset saved from a browser upload has no folder the server can see, but
its videos were uploaded with the jobs that ran on them, and each job keeps
its copy. `/stored-videos` and `/run` find and reuse those copies, so running
the dataset again needs no folder and no upload while they exist.
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path as FilePath
from typing import Any

from fastapi import APIRouter, Depends, Path
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...batch.types import BatchJob
from ...database.crud import SavedDatasetCRUD
from ...database.database import get_db
from ...storage.base import StorageBackend
from ..database import get_storage_backend
from ..errors import APIError
from ..middleware.auth import validate_api_key

logger = logging.getLogger("videoannotator.api")

router = APIRouter()


class VideoManifestEntry(BaseModel):
    """One remembered video's identity within a saved dataset (FR-002)."""

    filename: str
    size_bytes: int
    last_seen_at: datetime | None = None
    # Path within the dataset's folder, so same-named files in subfolders stay
    # distinct (spec 018). Absent in datasets saved before it.
    relative_path: str | None = None


class DatasetCreateRequest(BaseModel):
    """Body for `POST /datasets`. Also accepts a previously-exported
    dataset's `GET` response verbatim (FR-006) -- `id`/`owner_user_id`/
    timestamp fields present in an export are simply ignored here; a fresh
    id and the current caller as owner are always assigned."""

    name: str
    description: str | None = None
    video_manifest: list[VideoManifestEntry] = []
    server_folder: str | None = None
    server_folder_recursive: bool = False
    server_selection: bool = Field(
        default=False,
        description="The manifest is a chosen subset of server_folder (spec 022): "
        "using the dataset runs exactly those videos, and files added to the "
        "folder are not differences",
    )


class DatasetUpdateRequest(BaseModel):
    """Body for `PUT /datasets/{id}`. Only provided fields are changed."""

    name: str | None = None
    description: str | None = None
    video_manifest: list[VideoManifestEntry] | None = None
    server_selection: bool | None = None


class DatasetResponse(BaseModel):
    """A saved dataset, including its manifest -- this exact shape is also
    what export produces and import (`POST`) accepts (FR-006)."""

    id: str
    name: str
    description: str | None = None
    owner_user_id: str
    owner_name: str | None = None
    video_manifest: list[VideoManifestEntry]
    server_folder: str | None = None
    server_folder_recursive: bool = False
    server_selection: bool = False
    created_at: datetime
    updated_at: datetime | None = None
    last_used_at: datetime | None = None


class DatasetListResponse(BaseModel):
    datasets: list[DatasetResponse]
    total: int


def _owner_name(dataset: Any) -> str | None:
    owner = getattr(dataset, "owner", None)
    return (owner.username or owner.email) if owner is not None else None


def _to_response(dataset: Any) -> DatasetResponse:
    return DatasetResponse(
        id=str(dataset.id),
        name=dataset.name,
        description=dataset.description,
        owner_user_id=str(dataset.owner_user_id),
        owner_name=_owner_name(dataset),
        video_manifest=dataset.video_manifest or [],
        server_folder=dataset.server_folder,
        server_folder_recursive=bool(dataset.server_folder_recursive),
        server_selection=bool(getattr(dataset, "server_selection", False)),
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
        server_folder=request.server_folder,
        server_folder_recursive=request.server_folder_recursive,
        server_selection=request.server_selection,
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
        server_selection=request.server_selection,
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


def get_storage() -> StorageBackend:
    """Storage backend for finding the jobs that hold a dataset's videos."""
    return get_storage_backend()


class StoredVideo(BaseModel):
    filename: str
    size_bytes: int | None
    job_id: str | None = Field(
        description="A job whose stored copy of this video can be reused, or "
        "null when no job still has it"
    )


class StoredVideosResponse(BaseModel):
    dataset_id: str
    videos: list[StoredVideo]
    stored: int
    missing: int


class DatasetRunRequest(BaseModel):
    selected_pipelines: list[str] | None = None
    config: dict[str, Any] | None = None
    batch_name: str | None = None


class DatasetRunSkipped(BaseModel):
    filename: str
    reason: str


class DatasetRunResponse(BaseModel):
    batch_id: str
    batch_name: str | None
    created: list[str]
    skipped: list[DatasetRunSkipped]


def _stored_copies(
    dataset: Any, storage: StorageBackend
) -> list[tuple[dict[str, Any], BatchJob | None]]:
    """Each manifest entry with a job that still stores that video, if any.

    A copy matches on filename and, where the manifest has one, size. Jobs
    submitted from this dataset are preferred, then the newest.
    """
    by_name: dict[str, list[BatchJob]] = {}
    for job_id in storage.list_jobs():
        job = storage.load_job_metadata(job_id)
        if job is None or not job.video_path:
            continue
        by_name.setdefault(FilePath(str(job.video_path)).name, []).append(job)

    def rank(job: BatchJob) -> tuple[bool, str]:
        created = job.created_at.isoformat() if job.created_at else ""
        return (job.dataset_id == dataset.id, created)

    result = []
    for entry in dataset.video_manifest or []:
        found = None
        for job in sorted(by_name.get(entry["filename"], []), key=rank, reverse=True):
            video = FilePath(str(job.video_path))
            try:
                if not video.is_file():
                    continue
                if entry.get("size_bytes") is not None and (
                    video.stat().st_size != entry["size_bytes"]
                ):
                    continue
            except OSError:
                continue
            found = job
            break
        result.append((entry, found))
    return result


@router.get(
    "/{dataset_id}/stored-videos",
    response_model=StoredVideosResponse,
    summary="Which of a dataset's videos the server still has",
    description=(
        "For each video in the dataset, a job whose stored copy (uploaded with "
        "that job) can be reused, matched by filename and size. Null when no "
        "job has it any more, e.g. its jobs were deleted."
    ),
)
async def stored_videos(
    dataset_id: str = Path(..., description="The saved dataset's id"),
    db: Session = Depends(get_db),
    storage: StorageBackend = Depends(get_storage),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> StoredVideosResponse:
    """List a dataset's videos with the stored copy of each, if any."""
    dataset = _get_or_404(db, dataset_id)
    videos = [
        StoredVideo(
            filename=entry["filename"],
            size_bytes=entry.get("size_bytes"),
            job_id=job.job_id if job else None,
        )
        for entry, job in _stored_copies(dataset, storage)
    ]
    stored = sum(1 for v in videos if v.job_id)
    return StoredVideosResponse(
        dataset_id=dataset_id,
        videos=videos,
        stored=stored,
        missing=len(videos) - stored,
    )


@router.post(
    "/{dataset_id}/run",
    response_model=DatasetRunResponse,
    status_code=201,
    summary="Run a dataset from the server's stored copies",
    description=(
        "Creates one job per dataset video the server still stores (see "
        "`/stored-videos`), as one batch, with no upload: each video is "
        "hard-linked from the job that has it. Videos with no stored copy are "
        "listed in `skipped`. 422 `DATASET_NOT_STORED` when none are stored."
    ),
)
async def run_dataset(
    request: DatasetRunRequest,
    dataset_id: str = Path(..., description="The saved dataset's id"),
    db: Session = Depends(get_db),
    storage: StorageBackend = Depends(get_storage),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> DatasetRunResponse:
    """Create a batch of jobs on a dataset's stored videos."""
    from .jobs import job_from_stored_video, validate_pipeline_selection

    dataset = _get_or_404(db, dataset_id)
    validate_pipeline_selection(request.selected_pipelines, request.config)
    copies = _stored_copies(dataset, storage)
    if not any(job for _, job in copies):
        raise APIError(
            status_code=422,
            code="DATASET_NOT_STORED",
            message=f"The server no longer has any of the videos of '{dataset.name}'.",
            hint="Choose the folder they are in instead.",
        )

    batch_id = str(uuid.uuid4())
    batch_name = request.batch_name or dataset.name
    created: list[str] = []
    skipped: list[DatasetRunSkipped] = []
    for entry, original in copies:
        if original is None:
            skipped.append(
                DatasetRunSkipped(
                    filename=entry["filename"], reason="no longer stored on the server"
                )
            )
            continue
        try:
            job = job_from_stored_video(
                original,
                storage,
                request.selected_pipelines,
                request.config,
                batch_id=batch_id,
                batch_name=batch_name,
                dataset_id=dataset_id,
            )
            created.append(job.job_id)
        except Exception as e:  # one bad file shouldn't sink the batch
            logger.warning(
                f"[DATASET] Could not create job for {entry['filename']}: {e}"
            )
            skipped.append(DatasetRunSkipped(filename=entry["filename"], reason=str(e)))

    try:
        SavedDatasetCRUD.touch_last_used(db, dataset_id)
    except Exception as e:
        logger.debug(f"Could not touch last_used_at for dataset {dataset_id}: {e}")

    return DatasetRunResponse(
        batch_id=batch_id, batch_name=batch_name, created=created, skipped=skipped
    )
