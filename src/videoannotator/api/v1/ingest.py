"""Create jobs from videos already on the server's filesystem.

`POST /api/v1/jobs` takes one video per request as a multipart upload. That is
fine for a clip and miserable for a corpus: forty videos means forty uploads,
each streaming gigabytes through the browser, with the tab held open and no
resume. On the overwhelmingly common deployment -- a researcher running this
server on their own machine -- those bytes are already on the disk the server
can see, and copying them through HTTP to land somewhere else on the same disk
achieves nothing.

So this module creates jobs that *reference* videos where they already are. No
upload, no copy: one request turns a folder into a batch. Job outputs still go
to each job's own storage directory, so deleting a job never touches the
researcher's original video.

Because this reads the server's filesystem on a caller's instruction, it is
gated three ways, each independent:

1. **Admin only** -- same bar as the extras-install endpoints, the existing
   precedent for a privileged operation.
2. **Loopback callers only** -- the feature exists for "the server is on my
   machine"; a remote caller has no business naming local paths, and for them
   upload remains the only route.
3. **Inside an allowed root** -- `VIDEOANNOTATOR_INGEST_ROOTS`, defaulting to
   the server user's home directory. Paths are fully resolved (symlinks
   included) before the check, so `..` and symlink escapes cannot get out.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...batch.types import BatchJob, JobStatus
from ...config_env import INGEST_ROOTS
from ...database.crud import SavedDatasetCRUD
from ...database.database import get_db
from ...storage.base import StorageBackend
from ...storage.manager import get_storage_provider
from ..database import get_storage_backend
from ..errors import APIError
from ..middleware.auth import require_admin
from .jobs import extract_video_metadata, validate_pipeline_selection

logger = logging.getLogger("videoannotator.api")

router = APIRouter()

# Matches the extensions batch_orchestrator.py already discovers, so the CLI
# and the API agree on what counts as a video.
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm")

# A directory listing is for a human choosing a folder, not a data export.
MAX_LISTING_ENTRIES = 500


def get_storage() -> StorageBackend:
    """Get storage backend (mirrors jobs.py)."""
    return get_storage_backend()


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def allowed_roots() -> list[Path]:
    """Directories ingest may read from.

    Defaults to the server user's home directory: on a single-user research
    install that is where the data lives, and it keeps the feature usable
    without configuration while still being a real boundary.
    """
    configured = [entry for entry in INGEST_ROOTS.split(os.pathsep) if entry.strip()]
    if not configured:
        return [Path.home().resolve()]

    roots = []
    for entry in configured:
        try:
            roots.append(Path(entry.strip()).expanduser().resolve())
        except OSError as e:  # pragma: no cover - unresolvable configured path
            logger.warning(f"[INGEST] Ignoring unusable configured root {entry!r}: {e}")
    return roots


def require_local_caller(request: Request) -> None:
    """Reject callers that aren't on this machine.

    Naming a server-side path only makes sense when the caller *is* the server's
    user. A remote client still has `POST /api/v1/jobs` and its upload.
    """
    client = request.client
    host = client.host if client else None

    is_loopback = False
    if host:
        try:
            is_loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            # A non-address host (e.g. a unix socket's "testclient") is treated
            # as local: it cannot have come over the network.
            is_loopback = host in ("testclient", "localhost")

    if not is_loopback:
        raise APIError(
            status_code=403,
            code="INGEST_REMOTE_CALLER",
            message="Folder ingest is only available to clients running on the same machine as the server.",
            hint="Upload the videos through POST /api/v1/jobs instead, or run the viewer on the server's own machine.",
        )


def resolve_within_roots(raw_path: str) -> Path:
    """Resolve `raw_path` and confirm it sits inside an allowed root.

    Resolution happens first and follows symlinks, so neither `..` nor a
    symlink pointing outside a root can escape: the check is applied to the
    real location, not the string the caller sent.
    """
    if not raw_path or not raw_path.strip():
        raise APIError(
            status_code=422,
            code="INGEST_PATH_REQUIRED",
            message="A folder path is required.",
            hint="Use GET /api/v1/ingest/browse to discover readable folders.",
        )

    try:
        resolved = Path(raw_path.strip()).expanduser().resolve()
    except (OSError, RuntimeError) as e:
        raise APIError(
            status_code=422,
            code="INGEST_PATH_INVALID",
            message=f"Could not resolve path: {raw_path}",
            hint="Check the path is spelled correctly and reachable from the server.",
        ) from e

    roots = allowed_roots()
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise APIError(
            status_code=403,
            code="INGEST_PATH_NOT_ALLOWED",
            message=f"Path is outside the folders this server may read from: {resolved}",
            hint=(
                "Allowed: "
                + ", ".join(str(r) for r in roots)
                + ". Set VIDEOANNOTATOR_INGEST_ROOTS (os.pathsep-separated) to allow others."
            ),
        )

    if not resolved.exists():
        raise APIError(
            status_code=404,
            code="INGEST_PATH_NOT_FOUND",
            message=f"No such folder on the server: {resolved}",
            hint="Check the folder still exists and hasn't been moved or renamed.",
        )

    return resolved


# ---------------------------------------------------------------------------
# Browsing
# ---------------------------------------------------------------------------


class IngestDirectory(BaseModel):
    name: str
    path: str
    video_count: int


class IngestVideo(BaseModel):
    name: str
    path: str
    size_bytes: int | None = None


class IngestBrowseResponse(BaseModel):
    path: str | None = Field(
        default=None, description="Folder listed, or null when listing the roots"
    )
    parent: str | None = Field(
        default=None, description="Parent folder, or null at a root"
    )
    roots: list[str]
    directories: list[IngestDirectory]
    videos: list[IngestVideo]
    video_count: int = Field(description="Videos directly in this folder")
    truncated: bool = False


def _count_videos(directory: Path) -> int:
    """Videos directly inside `directory`. Not recursive, and never raises."""
    try:
        return sum(
            1
            for entry in directory.iterdir()
            if entry.is_file() and entry.suffix.lower() in VIDEO_EXTENSIONS
        )
    except OSError:
        # An unreadable folder shows as empty rather than breaking the listing.
        return 0


def find_videos(directory: Path, recursive: bool = False) -> list[Path]:
    """Video files in `directory`, sorted by name for a stable job order."""
    try:
        entries = directory.rglob("*") if recursive else directory.iterdir()
        videos = [
            entry
            for entry in entries
            if entry.is_file() and entry.suffix.lower() in VIDEO_EXTENSIONS
        ]
    except OSError as e:
        raise APIError(
            status_code=403,
            code="INGEST_PATH_UNREADABLE",
            message=f"Could not read folder: {directory}",
            hint=f"Check the server process has permission to read it ({e}).",
        ) from e

    return sorted(videos, key=lambda p: str(p).lower())


@router.get(
    "/browse",
    response_model=IngestBrowseResponse,
    summary="Browse server-side folders available for ingest",
    description="""
Lists folders on the server that jobs can be created from, so a client can
offer a folder picker instead of asking a user to type an absolute path (a
browser cannot discover one: neither the file input nor the File System Access
API exposes a real path).

Called with no `path`, returns the allowed roots. Called with a `path` inside
one of them, returns that folder's immediate subfolders (each with a count of
the videos directly inside it) and its own videos.

Admin-only, and only for callers on the server's own machine.
""",
)
async def browse(
    request: Request,
    path: str | None = Query(None, description="Folder to list; omit to list roots"),
    _user: dict[str, Any] = Depends(require_admin),
) -> IngestBrowseResponse:
    """List server-side folders a job batch could be created from."""
    require_local_caller(request)
    roots = allowed_roots()

    if path is None:
        # Present each root as a choosable directory rather than requiring the
        # client to know what a root is.
        return IngestBrowseResponse(
            path=None,
            parent=None,
            roots=[str(r) for r in roots],
            directories=[
                IngestDirectory(
                    name=str(root), path=str(root), video_count=_count_videos(root)
                )
                for root in roots
                if root.is_dir()
            ],
            videos=[],
            video_count=0,
        )

    resolved = resolve_within_roots(path)
    if not resolved.is_dir():
        raise APIError(
            status_code=422,
            code="INGEST_PATH_NOT_A_FOLDER",
            message=f"Not a folder: {resolved}",
            hint="Browse to the folder that contains the videos.",
        )

    try:
        entries = sorted(resolved.iterdir(), key=lambda p: str(p).lower())
    except OSError as e:
        raise APIError(
            status_code=403,
            code="INGEST_PATH_UNREADABLE",
            message=f"Could not read folder: {resolved}",
            hint=f"Check the server process has permission to read it ({e}).",
        ) from e

    directories: list[IngestDirectory] = []
    videos: list[IngestVideo] = []
    truncated = False

    for entry in entries:
        if len(directories) + len(videos) >= MAX_LISTING_ENTRIES:
            truncated = True
            break
        try:
            if entry.is_dir():
                directories.append(
                    IngestDirectory(
                        name=entry.name,
                        path=str(entry),
                        video_count=_count_videos(entry),
                    )
                )
            elif entry.is_file() and entry.suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(
                    IngestVideo(
                        name=entry.name,
                        path=str(entry),
                        size_bytes=entry.stat().st_size,
                    )
                )
        except OSError:
            # One unreadable entry shouldn't break browsing the folder.
            continue

    # Only offer a parent that is itself browsable.
    parent: str | None = None
    if resolved not in roots:
        candidate = resolved.parent
        if any(candidate == root or root in candidate.parents for root in roots):
            parent = str(candidate)

    return IngestBrowseResponse(
        path=str(resolved),
        parent=parent,
        roots=[str(r) for r in roots],
        directories=directories,
        videos=videos,
        video_count=len(videos),
        truncated=truncated,
    )


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    path: str = Field(description="Folder on the server containing the videos")
    recursive: bool = Field(
        default=False, description="Also include videos in subfolders"
    )
    selected_pipelines: list[str] | None = None
    config: dict[str, Any] | None = None
    batch_id: str | None = Field(
        default=None,
        description="Batch identifier to tag these jobs with. Generated if omitted.",
    )
    batch_name: str | None = Field(
        default=None,
        description="Human label for the batch. Defaults to the folder's name.",
    )
    dataset_id: str | None = None


class IngestSkipped(BaseModel):
    filename: str
    reason: str


class IngestResponse(BaseModel):
    batch_id: str
    batch_name: str | None
    path: str
    total: int = Field(description="Jobs created")
    created: list[str]
    skipped: list[IngestSkipped]


@router.post(
    "",
    response_model=IngestResponse,
    status_code=201,
    summary="Create a batch of jobs from a folder on the server",
    description="""
Turns every video in a server-side folder into a job, as one batch, in one
request -- no upload. Jobs reference the videos where they already are, so
nothing is copied and a 40-video corpus starts processing immediately instead
of after 40 multipart uploads.

The created jobs are ordinary jobs in every other respect: they queue, report
progress, cancel, and retry exactly like uploaded ones, and share a `batch_id`
so they can be tracked as a single run (`GET /api/v1/batches/{batch_id}`).
Deleting one removes its results, never the original video.

Files that cannot be used (unreadable, or empty) are reported in `skipped`
rather than failing the request, so one bad file in a corpus doesn't block the
other thirty-nine.

Admin-only, and only for callers on the server's own machine.
""",
)
async def ingest_folder(
    request: Request,
    body: IngestRequest,
    storage: StorageBackend = Depends(get_storage),
    db: Session = Depends(get_db),
    _user: dict[str, Any] = Depends(require_admin),
) -> IngestResponse:
    """Create one job per video in a server-side folder."""
    require_local_caller(request)
    resolved = resolve_within_roots(body.path)

    if not resolved.is_dir():
        raise APIError(
            status_code=422,
            code="INGEST_PATH_NOT_A_FOLDER",
            message=f"Not a folder: {resolved}",
            hint="Point this at the folder containing the videos.",
        )

    # Same validation an upload gets, before creating anything: an unavailable
    # pipeline or a bad config should fail the whole request, not leave a
    # half-created batch behind.
    validate_pipeline_selection(body.selected_pipelines, body.config)

    videos = find_videos(resolved, recursive=body.recursive)
    if not videos:
        raise APIError(
            status_code=422,
            code="INGEST_NO_VIDEOS",
            message=f"No videos found in {resolved}",
            hint=(
                "Supported extensions: "
                + ", ".join(VIDEO_EXTENSIONS)
                + ("." if body.recursive else ". Set recursive to search subfolders.")
            ),
        )

    batch_id = body.batch_id or str(uuid.uuid4())
    batch_name = body.batch_name or resolved.name
    provider = get_storage_provider()

    created: list[str] = []
    skipped: list[IngestSkipped] = []

    for video_path in videos:
        try:
            if video_path.stat().st_size == 0:
                skipped.append(
                    IngestSkipped(filename=video_path.name, reason="file is empty")
                )
                continue
        except OSError as e:
            skipped.append(
                IngestSkipped(filename=video_path.name, reason=f"unreadable ({e})")
            )
            continue

        try:
            job = BatchJob(
                # The original location, deliberately: ingest never copies.
                video_path=video_path,
                output_dir=None,
                config=body.config or {},
                status=JobStatus.PENDING,
                selected_pipelines=body.selected_pipelines,
                batch_id=batch_id,
                batch_name=batch_name,
                dataset_id=body.dataset_id,
            )
            # Results still live in this job's own storage directory, which is
            # what job deletion removes -- so deleting an ingested job never
            # touches the researcher's video.
            provider.create_job_dir(job.job_id)
            job.storage_path = provider.get_absolute_path(job.job_id, "")

            storage.save_job_metadata(job)
            created.append(job.job_id)
        except Exception as e:  # one bad file shouldn't sink the batch
            logger.warning(f"[INGEST] Could not create job for {video_path}: {e}")
            skipped.append(IngestSkipped(filename=video_path.name, reason=str(e)))

    if body.dataset_id:
        try:
            SavedDatasetCRUD.touch_last_used(db, body.dataset_id)
        except Exception as e:
            logger.debug(
                f"Could not touch last_used_at for dataset {body.dataset_id}: {e}"
            )

    logger.info(
        f"[INGEST] Created {len(created)} job(s) from {resolved} as batch {batch_id}"
    )

    return IngestResponse(
        batch_id=batch_id,
        batch_name=batch_name,
        path=str(resolved),
        total=len(created),
        created=created,
        skipped=skipped,
    )


__all__ = ["router", "extract_video_metadata", "find_videos", "VIDEO_EXTENSIONS"]
