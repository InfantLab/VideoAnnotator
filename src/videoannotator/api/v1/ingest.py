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
   the server user's home directory (none in a container). Paths are fully resolved (symlinks
   included) before the check, so `..` and symlink escapes cannot get out.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...batch.types import BatchJob, JobStatus
from ...config_env import (
    INGEST_ROOTS,
    LAUNCHER_REQUESTS_DIR,
    managed_by_launcher,
    missing_shares,
    published_locally,
)
from ...database.crud import SavedDatasetCRUD
from ...database.database import get_db
from ...results_folder import (
    RunFolder,
    display_path,
    folder_opener,
    folder_ref,
    give_to_owner,
    in_container,
    not_shared_message,
    results_root,
    run_name,
)
from ...storage.base import StorageBackend
from ...storage.manager import get_storage_provider
from ..database import get_storage_backend
from ..errors import APIError
from ..middleware.auth import require_admin, validate_api_key
from .jobs import FolderRef, extract_video_metadata, validate_pipeline_selection

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
    without configuration while still being a real boundary. Not in a
    container, whose home is the inside of the box (`/root`), never the
    researcher's: there, only folders shared with it count (spec 024, FR-016).
    """
    configured = [entry for entry in INGEST_ROOTS.split(os.pathsep) if entry.strip()]
    if not configured:
        return [] if in_container() else [Path.home().resolve()]

    roots = []
    for entry in configured:
        try:
            roots.append(Path(entry.strip()).expanduser().resolve())
        except OSError as e:  # pragma: no cover - unresolvable configured path
            logger.warning(f"[INGEST] Ignoring unusable configured root {entry!r}: {e}")
    return roots


def is_same_machine(request: Request) -> bool:
    """Whether the caller is on the server's own machine.

    Loopback callers are; so is everyone when the server is published on the
    host's loopback only (Docker, `VIDEOANNOTATOR_PUBLISHED_LOCALLY`), because
    then nothing but this machine can reach it. Decided here, from how the
    request arrived -- never from anything the client says about itself.
    """
    if published_locally():
        return True
    client = request.client
    host = client.host if client else None
    if not host:
        return False
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        # A non-address host (e.g. a unix socket's "testclient") is treated
        # as local: it cannot have come over the network.
        return host in ("testclient", "localhost")


def require_local_caller(request: Request) -> None:
    """Reject callers that aren't on this machine.

    Naming a server-side path only makes sense when the caller *is* the server's
    user. A remote client still has `POST /api/v1/jobs` and its upload.
    """
    if not is_same_machine(request):
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
            message=not_shared_message(resolved),
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
# Access
# ---------------------------------------------------------------------------


class Place(BaseModel):
    label: str = Field(description="What to call it: 'Home', 'Videos', 'Desktop', ...")
    path: str
    display_path: str
    has_videos: bool


# Where a researcher's videos usually are, under their home folder, in the
# order My folders offers them. Only those that exist are listed.
HOME_PLACES = (
    "Videos",
    "Movies",
    "Desktop",
    "Documents",
    "Downloads",
    "Pictures",
    "OneDrive",
)


def places_in(folders: list[Path]) -> list[Place]:
    """Starting points for My folders: each allowed folder, and the usual
    places under home, so a researcher starts somewhere they recognise
    rather than in a listing of their home folder (spec 022)."""
    home = Path.home().resolve()
    places: list[Place] = []
    for root in folders:
        candidates = [("Home" if root == home else root.name or str(root), root)]
        if root == home:
            candidates += [(name, root / name) for name in HOME_PLACES]
        for label, folder in candidates:
            if folder.is_dir():
                places.append(
                    Place(
                        label=label,
                        path=str(folder),
                        display_path=display_path(folder),
                        has_videos=has_videos_below(folder),
                    )
                )
    return places


class Share(BaseModel):
    """A folder shared with VideoAnnotator when it started (spec 024)."""

    path: str = Field(
        description="Where the server sees it; the host path when not present"
    )
    display_path: str = Field(description="As the researcher's computer shows it")
    present: bool = Field(description="Found when VideoAnnotator last started")
    stop_requested: bool = Field(
        default=False,
        description="Stop sharing was asked for: it stops at the next start",
    )


# One host path per line; the launcher applies and deletes it at the next start.
STOP_SHARING_FILE = "stop-sharing.txt"


def _stop_requests() -> set[str]:
    try:
        text = (LAUNCHER_REQUESTS_DIR / STOP_SHARING_FILE).read_text(encoding="utf-8")
    except OSError:
        return set()
    return {line.strip() for line in text.splitlines() if line.strip()}


def current_shares() -> list[Share]:
    """Every shared folder: those configured, then those missing at this start.

    Only folders shared on purpose; the home folder a plain install reads by
    default isn't a share.
    """
    requested = _stop_requests()
    shares = []
    for entry in INGEST_ROOTS.split(os.pathsep):
        if not entry.strip():
            continue
        folder = Path(entry.strip()).expanduser().resolve()
        shown = display_path(folder)
        shares.append(
            Share(
                path=str(folder),
                display_path=shown,
                present=folder.is_dir(),
                stop_requested=shown in requested,
            )
        )
    for host in missing_shares():
        shares.append(
            Share(
                path=host,
                display_path=host,
                present=False,
                stop_requested=host in requested,
            )
        )
    return shares


class IngestAccessResponse(BaseModel):
    same_machine: bool = Field(
        description="The caller is on the server's own machine (loopback, or "
        "VIDEOANNOTATOR_PUBLISHED_LOCALLY under Docker)"
    )
    can_read_in_place: bool = Field(
        description="The caller may choose videos on this machine (My folders)"
    )
    reason: str | None = Field(
        default=None,
        description="Why `can_read_in_place` is false, for researchers",
    )
    allowed_folders: list[FolderRef]
    results_root: FolderRef
    can_open_folders: bool = Field(
        description="`POST /api/v1/results/open` can show a folder on this machine"
    )
    places: list[Place] = Field(
        default_factory=list,
        description="Where My folders starts: allowed folders and the usual video "
        "places under home that exist (empty unless `can_read_in_place`)",
    )
    in_container: bool = Field(
        default=False, description="The server runs in a container (spec 024)"
    )
    managed_by_launcher: bool = Field(
        default=False,
        description="Started by videoannotator-start, which can stop sharing a "
        "folder (spec 024)",
    )
    shares: list[Share] = Field(
        default_factory=list,
        description="Folders shared with VideoAnnotator, present or not (spec 024)",
    )


NO_SHARE_LAUNCHER = (
    "VideoAnnotator can only see folders you share with it. To share one, run: "
    "videoannotator-start share"
)
NO_SHARE_COMPOSE = (
    "VideoAnnotator can only see folders you share with it. Set VIDEOS_DIR when "
    "starting it (see the installation guide)."
)


def _ref(folder: Path) -> FolderRef:
    return FolderRef(path=str(folder), display_path=display_path(folder))


def _usable(folder: Path) -> bool:
    try:
        return folder.is_dir() and any(folder.iterdir())
    except OSError:
        return False


@router.get(
    "/access",
    response_model=IngestAccessResponse,
    summary="What this caller may do with videos on the server's machine",
    description="""
Tells a client, for the caller making the request, whether it counts as being
on the server's own machine and so may choose videos where they are (spec
022): the viewer offers "My folders" only when `can_read_in_place` is true,
and otherwise explains `reason` and offers upload. Decided by the server from
how the request arrived; the client can't claim it.

Any authenticated caller may ask.
""",
)
async def access(
    request: Request,
    user: dict[str, Any] | None = Depends(validate_api_key),
) -> IngestAccessResponse:
    """Report this caller's access to in-place reading."""
    same_machine = is_same_machine(request)
    folders = [root for root in allowed_roots() if _usable(root)]
    reason = None
    if not same_machine:
        reason = (
            "This server is on another computer, so its folders aren't yours. "
            "Upload your videos instead."
        )
    elif not (user or {}).get("is_admin", False):
        reason = (
            "Only an administrator can choose videos on this computer. Use an "
            "administrator key (see Settings), or upload the videos."
        )
    elif not folders:
        if not in_container():
            reason = (
                "None of the allowed video folders exist. Set "
                "VIDEOANNOTATOR_INGEST_ROOTS to the folders your videos are in."
            )
        elif managed_by_launcher():
            reason = NO_SHARE_LAUNCHER
        else:
            reason = NO_SHARE_COMPOSE
    return IngestAccessResponse(
        same_machine=same_machine,
        can_read_in_place=reason is None,
        reason=reason,
        allowed_folders=[_ref(f) for f in folders],
        results_root=_ref(results_root()),
        can_open_folders=same_machine and folder_opener() is not None,
        places=places_in(folders) if reason is None else [],
        in_container=in_container(),
        managed_by_launcher=managed_by_launcher(),
        shares=current_shares(),
    )


class StopSharingRequest(BaseModel):
    path: str = Field(
        description="The shared folder, as shown or as the server sees it"
    )


@router.post(
    "/shares/stop",
    response_model=Share,
    summary="Stop sharing a folder, from the next start",
    description="""
Asks `videoannotator-start` to stop sharing a folder (spec 024): the request is
left in its requests folder and applied the next time VideoAnnotator starts.
The server can't change what is shared itself, and a request can only ever
make sharing narrower.

Administrator only, on this computer only. 404 `SHARE_NOT_FOUND` for a folder
that isn't shared; 409 `NOT_MANAGED_BY_LAUNCHER` when VideoAnnotator wasn't
started by `videoannotator-start` (change the compose settings instead).
""",
)
async def stop_sharing(
    request: Request,
    body: StopSharingRequest,
    _user: dict[str, Any] = Depends(require_admin),
) -> Share:
    """Record a request to stop sharing a folder."""
    require_local_caller(request)
    if not managed_by_launcher():
        raise APIError(
            status_code=409,
            code="NOT_MANAGED_BY_LAUNCHER",
            message="Shared folders are set when VideoAnnotator starts.",
            hint="Change VIDEOS_DIR in the compose settings (see the installation "
            "guide), then restart.",
        )
    wanted = body.path.strip()
    share = next(
        (s for s in current_shares() if wanted in (s.display_path, s.path)), None
    )
    if share is None:
        raise APIError(
            status_code=404,
            code="SHARE_NOT_FOUND",
            message=f"{wanted} isn't a shared folder.",
            hint="GET /api/v1/ingest/access lists the shared folders.",
        )
    if not share.stop_requested:
        requests = LAUNCHER_REQUESTS_DIR / STOP_SHARING_FILE
        try:
            with requests.open("a", encoding="utf-8") as f:
                f.write(share.display_path + "\n")
        except OSError as e:
            raise APIError(
                status_code=500,
                code="STOP_SHARING_FAILED",
                message="Couldn't record the request to stop sharing.",
                hint=f"Run: videoannotator-start unshare ({e}).",
            ) from e
        # Under rootful Docker the server is root; the launcher deletes it.
        give_to_owner(requests)
    return share.model_copy(update={"stop_requested": True})


# ---------------------------------------------------------------------------
# Browsing
# ---------------------------------------------------------------------------


class IngestDirectory(BaseModel):
    name: str
    path: str
    video_count: int
    has_videos: bool = Field(
        default=False,
        description="Videos are in this folder or a few levels below it (spec 022), "
        "so a researcher can tell which folders lead somewhere",
    )


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


# Folders a researcher never keeps videos in, hidden from My folders (spec 022):
# a home folder is otherwise mostly tool and system clutter.
NOISE_FOLDERS = {
    "node_modules",
    "__pycache__",
    "site-packages",
    "venv",
    "AppData",
    "Application Data",
    "Library",
    "Applications",
    "snap",
    "go",
    "miniconda3",
    "anaconda3",
    "System Volume Information",
}
SEARCH_DEPTH = 3
SEARCH_BUDGET = 400


def is_noise(entry: Path, results: Path) -> bool:
    """Hidden, system or tool folders, and the results folder (results, not videos)."""
    name = entry.name
    return name.startswith((".", "$", "~")) or name in NOISE_FOLDERS or entry == results


def has_videos_below(directory: Path) -> bool:
    """Whether a video is in `directory` or a few levels below.

    Bounded in depth and in entries looked at, so a listing stays quick on a
    big home folder; past the budget a folder reads as having none.
    """
    results = results_root()
    stack = [(directory, 0)]
    seen = 0
    while stack:
        folder, depth = stack.pop()
        try:
            entries = list(os.scandir(folder))
        except OSError:
            continue
        for entry in entries:
            seen += 1
            if seen > SEARCH_BUDGET:
                return False
            path = Path(entry.path)
            if is_noise(path, results):
                continue
            try:
                if entry.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                    return True
                if entry.is_dir(follow_symlinks=False) and depth + 1 < SEARCH_DEPTH:
                    stack.append((path, depth + 1))
            except OSError:
                continue
    return False


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
    results = results_root()
    try:
        entries = directory.rglob("*") if recursive else directory.iterdir()
        videos = [
            entry
            for entry in entries
            if entry.is_file()
            and entry.suffix.lower() in VIDEO_EXTENSIONS
            and not entry.is_relative_to(results)
        ]
    except OSError as e:
        raise APIError(
            status_code=403,
            code="INGEST_PATH_UNREADABLE",
            message=f"Could not read folder: {directory}",
            hint=f"Check the server process has permission to read it ({e}).",
        ) from e

    return sorted(videos, key=lambda p: str(p).lower())


class ScannedVideo(BaseModel):
    relative_path: str
    name: str
    size_bytes: int


class IngestScanResponse(BaseModel):
    path: str
    recursive: bool
    videos: list[ScannedVideo]


@router.get(
    "/scan",
    response_model=IngestScanResponse,
    summary="List the videos a folder ingest would use",
    description="""
Every video `POST /api/v1/ingest` would create a job for, with its path within
the folder and its size, without creating anything. Used to save a server
folder as a dataset and to show what changed in it since (spec 018).

Admin-only, and only for callers on the server's own machine, as ingest is.
""",
)
async def scan(
    request: Request,
    path: str = Query(..., description="Folder on the server"),
    recursive: bool = Query(False, description="Include subfolders"),
    _user: dict[str, Any] = Depends(require_admin),
) -> IngestScanResponse:
    """List a server folder's videos with sizes."""
    require_local_caller(request)
    resolved = resolve_within_roots(path)
    if not resolved.is_dir():
        raise APIError(
            status_code=422,
            code="INGEST_PATH_NOT_A_FOLDER",
            message=f"Not a folder: {resolved}",
            hint="Choose the folder that contains the videos.",
        )
    videos = [
        ScannedVideo(
            relative_path=video.relative_to(resolved).as_posix(),
            name=video.name,
            size_bytes=video.stat().st_size,
        )
        for video in find_videos(resolved, recursive=recursive)
    ]
    return IngestScanResponse(path=str(resolved), recursive=recursive, videos=videos)


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
    results = results_root()

    for entry in entries:
        if len(directories) + len(videos) >= MAX_LISTING_ENTRIES:
            truncated = True
            break
        if is_noise(entry, results):
            continue
        try:
            if entry.is_dir():
                count = _count_videos(entry)
                directories.append(
                    IngestDirectory(
                        name=entry.name,
                        path=str(entry),
                        video_count=count,
                        has_videos=count > 0 or has_videos_below(entry),
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

    # Folders that lead to videos first: that's what the researcher is after.
    directories.sort(key=lambda d: not d.has_videos)

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
    files: list[str] | None = Field(
        default=None,
        description="Only these videos, as paths relative to `path` (spec 022). "
        "Omit for every video in the folder; `recursive` is ignored when set.",
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
    results_folder: FolderRef | None = Field(
        default=None, description="The run's results folder (spec 022)"
    )


def _chosen_videos(
    folder: Path, files: list[str], skipped: list[IngestSkipped]
) -> list[Path]:
    """The chosen videos, each checked against the allowed folders on its own.

    Every file is resolved before the check, so `..` or a symlink can't reach
    outside them; a file chosen twice (or by two spellings) runs once.
    """
    results = results_root()
    seen: set[Path] = set()
    videos: list[Path] = []
    for relative in files:
        try:
            video = resolve_within_roots(str(folder / relative))
        except APIError as e:
            skipped.append(IngestSkipped(filename=relative, reason=e.message))
            continue
        if not video.is_file() or video.suffix.lower() not in VIDEO_EXTENSIONS:
            skipped.append(IngestSkipped(filename=relative, reason="not a video file"))
            continue
        if video.is_relative_to(results):
            skipped.append(
                IngestSkipped(filename=relative, reason="inside the results folder")
            )
            continue
        if video not in seen:
            seen.add(video)
            videos.append(video)
    return videos


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

    skipped: list[IngestSkipped] = []
    if body.files is not None:
        videos = _chosen_videos(resolved, body.files, skipped)
    else:
        videos = find_videos(resolved, recursive=body.recursive)
    if not videos:
        raise APIError(
            status_code=422,
            code="INGEST_NO_VIDEOS",
            message=f"No videos found in {resolved}",
            hint=(
                "; ".join(f"{s.filename}: {s.reason}" for s in skipped)
                if skipped
                else "Supported extensions: "
                + ", ".join(VIDEO_EXTENSIONS)
                + ("." if body.recursive else ". Set recursive to search subfolders.")
            ),
        )

    batch_id = body.batch_id or str(uuid.uuid4())
    batch_name = body.batch_name or resolved.name
    provider = get_storage_provider()
    # Created before any job, so an unwritable results folder refuses the
    # whole run (FR-026) instead of failing it one job at a time.
    run = RunFolder.create(
        run_name(batch_name=batch_name, source_folder=resolved),
        batch_id=batch_id,
        pipelines=body.selected_pipelines,
        config=body.config,
        source_root=resolved,
    )

    created: list[str] = []

    for video_path in videos:
        try:
            size_bytes = video_path.stat().st_size
            if size_bytes == 0:
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
                config=body.config or {},
                status=JobStatus.PENDING,
                selected_pipelines=body.selected_pipelines,
                batch_id=batch_id,
                batch_name=batch_name,
                dataset_id=body.dataset_id,
            )
            # Results go to the run's folder, never beside the video; deleting
            # the job removes them and never touches the researcher's video.
            run.add_video(job, video_path, size_bytes=size_bytes)
            provider.create_job_dir(job.job_id)
            job.storage_path = provider.get_absolute_path(job.job_id, "")

            storage.save_job_metadata(job)
            created.append(job.job_id)
        except Exception as e:  # one bad file shouldn't sink the batch
            logger.warning(f"[INGEST] Could not create job for {video_path}: {e}")
            skipped.append(IngestSkipped(filename=video_path.name, reason=str(e)))

    if not created:
        shutil.rmtree(run.path, ignore_errors=True)

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
        results_folder=folder_ref(run.path) if created else None,
    )


__all__ = ["router", "extract_video_metadata", "find_videos", "VIDEO_EXTENSIONS"]
