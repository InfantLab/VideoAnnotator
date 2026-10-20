"""The videos the server knows, and runs on any of them (spec 029).

`GET /videos` lists videos for one scope (all, not in a project, a project, a
folder, a run) with the results each has and the counts for the Videos page's
left side (R2). `POST /videos/run` starts one run on any videos the server can
read (R8), `GET /videos/run-settings` says what New run starts from (R6), and
`GET /videos/activity` lists the runs going on (R17).

A video is a group of paths (R1, `video_index`); a project is a saved dataset
(R4, `projects`). The API keeps its words: jobs, batches, datasets.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ... import run_settings, video_index
from ...batch.types import BatchJob, JobStatus
from ...database.crud import SavedDatasetCRUD
from ...database.database import get_db
from ...projects import Catalog, catalog_for, folder_videos, with_videos
from ...results_folder import RunFolder, display_path, folder_ref, run_name
from ...storage.config import get_storage_root
from ...storage.manager import get_storage_provider
from ...video_index import Run, Video, VideoIndex
from ..database import get_storage_backend
from ..errors import APIError
from ..middleware.auth import validate_api_key
from .datasets import _owner_name
from .ingest import allowed_roots, places_in
from .videos import _in_readable_folder, _not_allowed

logger = logging.getLogger("videoannotator.api")

router = APIRouter()

UPLOADS = video_index.UPLOADS
MAX_LIMIT = 2_000


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ProjectRef(BaseModel):
    id: str
    name: str


class VideoResult(BaseModel):
    state: str = Field(description="done, running, waiting or failed")
    run_id: str = Field(description="The run it is from: a batch id or a job id")
    finished_at: datetime | None = None
    error: str | None = Field(default=None, description="Why it failed, if it did")


class VideoItem(BaseModel):
    path: str = Field(description="Its newest existing path: the one to open it by")
    paths: list[str] = Field(description="Every path of this video, newest first")
    name: str
    folder: str = Field(description="Its folder, or 'uploads' for an uploaded video")
    available: bool
    size_bytes: int | None = None
    projects: list[ProjectRef]
    results: dict[str, VideoResult] = Field(
        description="Per pipeline that ever ran on it, its newest finished "
        "result, else its newest state"
    )
    last_run_at: datetime | None = None


class ProjectCount(BaseModel):
    id: str
    name: str
    owner_name: str | None = None
    count: int
    follows_folder: bool = False
    server_folder: str | None = None


class FolderCount(BaseModel):
    path: str = Field(description="The folder, or 'uploads'")
    label: str
    count: int
    uploads: bool = False
    recursive: bool = Field(
        default=False, description="A place to look in, listed with its subfolders"
    )


class VideoCounts(BaseModel):
    all: int
    not_in_project: int
    projects: list[ProjectCount]
    folders: list[FolderCount]


class NewVideo(BaseModel):
    path: str
    name: str


class RunSummary(BaseModel):
    id: str
    name: str
    total: int
    done: int
    failed: int
    running: int
    waiting: int
    cancelled: int
    started_at: datetime | None = None
    finished_at: datetime | None = None
    pipelines: list[str]
    project: ProjectRef | None = None
    active: bool
    job_ids: list[str] = Field(default_factory=list)


class VideoScope(BaseModel):
    kind: str = Field(description="all, not_in_project, project, folder or run")
    id: str | None = None
    name: str | None = None
    path: str | None = None
    recursive: bool = False


class VideosResponse(BaseModel):
    scope: VideoScope
    videos: list[VideoItem]
    total: int
    counts: VideoCounts
    new_in_folder: list[NewVideo] = Field(default_factory=list)
    runs: list[RunSummary] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Building the list
# ---------------------------------------------------------------------------


def _item(video: Video, catalog: Catalog) -> VideoItem:
    return VideoItem(
        path=video.path,
        paths=video.paths,
        name=video.name,
        folder=video.folder,
        available=video.available,
        size_bytes=video.size_bytes,
        projects=[ProjectRef(**p) for p in catalog.projects_of(video.key)],
        results={k: VideoResult(**v.to_dict()) for k, v in video.results.items()},
        last_run_at=video.last_run_at,
    )


def run_summary(run: Run, catalog: Catalog) -> RunSummary:
    project = catalog.dataset(run.project_id) if run.project_id else None
    return RunSummary(
        **run.to_dict(),
        project=ProjectRef(id=str(project.id), name=project.name) if project else None,
        active=run.active,
        job_ids=[j.job_id for j in run.jobs],
    )


def _in_folder(video: Video, folder: str, recursive: bool) -> bool:
    for path in video.paths:
        parent = str(Path(path).parent)
        if parent == folder or (recursive and parent.startswith(folder + "/")):
            return True
    return False


def _readable(folder: str) -> bool:
    try:
        return _in_readable_folder(Path(folder))
    except (OSError, ValueError):
        return False


_places_cache: tuple[float, list[Any]] | None = None


def _places() -> list[Any]:
    """Spec 022's places with videos below, looked at once a minute."""
    import time

    global _places_cache
    now = time.monotonic()
    if _places_cache is None or now - _places_cache[0] > 60:
        try:
            places = [p for p in places_in(allowed_roots()) if p.has_videos]
        except OSError:
            places = []
        _places_cache = (now, places)
    return _places_cache[1]


def _counts(catalog: Catalog) -> VideoCounts:
    videos = catalog.index.videos()
    held = catalog.in_any_project()
    folders: dict[str, int] = {}
    uploads = 0
    for video in videos:
        if video.uploaded:
            uploads += 1
        elif _readable(video.folder):
            folders[video.folder] = folders.get(video.folder, 0) + 1
    for dataset in catalog.datasets:
        if dataset.server_folder and _readable(dataset.server_folder):
            folders.setdefault(str(Path(dataset.server_folder).expanduser()), 0)
    listed = [
        FolderCount(path=f, label=display_path(f), count=n)
        for f, n in sorted(folders.items())
    ]
    for place in _places():
        if place.path in folders:
            continue
        inside = sum(1 for v in videos if _in_folder(v, place.path, True))
        listed.append(
            FolderCount(
                path=place.path, label=place.display_path, count=inside, recursive=True
            )
        )
    if uploads:
        listed.append(
            FolderCount(
                path=UPLOADS, label="Uploaded videos", count=uploads, uploads=True
            )
        )
    return VideoCounts(
        all=len(videos),
        not_in_project=sum(1 for v in videos if v.key not in held),
        projects=sorted(
            (
                ProjectCount(
                    id=str(d.id),
                    name=d.name,
                    owner_name=_owner_name(d),
                    count=len(catalog.members.get(str(d.id), ())),
                    follows_folder=bool(d.server_folder) and not d.server_selection,
                    server_folder=d.server_folder,
                )
                for d in catalog.datasets
            ),
            key=lambda p: (p.name.lower(), p.owner_name or ""),
        ),
        folders=listed,
    )


def _resolved_folder(raw: str) -> str:
    try:
        return str(Path(raw.strip()).expanduser().resolve())
    except (OSError, RuntimeError) as e:
        raise _not_allowed(Path(raw)) from e


def _not_found(code: str, message: str) -> APIError:
    return APIError(status_code=404, code=code, message=message)


def scoped_videos(
    catalog: Catalog,
    *,
    project: str | None = None,
    folder: str | None = None,
    recursive: bool = False,
    not_in_project: bool = False,
    run: str | None = None,
) -> tuple[VideoScope, list[Video], VideoIndex, list[Run], list[Video]]:
    """The scope's videos, the index they come from (a folder adds its
    files), its runs, and a selection project's new folder videos."""
    index = catalog.index
    if project:
        dataset = catalog.dataset(project)
        if dataset is None:
            raise _not_found("DATASET_NOT_FOUND", f"Dataset '{project}' not found")
        held = catalog.members.get(project, set())
        runs = [r for r in video_index.runs(index) if r.project_id == project]
        return (
            VideoScope(kind="project", id=project, name=dataset.name),
            [v for v in index.videos() if v.key in held],
            index,
            runs,
            catalog.new.get(project, []),
        )
    if run:
        found = video_index.run(index, run)
        if found is None:
            raise _not_found("RUN_NOT_FOUND", f"No run '{run}' on this server.")
        return (
            VideoScope(kind="run", id=run, name=found.name),
            video_index.run_videos(index, run),
            index,
            [found],
            [],
        )
    if folder:
        if folder == UPLOADS:
            return (
                VideoScope(kind="folder", path=UPLOADS, name="Uploaded videos"),
                [v for v in index.videos() if v.uploaded],
                index,
                [],
                [],
            )
        resolved = _resolved_folder(folder)
        if not _readable(resolved):
            raise _not_allowed(Path(resolved))
        files = folder_videos(resolved, recursive)
        index = index.extended(str(f) for f in files)
        keys = {index.key_of(f) for f in files}
        videos = [
            v
            for v in index.videos()
            if v.key in keys or (not v.uploaded and _in_folder(v, resolved, recursive))
        ]
        return (
            VideoScope(
                kind="folder",
                path=resolved,
                name=display_path(resolved),
                recursive=recursive,
            ),
            videos,
            index,
            [],
            [],
        )
    if not_in_project:
        held = catalog.in_any_project()
        return (
            VideoScope(kind="not_in_project"),
            [v for v in index.videos() if v.key not in held],
            index,
            [],
            [],
        )
    return VideoScope(kind="all"), index.videos(), index, [], []


@router.get(
    "",
    response_model=VideosResponse,
    summary="The videos the server knows, for one scope (spec 029)",
    description=(
        "One of `project`, `folder` (+`recursive`; `uploads` for uploaded "
        "videos), `not_in_project`, `run`; none means every video. A video is "
        "every path the server can tell holds one file. `counts` are for the "
        "Videos page's left side. 403 `PATH_NOT_ALLOWED` for a folder outside "
        "the shared folders; 404 `DATASET_NOT_FOUND` / `RUN_NOT_FOUND`."
    ),
)
async def list_videos(
    project: str | None = Query(None, description="A saved dataset's id"),
    folder: str | None = Query(None, description="A folder, or 'uploads'"),
    recursive: bool = Query(False),
    not_in_project: bool = Query(False),
    run: str | None = Query(None, description="A batch id, or a job id"),
    q: str | None = Query(None, description="Names containing this, any case"),
    offset: int = Query(0, ge=0),
    limit: int = Query(500, ge=1, le=MAX_LIMIT),
    db: Session = Depends(get_db),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> VideosResponse:
    catalog = catalog_for(get_storage_backend(), db)
    scope, videos, _, runs, new = scoped_videos(
        catalog,
        project=project,
        folder=folder,
        recursive=recursive,
        not_in_project=not_in_project,
        run=run,
    )
    if q and q.strip():
        needle = q.strip().lower()
        videos = [v for v in videos if needle in v.name.lower()]
    return VideosResponse(
        scope=scope,
        videos=[_item(v, catalog) for v in videos[offset : offset + limit]],
        total=len(videos),
        counts=_counts(catalog),
        new_in_folder=[NewVideo(path=v.path, name=v.name) for v in new],
        runs=[run_summary(r, catalog) for r in runs],
    )


# ---------------------------------------------------------------------------
# What New run starts from
# ---------------------------------------------------------------------------


class RunSettingsSource(BaseModel):
    kind: str = Field(description="workflow, project, last_used or defaults")
    id: str | None = None
    name: str | None = None
    changed_at: str | None = None
    from_: dict[str, Any] | None = Field(
        default=None,
        alias="from",
        description="For a project's settings: the run, edit or workflow they came from",
    )

    model_config = {"populate_by_name": True}


class RunSettingsResponse(BaseModel):
    selected_pipelines: list[str] = Field(
        description="Empty with `defaults`: the viewer selects its defaults"
    )
    config: dict[str, Any]
    source: RunSettingsSource


@router.get(
    "/run-settings",
    response_model=RunSettingsResponse,
    response_model_by_alias=True,
    summary="The settings a new run starts from (spec 029 FR-012)",
    description=(
        "The chosen workflow's, else the project's current settings, else the "
        "settings of its newest run, else the caller's last run's, else the "
        "defaults; `source` says which."
    ),
)
async def get_run_settings(
    project: str | None = Query(None, description="A saved dataset's id"),
    workflow: str | None = Query(None, description="A saved preset's id"),
    db: Session = Depends(get_db),
    user: dict[str, Any] | None = Depends(validate_api_key),
) -> RunSettingsResponse:
    found = run_settings.prefill(
        db,
        user=user,
        project_id=project,
        workflow_id=workflow,
        storage=get_storage_backend(),
    )
    return RunSettingsResponse.model_validate(found)


# ---------------------------------------------------------------------------
# One run on any videos (R8)
# ---------------------------------------------------------------------------


class VideosRunRequest(BaseModel):
    paths: list[str]
    selected_pipelines: list[str] | None = None
    config: dict[str, Any] | None = None
    name: str | None = Field(default=None, description="The run's name")
    project_id: str | None = Field(
        default=None, description="The project (saved dataset) it runs for"
    )


class RunSkipped(BaseModel):
    path: str
    reason: str


class VideosRunResponse(BaseModel):
    run_id: str = Field(description="The batch id of the new jobs")
    name: str
    created: list[str]
    skipped: list[RunSkipped]
    results_folder: dict[str, Any] | None = None


def _stored_upload(index: VideoIndex, resolved: Path) -> Any | None:
    """The job whose own folder holds `resolved` (an upload), if any."""
    video = index.video_for(resolved)
    for job in video.jobs if video else []:
        folder = job.storage_path
        if folder and resolved.is_relative_to(Path(folder).expanduser().resolve()):
            return job
    return None


def _unusable(raw: str, storage: Any) -> tuple[Path | None, str | None]:
    try:
        resolved = Path(raw.strip()).expanduser().resolve()
    except (OSError, RuntimeError):
        return None, f"No video at '{raw}': it may have been moved or deleted."
    allowed = (
        _in_readable_folder(resolved)
        or storage.has_video_path(raw.strip())
        or storage.has_video_path(str(resolved))
    )
    if not allowed:
        return (
            None,
            f"{display_path(resolved)} isn't in a folder shared with VideoAnnotator",
        )
    if not resolved.is_file():
        return None, f"No video at '{resolved}': it may have been moved or deleted."
    return resolved, None


def _default_name(paths: list[Path], dataset: Any | None, index: VideoIndex) -> str:
    if dataset is not None:
        return str(dataset.name)
    folders = {p.parent for p in paths if not _uploaded(index, p)}
    if len(folders) == 1 and len(paths) > 1:
        return next(iter(folders)).name
    if len(paths) == 1:
        return paths[0].stem
    return run_name()


def _uploaded(index: VideoIndex, path: Path) -> bool:
    video = index.video_for(path)
    return video is not None and video.uploaded


def _in_place_job(
    video: Path,
    storage: Any,
    pipelines: list[str] | None,
    config: dict[str, Any] | None,
    *,
    batch_id: str,
    batch_name: str,
    dataset_id: str | None,
) -> BatchJob:
    """A job reading `video` where it is, in the run's folder (spec 022)."""
    job = BatchJob(
        video_path=video,
        config=config or {},
        status=JobStatus.PENDING,
        selected_pipelines=pipelines,
        batch_id=batch_id,
        batch_name=batch_name,
        dataset_id=dataset_id,
    )
    run = RunFolder.for_batch(
        batch_id,
        lambda: RunFolder.create(
            batch_name, batch_id=batch_id, pipelines=pipelines, config=config
        ),
    )
    run.add_video(job, video, size_bytes=video.stat().st_size)
    provider = get_storage_provider()
    provider.create_job_dir(job.job_id)
    job.storage_path = provider.get_absolute_path(job.job_id, "")
    storage.save_job_metadata(job)
    return job


def start_run(
    db: Session,
    *,
    paths: list[str],
    selected_pipelines: list[str] | None,
    config: dict[str, Any] | None,
    name: str | None,
    project_id: str | None,
    user: dict[str, Any] | None,
) -> VideosRunResponse:
    from .jobs import job_from_stored_video, validate_pipeline_selection

    storage = get_storage_backend()
    dataset = None
    if project_id:
        dataset = SavedDatasetCRUD.get_by_id(db, project_id)
        if dataset is None:
            raise _not_found("DATASET_NOT_FOUND", f"Dataset '{project_id}' not found")
    validate_pipeline_selection(selected_pipelines, config)

    catalog = catalog_for(storage, db)
    usable: list[Path] = []
    skipped: list[RunSkipped] = []
    seen: set[str] = set()
    for raw in paths:
        resolved, reason = _unusable(raw, storage)
        if resolved is None:
            skipped.append(RunSkipped(path=raw, reason=reason or ""))
        elif str(resolved) not in seen:
            seen.add(str(resolved))
            usable.append(resolved)
    if not usable:
        raise APIError(
            status_code=422,
            code="NOTHING_TO_RUN",
            message="None of these videos can be run.",
            hint="; ".join(s.reason for s in skipped) or "Choose some videos.",
        )

    index = catalog.index.extended(str(p) for p in usable)
    batch_id = str(uuid.uuid4())
    batch_name = (name or "").strip() or _default_name(usable, dataset, index)
    created: list[BatchJob] = []
    for video in usable:
        try:
            upload = _stored_upload(index, video)
            original = storage.load_job_metadata(upload.job_id) if upload else None
            if original is not None:
                job = job_from_stored_video(
                    original,
                    storage,
                    selected_pipelines,
                    config,
                    batch_id=batch_id,
                    batch_name=batch_name,
                    dataset_id=project_id,
                    video=video,
                )
            else:
                job = _in_place_job(
                    video,
                    storage,
                    selected_pipelines,
                    config,
                    batch_id=batch_id,
                    batch_name=batch_name,
                    dataset_id=project_id,
                )
            created.append(job)
        except Exception as e:  # one bad file shouldn't sink the run
            logger.warning(f"[RUN] Could not start {video}: {e}")
            skipped.append(RunSkipped(path=str(video), reason=str(e)))
    if not created:
        raise APIError(
            status_code=422,
            code="NOTHING_TO_RUN",
            message="None of these videos could be started.",
            hint="; ".join(s.reason for s in skipped),
        )

    if dataset is not None:
        _hold(db, dataset, [str(p) for p in usable], catalog)
    run_settings.record_run(
        db,
        user=user,
        dataset_id=project_id,
        selected_pipelines=selected_pipelines,
        config=config,
        run_id=batch_id,
        run_name=batch_name,
    )
    first = created[0]
    folder = Path(first.output_dir).parent if first.output_dir else None
    return VideosRunResponse(
        run_id=batch_id,
        name=batch_name,
        created=[j.job_id for j in created],
        skipped=skipped,
        results_folder=folder_ref(folder) if folder else None,
    )


def _hold(db: Session, dataset: Any, paths: list[str], catalog: Catalog) -> None:
    """A run for a project is on its videos: add any it doesn't hold yet."""
    try:
        change = with_videos(dataset, paths, catalog)
        if change is not None:
            entries, selection = change
            SavedDatasetCRUD.update(
                db, str(dataset.id), video_manifest=entries, server_selection=selection
            )
        SavedDatasetCRUD.touch_last_used(db, str(dataset.id))
    except Exception as e:
        logger.warning(f"Could not add the run's videos to {dataset.name}: {e}")


@router.post(
    "/run",
    response_model=VideosRunResponse,
    status_code=201,
    summary="One run on any videos the server can read (spec 029 R8)",
    description=(
        "Creates one job per usable path, sharing one batch id. A video in a "
        "shared folder is read in place; an upload is hard-linked (copied if "
        "linking fails) from its stored copy. Paths it can't use are listed in "
        "`skipped`; 422 `NOTHING_TO_RUN` when none are usable. With "
        "`project_id`, the jobs carry it as `dataset_id`, the videos join the "
        "project, and its current settings become these. Any signed-in user."
    ),
)
async def run_videos(
    body: VideosRunRequest,
    db: Session = Depends(get_db),
    user: dict[str, Any] | None = Depends(validate_api_key),
) -> VideosRunResponse:
    return start_run(
        db,
        paths=body.paths,
        selected_pipelines=body.selected_pipelines,
        config=body.config,
        name=body.name,
        project_id=body.project_id,
        user=user,
    )


# ---------------------------------------------------------------------------
# What is going on (R17)
# ---------------------------------------------------------------------------


class ActivityRun(BaseModel):
    id: str
    name: str
    total: int
    done: int
    failed: int
    running: int
    waiting: int
    cancelled: int
    project: ProjectRef | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    stopped: bool = False


class ActivityResponse(BaseModel):
    active: list[ActivityRun]
    recent: list[ActivityRun] = Field(
        description="Up to 5 runs finished in the last 24 hours, newest first"
    )


RECENT = 5


def _activity(run: Run, catalog: Catalog) -> ActivityRun:
    summary = run_summary(run, catalog)
    return ActivityRun(
        **summary.model_dump(
            include={
                "id",
                "name",
                "total",
                "done",
                "failed",
                "running",
                "waiting",
                "cancelled",
                "project",
                "started_at",
                "finished_at",
            }
        ),
        stopped=not run.active and run.cancelled > 0,
    )


def _naive_local(value: datetime) -> datetime:
    """Jobs record naive local times; a timezone-aware one is made the same."""
    return value.astimezone().replace(tzinfo=None) if value.tzinfo else value


@router.get(
    "/activity",
    response_model=ActivityResponse,
    summary="Runs with videos waiting or running, and those just finished",
)
async def activity(
    db: Session = Depends(get_db),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> ActivityResponse:
    storage = get_storage_backend()
    index = video_index.cached_index(storage, uploads_root=get_storage_root())
    all_runs = video_index.runs(index)
    active = [r for r in all_runs if r.active]
    since = datetime.now() - timedelta(hours=24)
    finished = sorted(
        (
            r
            for r in all_runs
            if not r.active and r.finished_at and _naive_local(r.finished_at) >= since
        ),
        key=lambda r: _naive_local(r.finished_at or since),
        reverse=True,
    )[:RECENT]
    if not active and not finished:
        return ActivityResponse(active=[], recent=[])
    catalog = catalog_for(storage, db)
    return ActivityResponse(
        active=[_activity(r, catalog) for r in active],
        recent=[_activity(r, catalog) for r in finished],
    )
