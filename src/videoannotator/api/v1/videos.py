"""Videos the browser can play and Ask AI can read (spec 027 R12; spec 028 too).

Ask AI works on a video the server can read: one under the folders shared with
it, one uploaded here, or one an earlier run read. Any signed-in user may use
any of them (no admin or same-machine check: everyone on a server sees
everything, as with the history).

An upload is stored once per content, under the storage root's `videos/`, so
the same file dropped twice is kept once and a later run reads it in place.

A `<video>` element can't send the API key, so playback uses a short-lived
token instead: an HMAC of the path and its expiry with a secret kept beside
the database.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import shutil
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ... import playable, video_notes
from ...database.database import get_db
from ...database_location import database_path
from ...projects import catalog_for
from ...results_folder import display_path, video_unavailable_reason
from ...storage.config import get_storage_root
from ...video_runs import runs_for, sha256_for, video_of
from ..database import get_storage_backend
from ..errors import APIError
from ..middleware.auth import validate_api_key
from .ingest import allowed_roots

logger = logging.getLogger("videoannotator.api")

router = APIRouter()

TOKEN_LIFETIME_SEC = 3600
_CHUNK = 1024 * 1024


def videos_root() -> Path:
    return (get_storage_root() / "videos").resolve()


def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _not_allowed(resolved: Path) -> APIError:
    return APIError(
        status_code=403,
        code="PATH_NOT_ALLOWED",
        message=f"{display_path(resolved)} isn't in a folder shared with "
        "VideoAnnotator",
        hint="Upload the video, or choose one in a folder shared with the server.",
    )


def _path_required() -> APIError:
    return APIError(
        status_code=422,
        code="VIDEO_PATH_REQUIRED",
        message="Say which video to use.",
    )


def _in_readable_folder(resolved: Path) -> bool:
    return any(_within(resolved, root) for root in [*allowed_roots(), videos_root()])


def readable_video(raw_path: str) -> Path:
    """`raw_path` resolved, if Ask AI may read it; else the API error saying why.

    Resolved first (symlinks included), so `..` can't escape a folder.
    """
    if not raw_path or not raw_path.strip():
        raise _path_required()
    try:
        resolved = Path(raw_path.strip()).expanduser().resolve()
    except (OSError, RuntimeError) as e:
        raise APIError(
            status_code=404,
            code="VIDEO_NOT_FOUND",
            message=f"Could not find the video at '{raw_path}'",
        ) from e

    allowed = (
        _in_readable_folder(resolved)
        or get_storage_backend().has_video_path(raw_path.strip())
        or get_storage_backend().has_video_path(str(resolved))
    )
    if not allowed:
        raise _not_allowed(resolved)
    if not resolved.is_file():
        raise APIError(
            status_code=404,
            code="VIDEO_NOT_FOUND",
            message=f"No video at '{resolved}': it may have been moved or deleted.",
        )
    return resolved


def video_timing(path: Path) -> tuple[float | None, float | None]:
    """(duration in seconds, frames per second), each None when unreadable."""
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        finally:
            cap.release()
        if fps > 0:
            return round(frames / fps, 3), round(fps, 3)
        return None, None
    except Exception as e:
        logger.debug(f"Could not read the duration of {path}: {e}")
        return None, None


def video_duration(path: Path) -> float | None:
    return video_timing(path)[0]


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


class VideoUploadResponse(BaseModel):
    path: str
    sha256: str
    name: str
    duration_sec: float | None = None


@router.post(
    "",
    response_model=VideoUploadResponse,
    status_code=201,
    summary="Upload a video once, to use in Ask AI and later runs",
)
async def upload_video(
    video: UploadFile = File(..., description="The video file"),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> VideoUploadResponse:
    root = videos_root()
    root.mkdir(parents=True, exist_ok=True)
    name = Path(video.filename or "video").name or "video"

    digest = hashlib.sha256()
    fd, temp_name = tempfile.mkstemp(dir=root, prefix=".incoming-")
    try:
        await video.seek(0)
        with os.fdopen(fd, "wb") as out:
            while chunk := await video.read(_CHUNK):
                digest.update(chunk)
                out.write(chunk)
        sha = digest.hexdigest()
        folder = root / sha[:16]
        existing = sorted(p for p in folder.glob("*") if p.is_file())
        if existing:
            target = existing[0]
        else:
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / name
            shutil.move(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)

    return VideoUploadResponse(
        path=str(target),
        sha256=sha,
        name=target.name,
        duration_sec=video_duration(target),
    )


# ---------------------------------------------------------------------------
# Runs on a video (spec 028)
# ---------------------------------------------------------------------------


class RunFile(BaseModel):
    name: str
    kind: str = Field(description="result, provenance or other")
    download_url: str


class PipelineRunInfo(BaseModel):
    name: str
    status: str
    error: str | None = None
    files: list[RunFile] = Field(default_factory=list)


class RunInfo(BaseModel):
    job_id: str
    status: str
    created_at: datetime | None = None
    completed_at: datetime | None = None
    batch_id: str | None = None
    dataset_id: str | None = None
    run_name: str | None = Field(
        default=None, description="The run folder's name (spec 022), when it has one"
    )
    pipelines: list[PipelineRunInfo] = Field(default_factory=list)
    batch_name: str | None = None
    rerun_of: str | None = None
    progress: float = Field(default=0.0, description="Percent of its pipelines done")
    queue_position: int | None = Field(
        default=None, description="While waiting: its place in the queue, from 1"
    )
    error: str | None = None
    selected_pipelines: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(
        default_factory=dict, description="The settings it ran with"
    )
    results_folder: dict[str, Any] | None = Field(
        default=None, description="Where its results are (spec 022), and `exists`"
    )


class ProjectRef(BaseModel):
    id: str
    name: str


class VideoRunsResponse(BaseModel):
    path: str
    paths: list[str] = Field(
        default_factory=list,
        description="Every path of this video, newest first (spec 029 R1): "
        "an upload and the copies its reruns made",
    )
    name: str
    available: bool
    unavailable_reason: str | None = None
    duration_sec: float | None = None
    sha256: str | None = Field(
        default=None,
        description="From a run's provenance; null when unknown (never computed here)",
    )
    runs: list[RunInfo]
    projects: list[ProjectRef]


def _video_for_page(raw_path: str) -> tuple[Path, list[Any]]:
    """The video a Video page shows, resolved, with its runs (spec 028).

    Any of its paths opens it (spec 029 R1); the path returned is the one to
    play: the asked one if it is there, else its newest existing one.
    Allowed when a job read it (even if it has moved since) or it is in a
    folder Ask AI may read; else 403. A missing file no job read is 404.
    """
    if not raw_path or not raw_path.strip():
        raise _path_required()
    try:
        resolved = Path(raw_path.strip()).expanduser().resolve()
    except (OSError, RuntimeError) as e:
        raise _not_allowed(Path(raw_path.strip())) from e
    runs = runs_for(get_storage_backend(), resolved)
    if not runs and not _in_readable_folder(resolved):
        raise _not_allowed(resolved)
    if not runs and not resolved.is_file():
        raise APIError(
            status_code=404,
            code="VIDEO_NOT_FOUND",
            message=f"No video at '{resolved}': it may have been moved or deleted.",
        )
    if not resolved.is_file():
        video = video_of(get_storage_backend(), resolved)
        if video is not None and video.available:
            resolved = Path(video.path)
    return resolved, runs


@router.get(
    "/runs",
    response_model=VideoRunsResponse,
    summary="Every run on one video, with its files, and the projects holding it",
)
async def video_runs(
    path: str = Query("", description="The video's path on the server"),
    db: Session = Depends(get_db),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> VideoRunsResponse:
    resolved, runs = _video_for_page(path)
    available = resolved.is_file()
    storage = get_storage_backend()
    video = video_of(storage, resolved)
    catalog = catalog_for(storage, db)
    key = catalog.index.key_of(resolved)
    return VideoRunsResponse(
        path=str(resolved),
        paths=video.paths if video is not None else [str(resolved)],
        name=resolved.name,
        available=available,
        unavailable_reason=None if available else video_unavailable_reason(resolved),
        duration_sec=video_duration(resolved) if available else None,
        sha256=sha256_for(runs),
        runs=[RunInfo.model_validate(r.to_dict()) for r in runs],
        projects=[ProjectRef(**p) for p in catalog.projects_of(key)] if key else [],
    )


# ---------------------------------------------------------------------------
# Notes on a video (spec 028 R9)
# ---------------------------------------------------------------------------


class Note(BaseModel):
    id: str
    video_path: str
    start_sec: float
    end_sec: float | None = None
    text: str
    author_id: str
    author_name: str
    created_at: datetime
    updated_at: datetime | None = None
    updated_by: str | None = None


class NotesResponse(BaseModel):
    notes: list[Note]


class NoteCreate(BaseModel):
    path: str
    start_sec: float
    end_sec: float | None = None
    text: str


class NoteUpdate(BaseModel):
    text: str | None = None
    start_sec: float | None = None
    end_sec: float | None = None


def _note_error(e: video_notes.NoteError) -> APIError:
    return APIError(status_code=422, code=e.code, message=e.message)


def _note_not_found(note_id: str) -> APIError:
    return APIError(
        status_code=404, code="NOTE_NOT_FOUND", message=f"No note {note_id}."
    )


def _duration_of(path: Path) -> float | None:
    return video_duration(path) if path.is_file() else None


@router.get("/notes", response_model=NotesResponse, summary="Notes on one video")
async def list_notes(
    path: str = Query("", description="The video's path on the server"),
    db: Session = Depends(get_db),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> NotesResponse:
    resolved, _ = _video_for_page(path)
    video = video_of(get_storage_backend(), resolved)
    notes = video_notes.list_for(db, resolved, video.paths if video else None)
    return NotesResponse(notes=[Note(**video_notes.to_dict(n)) for n in notes])


@router.post(
    "/notes", response_model=Note, status_code=201, summary="Add a note to a video"
)
async def create_note(
    body: NoteCreate,
    db: Session = Depends(get_db),
    user: dict[str, Any] | None = Depends(validate_api_key),
) -> Note:
    resolved, runs = _video_for_page(body.path)
    try:
        note = video_notes.create(
            db,
            path=resolved,
            start=body.start_sec,
            end=body.end_sec,
            text=body.text,
            user=user,
            duration=_duration_of(resolved),
            sha256=sha256_for(runs),
        )
    except video_notes.NoteError as e:
        raise _note_error(e) from e
    return Note(**video_notes.to_dict(note))


@router.patch("/notes/{note_id}", response_model=Note, summary="Change a note")
async def update_note(
    note_id: str,
    body: NoteUpdate,
    db: Session = Depends(get_db),
    user: dict[str, Any] | None = Depends(validate_api_key),
) -> Note:
    fields = body.model_dump(exclude_unset=True)
    changes: dict[str, Any] = {}
    if "text" in fields:
        changes["text"] = fields["text"] or ""
    if "start_sec" in fields and fields["start_sec"] is not None:
        changes["start"] = fields["start_sec"]
    if "end_sec" in fields:
        changes["end"] = fields["end_sec"]
    try:
        existing = video_notes.get(db, note_id)
        note = video_notes.update(
            db,
            note_id,
            user=user,
            duration=_duration_of(Path(existing.video_path)),
            **changes,
        )
    except video_notes.NoteNotFound as e:
        raise _note_not_found(note_id) from e
    except video_notes.NoteError as e:
        raise _note_error(e) from e
    return Note(**video_notes.to_dict(note))


@router.delete("/notes/{note_id}", status_code=204, summary="Delete a note")
async def delete_note(
    note_id: str,
    db: Session = Depends(get_db),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> Response:
    try:
        video_notes.delete(db, note_id)
    except video_notes.NoteNotFound as e:
        raise _note_not_found(note_id) from e
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def _secret() -> bytes:
    path = database_path().parent / "stream_secret"
    try:
        return path.read_bytes()
    except FileNotFoundError:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:  # another request created it first
            return path.read_bytes()
        with os.fdopen(fd, "wb") as f:
            f.write(secrets.token_bytes(32))
        return path.read_bytes()


def _signature(path: str, expires: int) -> str:
    return hmac.new(_secret(), f"{path}|{expires}".encode(), "sha256").hexdigest()


def make_stream_token(path: Path, now: float | None = None) -> tuple[str, int]:
    expires = int((now if now is not None else time.time()) + TOKEN_LIFETIME_SEC)
    return f"{expires}.{_signature(str(path), expires)}", expires


def token_valid(path: Path, token: str, now: float | None = None) -> bool:
    expires_text, _, signature = token.partition(".")
    try:
        expires = int(expires_text)
    except ValueError:
        return False
    if expires < (now if now is not None else time.time()):
        return False
    return hmac.compare_digest(signature, _signature(str(path), expires))


class StreamTokenRequest(BaseModel):
    path: str


class StreamTokenResponse(BaseModel):
    token: str
    expires_at: datetime
    path: str
    name: str
    duration_sec: float | None = None
    fps: float | None = Field(
        default=None, description="Frames per second, for frame numbers and stepping"
    )
    playable_copy: str = Field(
        default="none",
        description="An H.264 copy for browsers that can't play the original: "
        "none, making, ready, failed or unavailable (no ffmpeg)",
    )


@router.post(
    "/stream-token",
    response_model=StreamTokenResponse,
    summary="A short-lived token to play a video in the browser",
)
async def stream_token(
    body: StreamTokenRequest,
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> StreamTokenResponse:
    path = readable_video(body.path)
    token, expires = make_stream_token(path)
    duration, fps = video_timing(path)
    return StreamTokenResponse(
        token=token,
        expires_at=datetime.fromtimestamp(expires, UTC),
        path=str(path),
        name=path.name,
        # From the server's decoder: the window can draw the timeline even
        # when this browser can't play the video's codec.
        duration_sec=duration,
        fps=fps,
        playable_copy=playable.state(path).state,
    )


class PlayableCopyRequest(BaseModel):
    path: str


class PlayableCopyResponse(BaseModel):
    state: str = Field(description="none, making, ready, failed or unavailable")
    error: str | None = None


@router.get(
    "/playable-copy",
    response_model=PlayableCopyResponse,
    summary="Whether a video has a copy every browser can play (H.264)",
)
async def get_playable_copy(
    path: str = Query(...),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> PlayableCopyResponse:
    found = playable.state(readable_video(path))
    return PlayableCopyResponse(state=found.state, error=found.error)


@router.post(
    "/playable-copy",
    response_model=PlayableCopyResponse,
    summary="Make a copy of a video every browser can play (H.264), in the background",
)
async def make_playable_copy(
    body: PlayableCopyRequest,
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> PlayableCopyResponse:
    found = playable.start(readable_video(body.path))
    return PlayableCopyResponse(state=found.state, error=found.error)


@router.get(
    "/stream",
    summary="Play a video (range requests); the token is the credential",
    response_class=FileResponse,
)
async def stream_video(
    path: str = Query(...),
    token: str = Query(...),
    copy: bool = Query(False, description="Its playable copy, if one was made"),
) -> FileResponse:
    invalid = APIError(
        status_code=401,
        code="TOKEN_INVALID",
        message="This video link has expired or is for another video.",
        hint="Reload the page to get a new one.",
    )
    try:
        resolved = Path(path).expanduser().resolve()
    except (OSError, RuntimeError) as e:
        raise invalid from e
    if not token_valid(resolved, token):
        raise invalid
    if not resolved.is_file():
        raise APIError(
            status_code=404,
            code="VIDEO_NOT_FOUND",
            message=f"No video at '{resolved}': it may have been moved or deleted.",
        )
    if copy:
        made = playable.copy_path(resolved)
        if not made.is_file():
            raise APIError(
                status_code=404,
                code="NO_PLAYABLE_COPY",
                message="This video has no playable copy yet.",
            )
        return FileResponse(made, media_type="video/mp4")
    return FileResponse(resolved)
