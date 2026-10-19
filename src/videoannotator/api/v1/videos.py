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
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ...database_location import database_path
from ...results_folder import display_path
from ...storage.config import get_storage_root
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


def readable_video(raw_path: str) -> Path:
    """`raw_path` resolved, if Ask AI may read it; else the API error saying why.

    Resolved first (symlinks included), so `..` can't escape a folder.
    """
    if not raw_path or not raw_path.strip():
        raise APIError(
            status_code=422,
            code="VIDEO_PATH_REQUIRED",
            message="Say which video to use.",
        )
    try:
        resolved = Path(raw_path.strip()).expanduser().resolve()
    except (OSError, RuntimeError) as e:
        raise APIError(
            status_code=404,
            code="VIDEO_NOT_FOUND",
            message=f"Could not find the video at '{raw_path}'",
        ) from e

    allowed = (
        any(_within(resolved, root) for root in [*allowed_roots(), videos_root()])
        or get_storage_backend().has_video_path(raw_path.strip())
        or get_storage_backend().has_video_path(str(resolved))
    )
    if not allowed:
        raise APIError(
            status_code=403,
            code="PATH_NOT_ALLOWED",
            message=f"{display_path(resolved)} isn't in a folder shared with "
            "VideoAnnotator",
            hint="Upload the video, or choose one in a folder shared with the server.",
        )
    if not resolved.is_file():
        raise APIError(
            status_code=404,
            code="VIDEO_NOT_FOUND",
            message=f"No video at '{resolved}': it may have been moved or deleted.",
        )
    return resolved


def video_duration(path: Path) -> float | None:
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        finally:
            cap.release()
        return round(frames / fps, 3) if fps > 0 else None
    except Exception as e:
        logger.debug(f"Could not read the duration of {path}: {e}")
        return None


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
    return StreamTokenResponse(
        token=token,
        expires_at=datetime.fromtimestamp(expires, UTC),
        path=str(path),
        name=path.name,
        # From the server's decoder: the window can draw the timeline even
        # when this browser can't play the video's codec.
        duration_sec=video_duration(path),
    )


@router.get(
    "/stream",
    summary="Play a video (range requests); the token is the credential",
    response_class=FileResponse,
)
async def stream_video(
    path: str = Query(...),
    token: str = Query(...),
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
    return FileResponse(resolved)
