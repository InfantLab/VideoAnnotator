"""Results folders on this machine (spec 022).

A browser page can't open a folder on the computer it runs on, so the server,
on that same computer, does it for the viewer's "Open folder" button.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, Field

from ...results_folder import display_path, open_folder, results_root
from ..errors import APIError
from ..middleware.auth import validate_api_key
from .ingest import is_same_machine

router = APIRouter()


class OpenFolderRequest(BaseModel):
    path: str = Field(description="A folder inside the results folder")


@router.post(
    "/open",
    status_code=204,
    summary="Open a results folder in this computer's file manager",
    description="""
Only for callers on the server's own machine (403 `NOT_SAME_MACHINE`), and only
for folders inside the results folder (422 `PATH_OUTSIDE_RESULTS`), so it can't
be used to look around the filesystem. 409 `OPEN_FOLDER_UNSUPPORTED` when there
is no desktop to open it on (a headless server, or Docker): show the location
for the researcher to copy instead.
""",
)
async def open_results_folder(
    body: OpenFolderRequest,
    request: Request,
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> Response:
    """Open a results folder on the server's own desktop."""
    if not is_same_machine(request):
        raise APIError(
            status_code=403,
            code="NOT_SAME_MACHINE",
            message="Folders can only be opened from the computer the server runs on.",
            hint="Copy the location instead, or download the results.",
        )
    root = results_root()
    try:
        target = Path(body.path).expanduser().resolve()
    except (OSError, RuntimeError):
        target = None
    if target is None or not (target == root or target.is_relative_to(root)):
        raise APIError(
            status_code=422,
            code="PATH_OUTSIDE_RESULTS",
            message=f"Only folders inside {display_path(root)} can be opened.",
        )
    if not target.is_dir():
        raise APIError(
            status_code=404,
            code="RESULTS_FOLDER_NOT_FOUND",
            message=f"Results not found at {display_path(target)}",
            hint="It may have been moved, renamed or deleted outside VideoAnnotator.",
        )
    try:
        open_folder(target)
    except OSError as e:
        raise APIError(
            status_code=409,
            code="OPEN_FOLDER_UNSUPPORTED",
            message=f"Can't open folders on this server ({e}).",
            hint=f"Copy the location instead: {display_path(target)}",
        ) from e
    return Response(status_code=204)
