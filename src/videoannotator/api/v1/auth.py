"""Current-identity endpoint for VideoAnnotator API.

Lets an authenticated client discover who it's authenticated as -- most
importantly, whether it has administrator privileges. Added alongside
specs/005-pipeline-extras-install's admin-only install endpoints: without
this, a client (e.g. the viewer) that hits a 403 from an admin-only action
has no way to self-diagnose why, since nothing else in the API surfaces
`is_admin` back to the caller.
"""

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..middleware.auth import validate_required_api_key

router = APIRouter()


class CurrentUserResponse(BaseModel):
    """Response for `GET /auth/me`."""

    id: str | None = None
    username: str | None = None
    email: str | None = None
    is_admin: bool = False


@router.get(
    "/me",
    response_model=CurrentUserResponse,
    summary="Get the currently authenticated user's identity",
    description="""
Returns identity information for whoever the caller's API key/token belongs to --
in particular `is_admin`, which gates actions like the pipeline extras-install
endpoints (`POST /api/v1/pipelines/extras/{extra}/install`). Always requires
authentication, regardless of the server's `AUTH_REQUIRED` setting.

A `false` `is_admin` here is the direct explanation for a `403 Administrator
privileges required` response from an admin-only endpoint -- surface this to
users rather than leaving that error unexplained.
""",
)
async def get_current_user_identity(
    user: dict[str, Any] = Depends(validate_required_api_key),
) -> CurrentUserResponse:
    """Return the current caller's identity, including admin status."""
    return CurrentUserResponse(
        id=user.get("id"),
        username=user.get("username"),
        email=user.get("email"),
        is_admin=bool(user.get("is_admin", False)),
    )
