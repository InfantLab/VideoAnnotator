"""Who owns a saved project or workflow (spec 029).

With sign-in, the caller. Without it (`AUTH_REQUIRED=false`, a single-user
install), a local "anonymous" user, so projects and workflows still work: they
are the heart of the Videos page. That user never counts as a real one.
"""

from __future__ import annotations

from typing import Any

from ..database.crud import UserCRUD
from .errors import APIError
from .middleware.auth import is_auth_required

ANONYMOUS_EMAIL = "anonymous@videoannotator.local"
ANONYMOUS_USERNAME = "anonymous"


def anonymous_user_id(db: Any) -> str:
    user = UserCRUD.get_by_email(db, ANONYMOUS_EMAIL)
    if user is None:
        user = UserCRUD.create(
            db,
            email=ANONYMOUS_EMAIL,
            username=ANONYMOUS_USERNAME,
            full_name="Without sign-in",
        )
    return str(user.id)


def owner_id(db: Any, user: dict[str, Any] | None, what: str) -> str:
    """The id to save `what` (a dataset, a preset) under."""
    if user is not None and user.get("id"):
        return str(user["id"])
    if not is_auth_required():
        return anonymous_user_id(db)
    raise APIError(
        status_code=400,
        code="OWNER_IDENTITY_REQUIRED",
        message=f"Saving or modifying a {what} requires an authenticated identity.",
        hint="Provide a valid API key (Authorization: Bearer ...).",
    )
