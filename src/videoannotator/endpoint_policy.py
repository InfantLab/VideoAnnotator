"""Whether frames may go to a model server's address (spec 033 R5).

Remote model servers are switched off on a fresh install, by one server-wide
setting an admin can switch on, and that the environment variable
`VIDEOANNOTATOR_REMOTE_MODEL_SERVERS` (`on`/`off`) fixes when set, so a lab
can lock it off for an ethics board. With the switch on, spec 027's
confirmation of each remote address still applies. Addresses confirmed
before the switch existed keep working without it (Principle V).

Every route that sends frames checks here: Ask AI's tries, runs and drafts,
the preview, and the pipeline itself for runs submitted any other way.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from .database.models import EndpointConfirmation, ServerSetting
from .endpoints import classify_endpoint, normalise_base_url
from .model_servers import ModelServerError

REMOTE_SETTING = "remote_model_servers"
REMOTE_ENV = "VIDEOANNOTATOR_REMOTE_MODEL_SERVERS"

REMOTE_WARNING = (
    "With remote model servers switched on, frames from the videos people ask "
    "about can be sent to servers outside this computer and your local network, "
    "run by someone else. For video of research participants this is usually a "
    "question for your ethics approval and consent forms. Each remote server "
    "still has to be confirmed before its first frames are sent."
)

REMOTE_NOT_ALLOWED_MESSAGE = "Remote model servers are switched off on this server."
REMOTE_NOT_ALLOWED_HINT = (
    "An administrator can switch them on in Settings → Model servers, unless the "
    "server's configuration fixes it."
)


def _env_value() -> bool | None:
    value = os.environ.get(REMOTE_ENV, "").strip().lower()
    if value in ("on", "true", "1", "yes"):
        return True
    if value in ("off", "false", "0", "no"):
        return False
    return None


def remote_allowed(db: Session) -> tuple[bool, bool]:
    """(allowed, fixed by the environment)."""
    fixed = _env_value()
    if fixed is not None:
        return fixed, True
    row = db.get(ServerSetting, REMOTE_SETTING)
    return (bool(row.value) if row is not None else False), False


def set_remote_allowed(db: Session, value: bool, user_id: str | None) -> None:
    if _env_value() is not None:
        raise ModelServerError(
            409,
            "FIXED_BY_ENVIRONMENT",
            f"This is set by this server's configuration ({REMOTE_ENV}).",
        )
    row = db.get(ServerSetting, REMOTE_SETTING)
    now = datetime.now(UTC)
    if row is None:
        db.add(
            ServerSetting(
                name=REMOTE_SETTING, value=value, updated_by=user_id, updated_at=now
            )
        )
    else:
        row.value = value
        row.updated_by = user_id
        row.updated_at = now
    db.commit()


def endpoint_state(db: Session, base_url: str) -> dict[str, Any]:
    info = classify_endpoint(base_url)
    confirmation = (
        None
        if info.cls == "local"
        else db.get(EndpointConfirmation, normalise_base_url(base_url))
    )
    before_switch = bool(confirmation is not None and confirmation.before_switch)
    if info.cls == "local":
        sendable = True
    else:
        allowed, _ = remote_allowed(db)
        sendable = confirmation is not None and (allowed or before_switch)
    return {
        "base_url": base_url,
        "host": info.host,
        "class": info.cls,
        "confirmed": info.cls == "local" or confirmation is not None,
        "before_switch": before_switch,
        "sendable": sendable,
    }


def _not_allowed(host: str, base_url: str) -> ModelServerError:
    error = ModelServerError(
        403, "REMOTE_NOT_ALLOWED", REMOTE_NOT_ALLOWED_MESSAGE, REMOTE_NOT_ALLOWED_HINT
    )
    error.details = {"host": host, "base_url": base_url}
    return error


def require_addable(db: Session, base_url: str) -> None:
    """A remote address can be listed only while remote servers are on,
    or if it was confirmed before the switch existed."""
    state = endpoint_state(db, base_url)
    if state["class"] == "local" or state["before_switch"]:
        return
    if not remote_allowed(db)[0]:
        raise _not_allowed(state["host"], base_url)


def require_sendable(db: Session, base_url: str) -> dict[str, Any]:
    """The address's state, if frames may go there; else why not: 403
    REMOTE_NOT_ALLOWED before 409 REMOTE_NOT_CONFIRMED."""
    state = endpoint_state(db, base_url)
    if state["sendable"]:
        return state
    if not state["before_switch"] and not remote_allowed(db)[0]:
        raise _not_allowed(state["host"], base_url)
    error = ModelServerError(
        409,
        "REMOTE_NOT_CONFIRMED",
        f"Frames from your video would be sent to {state['host']}, a remote "
        "server. Confirm that first.",
        "POST /api/v1/ask/endpoints/confirm with this base_url.",
    )
    error.details = {"host": state["host"], "base_url": base_url}
    raise error


def require_sendable_for_url(base_url: str) -> dict[str, Any]:
    """`require_sendable` through a session of its own, for a run."""
    from .ask_history import own_session

    with own_session() as db:
        return require_sendable(db, base_url)
