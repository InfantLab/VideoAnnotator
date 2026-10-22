"""The model servers this server knows about (spec 033 R4).

The default Ollama server (`OLLAMA_BASE_URL`) is always listed first and is
not a row; others are added in Settings → Model servers or with
`videoannotator model-servers add`. A run's config carries `backend` and
`base_url`, never a server id, so a config copied elsewhere still says where
it went; a server row supplies only its name and its key.

Keys are resolved per call, for the normalised address: the environment
variable the row names, when set and non-empty; else the key stored in
`model_server_keys.json`; else none.
"""

from __future__ import annotations

import os
import re
import uuid
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from . import secrets_store
from .config_env import default_ollama_base_url
from .database.models import ModelServer
from .endpoints import normalise_base_url

DEFAULT_ID = "default"
DEFAULT_NAME = "This computer (Ollama)"
KINDS = ("ollama", "openai_compatible")
_ENV_NAME = re.compile(r"[A-Z_][A-Z0-9_]*")


class ModelServerError(ValueError):
    """Why a server can't be added or changed, as the API reports it."""

    def __init__(
        self, status: int, code: str, message: str, hint: str | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.hint = hint
        self.details: dict[str, Any] | None = None


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return (value if value.tzinfo else value.replace(tzinfo=UTC)).isoformat()


def _default() -> dict[str, Any]:
    return {
        "id": DEFAULT_ID,
        "name": DEFAULT_NAME,
        "kind": "ollama",
        "base_url": normalise_base_url(default_ollama_base_url()),
        "api_key_env": None,
        "structured_output": None,
        "video_models": {},
        "removable": False,
        "created_at": None,
    }


def _view(row: ModelServer) -> dict[str, Any]:
    return {
        "id": str(row.id),
        "name": row.name,
        "kind": row.kind,
        "base_url": row.base_url,
        "api_key_env": row.api_key_env,
        "structured_output": row.structured_output,
        "video_models": dict(row.video_models or {}),
        "removable": True,
        "created_at": _iso(row.created_at),
    }


def _refresh_env_names(db: Session) -> None:
    secrets_store.KEY_ENV_NAMES.clear()
    secrets_store.KEY_ENV_NAMES.update(
        name
        for (name,) in db.query(ModelServer.api_key_env).filter(
            ModelServer.api_key_env.isnot(None)
        )
    )


def key_source(server: dict[str, Any]) -> str | None:
    """Where the key comes from: "env" (set), "stored", or None."""
    env = server.get("api_key_env")
    if env and os.environ.get(env):
        return "env"
    if secrets_store.read_model_server_key(server["base_url"]):
        return "stored"
    return None


def list_servers(db: Session) -> list[dict[str, Any]]:
    rows = db.query(ModelServer).order_by(ModelServer.name).all()
    return [_default(), *(_view(r) for r in rows)]


def _row(db: Session, server_id: str) -> ModelServer | None:
    try:
        return db.get(ModelServer, uuid.UUID(server_id))
    except ValueError:
        return None


def get(db: Session, server_id: str) -> dict[str, Any] | None:
    if server_id == DEFAULT_ID:
        return _default()
    row = _row(db, server_id)
    return _view(row) if row is not None else None


def server_for_url(db: Session, base_url: str) -> dict[str, Any] | None:
    key = normalise_base_url(base_url)
    if key == normalise_base_url(default_ollama_base_url()):
        return _default()
    row = db.query(ModelServer).filter(ModelServer.base_url == key).one_or_none()
    return _view(row) if row is not None else None


def check_base_url(base_url: str) -> str:
    parts = urlsplit(base_url.strip())
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ModelServerError(
            422,
            "INVALID_BASE_URL",
            "The address must start with http:// or https:// and name a host.",
            hint="For example http://127.0.0.1:8080/v1",
        )
    return normalise_base_url(base_url)


def _check_env_name(api_key_env: str | None) -> str | None:
    if not api_key_env:
        return None
    name = api_key_env.strip().lstrip("$")
    if not _ENV_NAME.fullmatch(name) or len(name) > 100:
        raise ModelServerError(
            422,
            "INVALID_ENV_NAME",
            f"'{api_key_env}' isn't an environment variable name.",
            hint="Capital letters, digits and _, like LAB_GPU_KEY.",
        )
    return name


def _check_name(name: str) -> str:
    name = (name or "").strip()
    if not name or len(name) > 100:
        raise ModelServerError(
            422, "INVALID_NAME", "Give the server a name (up to 100 characters)."
        )
    return name


def add(
    db: Session,
    name: str,
    kind: str,
    base_url: str,
    api_key: str | None = None,
    api_key_env: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    from . import endpoint_policy

    name = _check_name(name)
    if kind not in KINDS:
        raise ModelServerError(
            422, "INVALID_KIND", "The kind must be ollama or openai_compatible."
        )
    key = check_base_url(base_url)
    env = _check_env_name(api_key_env)
    if server_for_url(db, key) is not None:
        raise ModelServerError(
            409, "SERVER_EXISTS", f"A model server at {key} is already listed."
        )
    endpoint_policy.require_addable(db, key)
    row = ModelServer(
        id=uuid.uuid4(),
        name=name,
        kind=kind,
        base_url=key,
        api_key_env=env,
        structured_output=None,
        video_models={},
        created_by=user_id,
        created_at=datetime.now(UTC),
    )
    db.add(row)
    db.commit()
    if api_key:
        secrets_store.save_model_server_key(key, api_key)
    _refresh_env_names(db)
    return _view(row)


def _require_row(db: Session, server_id: str) -> ModelServer:
    if server_id == DEFAULT_ID:
        raise ModelServerError(
            422,
            "DEFAULT_SERVER",
            "The default server comes from this server's configuration "
            "(OLLAMA_BASE_URL) and can't be changed here.",
        )
    row = _row(db, server_id)
    if row is None:
        raise ModelServerError(404, "SERVER_NOT_FOUND", "No such model server.")
    return row


_UNCHANGED: Any = object()


def update(
    db: Session,
    server_id: str,
    name: str | None = None,
    api_key: str | None = None,
    api_key_env: Any = _UNCHANGED,
    clear_key: bool = False,
) -> dict[str, Any]:
    row = _require_row(db, server_id)
    if name is not None:
        row.name = _check_name(name)
    if api_key_env is not _UNCHANGED:
        row.api_key_env = _check_env_name(api_key_env)
    db.commit()
    if clear_key:
        secrets_store.remove_model_server_key(row.base_url)
    if api_key:
        secrets_store.save_model_server_key(row.base_url, api_key)
    _refresh_env_names(db)
    return _view(row)


def remove(db: Session, server_id: str) -> None:
    row = _require_row(db, server_id)
    base_url = row.base_url
    db.delete(row)
    db.commit()
    secrets_store.remove_model_server_key(base_url)
    _refresh_env_names(db)


def record_capabilities(
    db: Session,
    base_url: str,
    structured_output: bool | None = None,
    video: tuple[str, bool] | None = None,
) -> None:
    """What calls and checks learnt about a listed server (R8, R12)."""
    row = (
        db.query(ModelServer)
        .filter(ModelServer.base_url == normalise_base_url(base_url))
        .one_or_none()
    )
    if row is None:
        return
    if structured_output is not None:
        row.structured_output = structured_output
    if video is not None:
        row.video_models = {**(row.video_models or {}), video[0]: video[1]}
    db.commit()


def key_for(db: Session, base_url: str) -> str | None:
    server = server_for_url(db, base_url)
    env = server.get("api_key_env") if server else None
    if env:
        value = os.environ.get(env)
        if value:
            return value
    return secrets_store.read_model_server_key(base_url)


def _own_session() -> Session:
    from .ask_history import own_session

    return own_session()


def key_for_url(base_url: str) -> str | None:
    """`key_for` through a session of its own, for a run outside a request."""
    try:
        with _own_session() as db:
            return key_for(db, base_url)
    except SQLAlchemyError:
        # A database from before spec 033 that the server hasn't upgraded
        # (`videoannotator process` without it): a stored key can still apply.
        return secrets_store.read_model_server_key(base_url)


def capabilities_for_url(base_url: str) -> dict[str, Any] | None:
    with _own_session() as db:
        return server_for_url(db, base_url)


def record_capabilities_for_url(
    base_url: str,
    structured_output: bool | None = None,
    video: tuple[str, bool] | None = None,
) -> None:
    with _own_session() as db:
        record_capabilities(db, base_url, structured_output, video)
