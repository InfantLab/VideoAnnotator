"""Model servers: the list, adding and checking one, and the remote switch
(spec 033 R4, R5, R13).

No response carries a key: a server says only whether it has one and where
it comes from. Checks are explicit, so listing never waits on a server.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from ... import endpoint_policy, model_servers, secrets_store
from ...database.database import get_db
from ...model_servers import ModelServerError
from ...pipelines.vlm_annotation.clients import check_server
from ..errors import APIError
from ..middleware.auth import require_admin, validate_api_key

router = APIRouter()

Kind = Literal["ollama", "openai_compatible"]


def _api_error(e: ModelServerError) -> APIError:
    return APIError(
        status_code=e.status,
        code=e.code,
        message=str(e),
        hint=e.hint,
        details=e.details,
    )


class ModelServerEndpoint(BaseModel):
    host: str
    class_: Literal["local", "remote"] = Field(alias="class")
    confirmed: bool
    sendable: bool
    before_switch: bool = False

    model_config = {"populate_by_name": True}


class ModelServerOut(BaseModel):
    id: str
    name: str
    kind: Kind
    base_url: str
    has_key: bool
    key_source: Literal["env", "stored"] | None = Field(
        None, description="env: from the named environment variable; never the key"
    )
    api_key_env: str | None = None
    endpoint: ModelServerEndpoint
    structured_output: bool | None = None
    video_models: dict[str, bool] = {}
    removable: bool


class RemoteModelServers(BaseModel):
    allowed: bool
    fixed_by_environment: bool
    warning: str = endpoint_policy.REMOTE_WARNING


class ModelServersResponse(BaseModel):
    servers: list[ModelServerOut]
    remote: RemoteModelServers
    keys_file: str


def _server(db: Session, server: dict[str, Any]) -> ModelServerOut:
    source = model_servers.key_source(server)
    state = endpoint_policy.endpoint_state(db, server["base_url"])
    return ModelServerOut(
        id=server["id"],
        name=server["name"],
        kind=server["kind"],
        base_url=server["base_url"],
        has_key=source is not None,
        key_source=source,
        api_key_env=server["api_key_env"],
        endpoint=ModelServerEndpoint(**state),
        structured_output=server["structured_output"],
        video_models=server["video_models"],
        removable=server["removable"],
    )


def _remote(db: Session) -> RemoteModelServers:
    allowed, fixed = endpoint_policy.remote_allowed(db)
    return RemoteModelServers(allowed=allowed, fixed_by_environment=fixed)


@router.get(
    "",
    response_model=ModelServersResponse,
    response_model_by_alias=True,
    summary="The model servers this server knows, the default Ollama one first",
)
def list_servers(
    db: Session = Depends(get_db),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> ModelServersResponse:
    return ModelServersResponse(
        servers=[_server(db, s) for s in model_servers.list_servers(db)],
        remote=_remote(db),
        keys_file=str(secrets_store.model_server_keys_path()),
    )


class ModelServerAdd(BaseModel):
    name: str
    kind: Kind
    base_url: str
    api_key: str | None = Field(
        None, description="Stored in the keys file, never shown"
    )
    api_key_env: str | None = Field(
        None, description="The name of an environment variable holding the key"
    )


def _user_id(user: dict[str, Any] | None) -> str | None:
    return str(user["id"]) if user and user.get("id") else None


@router.post(
    "",
    response_model=ModelServerOut,
    response_model_by_alias=True,
    status_code=201,
    summary="Add a model server",
    description="A remote address is refused while remote model servers are off.",
)
def add_server(
    body: ModelServerAdd,
    db: Session = Depends(get_db),
    user: dict[str, Any] | None = Depends(validate_api_key),
) -> ModelServerOut:
    try:
        server = model_servers.add(
            db,
            body.name,
            body.kind,
            body.base_url,
            api_key=body.api_key,
            api_key_env=body.api_key_env,
            user_id=_user_id(user),
        )
    except ModelServerError as e:
        raise _api_error(e) from e
    return _server(db, server)


class ModelServerPatch(BaseModel):
    name: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    clear_key: bool = False


class ModelServerSettings(BaseModel):
    remote_model_servers: bool
    fixed_by_environment: bool = False
    warning: str = endpoint_policy.REMOTE_WARNING


@router.get(
    "/settings",
    response_model=ModelServerSettings,
    summary="Whether remote model servers are switched on",
)
def get_settings(
    db: Session = Depends(get_db),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> ModelServerSettings:
    allowed, fixed = endpoint_policy.remote_allowed(db)
    return ModelServerSettings(remote_model_servers=allowed, fixed_by_environment=fixed)


class ModelServerSettingsChange(BaseModel):
    remote_model_servers: bool


@router.put(
    "/settings",
    response_model=ModelServerSettings,
    summary="Switch remote model servers on or off (administrators)",
    description="409 FIXED_BY_ENVIRONMENT when VIDEOANNOTATOR_REMOTE_MODEL_SERVERS "
    "is set.",
)
def put_settings(
    body: ModelServerSettingsChange,
    db: Session = Depends(get_db),
    user: dict[str, Any] = Depends(require_admin),
) -> ModelServerSettings:
    try:
        endpoint_policy.set_remote_allowed(
            db, body.remote_model_servers, _user_id(user)
        )
    except ModelServerError as e:
        raise _api_error(e) from e
    return get_settings(db, user)


@router.patch(
    "/{server_id}",
    response_model=ModelServerOut,
    response_model_by_alias=True,
    summary="Rename a server, or change its key or key variable",
    description="The address and kind are fixed: remove the server and add it again.",
)
def update_server(
    server_id: str,
    body: ModelServerPatch,
    db: Session = Depends(get_db),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> ModelServerOut:
    changes: dict[str, Any] = {
        "name": body.name,
        "api_key": body.api_key,
        "clear_key": body.clear_key,
    }
    if "api_key_env" in body.model_fields_set:
        changes["api_key_env"] = body.api_key_env
    try:
        server = model_servers.update(db, server_id, **changes)
    except ModelServerError as e:
        raise _api_error(e) from e
    return _server(db, server)


@router.delete(
    "/{server_id}",
    status_code=204,
    summary="Remove a server and its stored key; results that used it stay",
)
def delete_server(
    server_id: str,
    db: Session = Depends(get_db),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> Response:
    try:
        model_servers.remove(db, server_id)
    except ModelServerError as e:
        raise _api_error(e) from e
    return Response(status_code=204)


class ModelServerCheckRequest(BaseModel):
    kind: Kind
    base_url: str
    api_key: str | None = None
    id: str | None = Field(None, description="Use this listed server's key")
    model: str | None = Field(
        None, description="Also ask whether this model takes video (a synthetic clip)"
    )


class ModelServerCheckEndpoint(BaseModel):
    host: str
    class_: Literal["local", "remote"] = Field(alias="class")
    confirmed: bool
    sendable: bool

    model_config = {"populate_by_name": True}


class ModelServerCheck(BaseModel):
    outcome: Literal["unreachable", "wrong_kind", "no_models", "models", "key_refused"]
    message: str
    hint: str | None = None
    kind_matches: bool | None = None
    suggest_kind: Kind | None = None
    suggest_base_url: str | None = None
    models: list[str] = []
    endpoint: ModelServerCheckEndpoint
    video: bool | None = None


@router.post(
    "/check",
    response_model=ModelServerCheck,
    response_model_by_alias=True,
    summary="Check a model server, listed or not yet",
    description="""
Reachable? The right kind? Any models? Does it take the key? With `model`, also
whether that model takes video, from one tiny synthetic clip (never a
participant's frames, and not to a remote address frames can't go to yet).
""",
)
async def check(
    body: ModelServerCheckRequest,
    db: Session = Depends(get_db),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> ModelServerCheck:
    try:
        base_url = model_servers.check_base_url(body.base_url)
    except ModelServerError as e:
        raise _api_error(e) from e
    api_key = body.api_key
    if not api_key and body.id:
        listed = model_servers.get(db, body.id)
        if listed is not None:
            api_key = model_servers.key_for(db, listed["base_url"])
    if not api_key:
        api_key = model_servers.key_for(db, base_url)
    state = endpoint_policy.endpoint_state(db, base_url)
    model = body.model if state["sendable"] else None
    result = await run_in_threadpool(check_server, body.kind, base_url, api_key, model)
    if result["video"] is not None and model:
        model_servers.record_capabilities(db, base_url, video=(model, result["video"]))
    return ModelServerCheck(
        **result,
        endpoint=ModelServerCheckEndpoint(
            host=state["host"],
            class_=state["class"],
            confirmed=state["confirmed"],
            sendable=state["sendable"],
        ),
    )
