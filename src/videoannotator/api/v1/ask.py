"""Ask AI: try a question on a video, keep every try, run it on the whole video
(spec 027).

A try is saved as a history entry before any model call and answered in a
server thread, one moment at a time, so closing the browser loses nothing
(SC-002) and the server keeps serving everyone else meanwhile. Tries go ahead
of runs on the same model (gate.py). Whether the model server is on this
computer or the local network is decided here, by address (endpoints.py), and
frames go to a remote one only after someone on the server has agreed to it.

The handlers are plain functions: FastAPI runs them in its thread pool, so the
model server and video checks below never block the event loop.
"""

from __future__ import annotations

import io
import json
import logging
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, Query, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ... import ask_history, endpoint_policy, model_servers
from ...batch.result_files import job_folder
from ...batch.types import BatchJob, JobStatus
from ...config_env import default_ollama_base_url
from ...database.database import get_db
from ...database.models import AskEntry, EndpointConfirmation, Prompt
from ...diagnostics.ollama import unreachable_hint
from ...endpoints import classify_endpoint, normalise_base_url
from ...exporters.table import (
    vlm_coco_from_answers,
    write_vlm_answers_csv,
    write_vlm_wide_csv,
)
from ...model_servers import ModelServerError
from ...pipelines.vlm_annotation import answer_shape as shapes
from ...pipelines.vlm_annotation import frames as vlm_frames
from ...pipelines.vlm_annotation import sampling as vlm_sampling
from ...pipelines.vlm_annotation.clients import (
    KeyRefusedError,
    VLMClient,
    WrongKindError,
    make_client,
)
from ...pipelines.vlm_annotation.gate import gate_for
from ...pipelines.vlm_annotation.ollama_client import (
    OllamaUnavailableError,
    OllamaVLMClient,
)
from ...pipelines.vlm_annotation.openai_client import KEY_REFUSED_MESSAGE
from ...pipelines.vlm_annotation.vlm_pipeline import (
    VLMAnnotationPipeline,
    _parse_label,
)
from ...registry.pipeline_loader import extras_available
from ...results_folder import RunFolder, run_name, sanitize_component
from ...storage.manager import get_storage_provider
from ..database import get_storage_backend
from ..errors import APIError
from ..middleware.auth import validate_api_key
from .exceptions import PipelineUnavailableException
from .videos import readable_video

logger = logging.getLogger("videoannotator.api")

router = APIRouter()

VLM_REQUIRES_EXTRAS = ["llm"]
DEFAULT_TRY_COUNT = 10
_SLACK = 1e-3

BackendName = Literal["ollama", "openai_compatible"]
ShapeIn = list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Shapes (data-model.md)
# ---------------------------------------------------------------------------


class Clip(BaseModel):
    length: float = 2.0
    fps: float = 4.0


class Sampling(BaseModel):
    interval: float = 5.0
    start: float = 0.0
    end: float | None = None
    moments: list[float] | None = None
    frames: int = 1
    spacing: float | None = None
    clip: Clip | None = Field(
        None, description="A short clip per sample instead of frames (spec 033)"
    )


class Answer(BaseModel):
    t: float
    frame_numbers: list[int] = []
    frame_times: list[float] = []
    status: Literal["answered", "error", "skipped"]
    label: str | None = None
    reasoning: str | None = None
    raw_response: str | None = None
    error: str | None = None
    total_time: float | None = None
    fields: dict[str, Any] | None = None
    fit: bool | None = None
    fit_problem: str | None = None
    clip_start: float | None = None
    clip_end: float | None = None
    clip_fps: float | None = None


class Entry(BaseModel):
    id: str
    kind: Literal["try", "run", "draft"]
    status: Literal["running", "done", "stopped", "failed"]
    prompt: str
    prompt_sha256: str
    model: str
    base_url: str
    backend: BackendName = "ollama"
    answer_shape: ShapeIn | None = None
    endpoint_class: Literal["local", "remote"]
    sampling: Sampling
    think: bool
    video_path: str
    video_name: str
    video_sha256: str | None = None
    moments: list[float] | None = None
    answers: list[Answer] | None = None
    job_id: str | None = None
    user_id: str | None = None
    user_name: str | None = Field(None, description="Who asked, by name")
    name: str | None = None
    starred: bool = False
    created_at: str
    finished_at: str | None = None
    job_status: str | None = Field(None, description="Runs: the job's status")
    progress: float | None = Field(None, description="Runs: percent done")


class TryRequest(BaseModel):
    video_path: str
    prompt: str
    model: str
    base_url: str | None = None
    backend: BackendName = "ollama"
    answer_shape: ShapeIn | None = None
    think: bool = False
    enforce_shape: bool = Field(
        False,
        description="Force the answer shape through the server's structured "
        "output (D5); off, the shape only reads the reply into fields",
    )
    sampling: Sampling = Sampling()
    at: float | None = Field(None, description='One moment: "Try on this frame"')
    count: int | None = Field(None, description='"Try on N moments"')


# ---------------------------------------------------------------------------
# Shared checks
# ---------------------------------------------------------------------------


def require_llm() -> None:
    if not extras_available(VLM_REQUIRES_EXTRAS):
        raise PipelineUnavailableException("vlm_annotation", VLM_REQUIRES_EXTRAS)


def _invalid(field: str, message: str) -> APIError:
    return APIError(
        status_code=422,
        code="INVALID_SAMPLING",
        message=message,
        details={"field": field},
    )


def check_sampling(s: Sampling, duration: float) -> None:
    if not s.interval > 0:
        raise _invalid("interval", "Sample every must be more than 0 seconds.")
    if s.start < 0 or s.start >= max(duration, _SLACK):
        raise _invalid(
            "start", f"From must be within the video (0 to {duration:.1f} s)."
        )
    if s.end is not None and not (s.start < s.end <= duration + _SLACK):
        raise _invalid(
            "end", f"To must be after From and within the video ({duration:.1f} s)."
        )
    if s.moments is not None:
        if not s.moments:
            raise _invalid("moments", "List at least one moment.")
        if any(t < 0 or t > duration + _SLACK for t in s.moments):
            raise _invalid(
                "moments", f"Every moment must be within 0 to {duration:.1f} s."
            )
    if not 1 <= s.frames <= 9:
        raise _invalid("frames", "Frames per sample must be 1 to 9.")
    if s.spacing is not None and not s.spacing > 0:
        raise _invalid("spacing", "Frames apart must be more than 0 seconds.")
    if s.clip is not None:
        if not 0 < s.clip.length <= 30:
            raise _invalid("clip", "A clip must be more than 0 and at most 30 seconds.")
        if not 0 < s.clip.fps <= 30:
            raise _invalid(
                "clip", "Clip frames per second must be more than 0, at most 30."
            )


def check_clip_backend(s: Sampling, backend: str) -> None:
    if s.clip is not None and backend == "ollama":
        raise APIError(
            status_code=422,
            code="CLIP_NOT_SUPPORTED",
            message="Ollama can't take video. Choose one frame or several, or a "
            "server that takes video.",
        )


def sampling_to_config(s: Sampling) -> dict[str, Any]:
    """The window's sampling as vlm_annotation config (pipeline-config.md)."""
    config: dict[str, Any] = {
        "frame_interval_sec": s.interval,
        "start_sec": s.start,
        "end_sec": s.end,
        "moments_sec": sorted(set(s.moments)) if s.moments is not None else None,
    }
    if s.clip is not None:
        config["sampling_mode"] = "clip"
        config["clip_length_sec"] = s.clip.length
        config["clip_fps"] = s.clip.fps
    elif s.frames == 1:
        config["sampling_mode"] = "single_frame"
    else:
        config["sampling_mode"] = "frame_burst"
        config["burst_offsets"] = vlm_sampling.burst_offsets_for(s.frames)
        config["burst_spacing_sec"] = s.spacing
    return config


def sample_points_for(s: Sampling, duration: float) -> list[float]:
    return vlm_sampling.sample_points(duration, s.interval, s.start, s.end, s.moments)


def video_info(path: str) -> dict[str, Any]:
    try:
        return vlm_frames.video_metadata(path)
    except (ValueError, OSError) as e:
        raise APIError(
            status_code=404,
            code="VIDEO_NOT_FOUND",
            message=f"Could not open the video at '{path}'",
            hint=str(e),
        ) from e


def endpoint_state(db: Session, base_url: str) -> dict[str, Any]:
    return endpoint_policy.endpoint_state(db, base_url)


def api_error(e: ModelServerError) -> APIError:
    return APIError(
        status_code=e.status,
        code=e.code,
        message=str(e),
        hint=e.hint,
        details=e.details,
    )


def require_sendable(db: Session, base_url: str) -> dict[str, Any]:
    """Frames may go to this address (spec 033 R5), or 403/409 saying why not."""
    try:
        return endpoint_policy.require_sendable(db, base_url)
    except ModelServerError as e:
        raise api_error(e) from e


def resolve_base_url(backend: str, base_url: str | None) -> str:
    if base_url:
        return base_url
    if backend == "ollama":
        return default_ollama_base_url()
    raise APIError(
        status_code=422,
        code="MISSING_BASE_URL",
        message="Say which model server to use.",
    )


def valid_shape(raw: ShapeIn | None) -> list[shapes.Field] | None:
    if raw is None:
        return None
    try:
        return shapes.validate_shape(raw)
    except ValueError as e:
        raise APIError(
            status_code=422, code="INVALID_ANSWER_SHAPE", message=str(e)
        ) from e


def client_for(
    backend: str, base_url: str, timeout: int, api_key: str | None = None
) -> VLMClient:
    """The backend's client; Ollama's is this module's name, so tests that
    replace it replace it here."""
    client = make_client(
        backend, base_url, timeout, api_key, ollama_class=OllamaVLMClient
    )
    if backend != "ollama":
        known = model_servers.capabilities_for_url(base_url)
        if known is not None and known.get("structured_output") is False:
            client.structured_output = False  # type: ignore[attr-defined]
    return client


def _key(db: Session, backend: str, base_url: str) -> str | None:
    return model_servers.key_for(db, base_url) if backend != "ollama" else None


def require_model_list(db: Session | None, backend: str, base_url: str) -> list[str]:
    """An OpenAI-compatible server's models, or the API error saying why not."""
    api_key = _key(db, backend, base_url) if db is not None else None
    try:
        return client_for(backend, base_url, 10, api_key).list_models()
    except KeyRefusedError as e:
        raise APIError(
            status_code=401,
            code="KEY_REFUSED",
            message=KEY_REFUSED_MESSAGE,
            hint="Change the key in Settings → Model servers.",
        ) from e
    except OllamaUnavailableError as e:
        hint = (
            "Check the address: an OpenAI-compatible server's usually ends in /v1."
            if isinstance(e, WrongKindError)
            else "Check the model server is running and reachable from this server."
        )
        raise APIError(
            status_code=503,
            code="MODEL_SERVER_UNREACHABLE",
            message=str(e),
            hint=hint,
        ) from e


def require_model(
    base_url: str, model: str, backend: str = "ollama", db: Session | None = None
) -> None:
    """One call to the model server: is it there, and is the model on it?"""
    if backend == "ollama":
        try:
            models = OllamaVLMClient(base_url=base_url, timeout=10).list_models()
        except OllamaUnavailableError as e:
            raise APIError(
                status_code=503,
                code="OLLAMA_UNREACHABLE",
                message=f"Cannot reach the model server at {base_url}",
                hint=unreachable_hint(base_url),
            ) from e
        if model not in models:
            raise APIError(
                status_code=422,
                code="MODEL_NOT_PULLED",
                message=f"The model '{model}' isn't on the model server.",
                hint=f"Get it with: ollama pull {model}",
                details={"models": models},
            )
        return
    models = require_model_list(db, backend, base_url)
    if model not in models:
        listed = ", ".join(models) if models else "no models"
        raise APIError(
            status_code=422,
            code="MODEL_NOT_FOUND",
            message=f"The model '{model}' isn't on the model server. "
            f"The server lists: {listed}.",
            details={"models": models},
        )


def shaped_answer(shape: list[shapes.Field] | None, raw_text: str) -> dict[str, Any]:
    """`label`, and for a shaped question its fields and fit (FR-020)."""
    if shape is None:
        return {"label": _parse_label(raw_text)}
    checked = shapes.check_reply(shape, raw_text)
    return {
        "label": checked.label,
        "fields": checked.fields,
        "fit": checked.fit,
        "fit_problem": checked.fit_problem,
    }


def note_structured_output(backend: str, base_url: str, outcome: str | None) -> None:
    """What a shaped call learnt about a listed server (R8)."""
    if backend == "ollama" or outcome is None:
        return
    try:
        model_servers.record_capabilities_for_url(
            base_url, structured_output=outcome == "enforced"
        )
    except Exception as e:
        logger.warning(f"Couldn't record what {base_url} supports: {e}")


def _user_id(user: dict[str, Any] | None) -> str | None:
    return str(user["id"]) if user and user.get("id") else None


def _entry(entry: Any) -> Entry:
    return Entry(**ask_history.to_dict(entry))


def _get_or_404(db: Session, entry_id: str) -> Any:
    entry = ask_history.get_entry(db, entry_id)
    if entry is None:
        raise APIError(
            status_code=404,
            code="ENTRY_NOT_FOUND",
            message="No such question in the history.",
        )
    return entry


# ---------------------------------------------------------------------------
# Tries
# ---------------------------------------------------------------------------


def _answer_try(
    entry_id: uuid.UUID,
    video_path: str,
    moments: list[float],
    prompt: str,
    model: str,
    base_url: str,
    think: bool,
    s: Sampling,
    backend: str = "ollama",
    shape: list[shapes.Field] | None = None,
    enforce: bool = False,
) -> None:
    """Answer each moment in turn, saving each answer as it lands."""
    import cv2

    offsets = None if s.frames == 1 else vlm_sampling.burst_offsets_for(s.frames)
    spacing = s.spacing or s.interval
    defaults = VLMAnnotationPipeline({}).config
    gate = gate_for(base_url, model)
    schema: dict[str, Any] = (
        {"schema": shapes.to_json_schema(shape)} if shape and enforce else {}
    )
    status = "done"
    with ask_history.own_session() as db:
        try:
            client = client_for(
                backend,
                base_url,
                int(defaults["request_timeout_sec"]),
                _key(db, backend, base_url),
            )
            meta = vlm_frames.video_metadata(video_path)
            cap = cv2.VideoCapture(video_path)
            try:
                for t in moments:
                    call: dict[str, Any] = dict(schema)
                    if s.clip is not None:
                        sent = vlm_frames.read_clip(
                            cap,
                            meta["fps"],
                            meta["duration"],
                            t,
                            s.clip.length,
                            s.clip.fps,
                        )
                        call.update(video=[f.jpeg for f in sent], video_fps=s.clip.fps)
                    else:
                        sent = vlm_frames.read_sample(
                            cap, meta["fps"], meta["duration"], t, offsets, spacing
                        )
                    answer: dict[str, Any] = {
                        "t": t,
                        "frame_numbers": [f.frame_number for f in sent],
                        "frame_times": [f.time for f in sent],
                    }
                    if s.clip is not None:
                        start, end = vlm_sampling.clip_span(
                            t, s.clip.length, meta["duration"]
                        )
                        answer.update(
                            clip_start=start, clip_end=end, clip_fps=s.clip.fps
                        )
                    if not sent:
                        answer.update(
                            status="skipped",
                            error=f"No readable frame at {t:.2f} s",
                        )
                    else:
                        try:
                            with gate.acquire("try"):
                                result = client.chat(
                                    model=model,
                                    prompt=prompt,
                                    images=[] if s.clip else [f.jpeg for f in sent],
                                    think=think,
                                    keep_alive=defaults["keep_alive"],
                                    options=defaults["options"],
                                    max_retries=defaults["max_retries"],
                                    retry_backoff_sec=defaults["retry_backoff_sec"],
                                    **call,
                                )
                        except KeyRefusedError:
                            answer.update(status="error", error=KEY_REFUSED_MESSAGE)
                            ask_history.add_answer(db, entry_id, answer)
                            status = "failed"
                            break
                        answer.update(
                            status="error" if result.error else "answered",
                            label=None,
                            reasoning=result.thinking or result.raw_text or None,
                            raw_response=result.raw_text or None,
                            error=result.error,
                            total_time=round(result.total_time, 4),
                        )
                        if not result.error:
                            answer.update(shaped_answer(shape, result.raw_text))
                            if shape:
                                note_structured_output(
                                    backend, base_url, result.structured_output
                                )
                    ask_history.add_answer(db, entry_id, answer)
            finally:
                cap.release()
        except Exception as e:
            logger.exception(f"Ask AI try {entry_id} failed")
            status = "failed"
            ask_history.add_answer(
                db,
                entry_id,
                {
                    "t": moments[0] if moments else 0.0,
                    "status": "error",
                    "error": str(e),
                },
            )
        ask_history.finish(db, entry_id, status)


@router.post(
    "/tries",
    response_model=Entry,
    status_code=201,
    summary="Try a question on one moment or a few, without a run",
)
def create_try(
    body: TryRequest,
    db: Session = Depends(get_db),
    user: dict[str, Any] | None = Depends(validate_api_key),
) -> Entry:
    require_llm()
    if (body.at is None) == (body.count is None):
        raise APIError(
            status_code=422,
            code="INVALID_TRY",
            message="Give exactly one of `at` (one moment) or `count` (N moments).",
        )
    path = readable_video(body.video_path)
    meta = video_info(str(path))
    duration = float(meta["duration"])
    check_sampling(body.sampling, duration)
    if body.at is not None:
        if not 0 <= body.at <= duration + _SLACK:
            raise _invalid("at", f"The moment must be within 0 to {duration:.1f} s.")
        moments = [round(body.at, 3)]
    else:
        if not body.count or body.count < 1:
            raise _invalid("count", "Try on at least one moment.")
        moments = vlm_sampling.choose_moments(
            sample_points_for(body.sampling, duration), body.count
        )
        if not moments:
            raise _invalid("start", "These settings give no moments to try.")

    check_clip_backend(body.sampling, body.backend)
    shape = valid_shape(body.answer_shape)
    base_url = resolve_base_url(body.backend, body.base_url)
    state = require_sendable(db, base_url)
    require_model(base_url, body.model, body.backend, db)

    prompt = body.prompt.replace("\r\n", "\n")
    entry = ask_history.create_entry(
        db,
        kind="try",
        prompt=prompt,
        model=body.model,
        base_url=base_url,
        endpoint_class=state["class"],
        sampling=body.sampling.model_dump(),
        think=body.think,
        video_path=str(path),
        moments=moments,
        user_id=_user_id(user),
        backend=body.backend,
        answer_shape=shapes.shape_dicts(shape) if shape else None,
    )
    response = _entry(entry)
    threading.Thread(
        target=_answer_try,
        args=(
            entry.id,
            str(path),
            moments,
            prompt,
            body.model,
            base_url,
            body.think,
            body.sampling,
            body.backend,
            shape,
            body.enforce_shape,
        ),
        name=f"ask-try-{entry.id}",
        daemon=True,
    ).start()
    return response


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


@router.get(
    "/entries/{entry_id}",
    response_model=Entry,
    summary="One history entry (polled while a try is running)",
)
def get_entry(
    entry_id: str,
    db: Session = Depends(get_db),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> Entry:
    return _entry_with_run(db, _get_or_404(db, entry_id))


class LibraryPrompt(BaseModel):
    sha256: str
    text: str
    answer_shape: ShapeIn | None = None
    name: str | None = None
    models: list[str] = []
    starred: bool = False
    last_used_at: str | None = None


class EntriesResponse(BaseModel):
    entries: list[Entry]
    library: list[LibraryPrompt]
    next_before: str | None = None


def _parse_before(before: str | None) -> datetime | None:
    if not before:
        return None
    try:
        value = datetime.fromisoformat(before.replace("Z", "+00:00"))
    except ValueError as e:
        raise APIError(
            status_code=422, code="INVALID_BEFORE", message="`before` is not a time."
        ) from e
    return value if value.tzinfo else value.replace(tzinfo=UTC)


@router.get(
    "/entries",
    response_model=EntriesResponse,
    summary="The history: every try and run, newest first, starred first",
    description="""
Entries are shared by everyone on the server; `mine=true` keeps the caller's own.
The first page lists starred entries first. `library` lists prompts from the
prompt library that no entry uses yet, so their records still appear; it is
empty when filtered by video and after the first page.
""",
)
def list_entries(
    video_path: str | None = None,
    mine: bool = False,
    starred: bool | None = None,
    limit: int = Query(50, ge=1, le=500),
    before: str | None = None,
    db: Session = Depends(get_db),
    user: dict[str, Any] | None = Depends(validate_api_key),
) -> EntriesResponse:
    before_at = _parse_before(before)
    query = db.query(AskEntry)
    if video_path:
        query = query.filter(AskEntry.video_path == video_path)
    me = _user_id(user)
    if mine:
        query = query.filter(AskEntry.user_id == me)
    if starred is not None:
        query = query.filter(AskEntry.starred.is_(starred))

    pinned: list[AskEntry] = []
    rest = query
    if starred is None:
        rest = query.filter(AskEntry.starred.is_(False))
        if before_at is None:
            pinned = (
                query.filter(AskEntry.starred.is_(True))
                .order_by(AskEntry.created_at.desc())
                .all()
            )
    if before_at is not None:
        rest = rest.filter(AskEntry.created_at < before_at)
    page = rest.order_by(AskEntry.created_at.desc()).limit(limit + 1).all()
    more = len(page) > limit
    page = page[:limit]

    library: list[LibraryPrompt] = []
    if not video_path and before_at is None:
        used = db.query(AskEntry.prompt_sha256).distinct()
        prompts = db.query(Prompt).filter(
            Prompt.hidden.is_(False), Prompt.sha256.not_in(used)
        )
        if mine:
            prompts = prompts.filter(Prompt.first_user_id == me)
        for p in prompts.order_by(Prompt.last_used_at.desc()).limit(200):
            library.append(
                LibraryPrompt(
                    sha256=p.sha256,
                    text=p.text,
                    answer_shape=p.answer_shape,
                    name=p.name,
                    models=sorted({u.model for u in p.uses}),
                    starred=bool(p.starred),
                    last_used_at=ask_history._iso(p.last_used_at),
                )
            )

    return EntriesResponse(
        entries=[_entry_with_run(db, e) for e in [*pinned, *page]],
        library=library,
        next_before=ask_history._iso(page[-1].created_at) if more and page else None,
    )


class EntryPatch(BaseModel):
    name: str | None = None
    starred: bool | None = None


@router.patch(
    "/entries/{entry_id}",
    response_model=Entry,
    summary="Name or star an entry (seen by everyone on the server)",
)
def update_entry(
    entry_id: str,
    body: EntryPatch,
    db: Session = Depends(get_db),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> Entry:
    entry = _get_or_404(db, entry_id)
    if body.name is not None:
        entry.name = body.name.strip() or None
    if body.starred is not None:
        entry.starred = body.starred
    db.commit()
    return _entry_with_run(db, entry)


@router.delete(
    "/entries/{entry_id}",
    status_code=204,
    summary="Delete an entry (its author or an admin); a run stays",
)
def delete_entry(
    entry_id: str,
    db: Session = Depends(get_db),
    user: dict[str, Any] | None = Depends(validate_api_key),
) -> Response:
    entry = _get_or_404(db, entry_id)
    me = _user_id(user)
    is_admin = bool(user and user.get("is_admin"))
    if entry.user_id and user is not None and me != entry.user_id and not is_admin:
        raise APIError(
            status_code=403,
            code="NOT_YOUR_ENTRY",
            message="Only the person who made this, or an admin, can delete it.",
        )
    db.delete(entry)
    db.commit()
    return Response(status_code=204)


class DraftRequest(BaseModel):
    video_path: str
    prompt: str
    model: str
    base_url: str | None = None
    backend: BackendName = "ollama"
    answer_shape: ShapeIn | None = None
    think: bool = False
    sampling: Sampling = Sampling()


@router.post(
    "/drafts",
    response_model=Entry,
    status_code=201,
    summary="Keep a question that hasn't been tried yet",
    description="Saved before another entry is restored over it, so it isn't lost.",
)
def create_draft(
    body: DraftRequest,
    db: Session = Depends(get_db),
    user: dict[str, Any] | None = Depends(validate_api_key),
) -> Entry:
    path = readable_video(body.video_path)
    shape = valid_shape(body.answer_shape)
    base_url = resolve_base_url(body.backend, body.base_url)
    entry = ask_history.create_entry(
        db,
        kind="draft",
        prompt=body.prompt.replace("\r\n", "\n"),
        model=body.model,
        base_url=base_url,
        endpoint_class=classify_endpoint(base_url).cls,
        sampling=body.sampling.model_dump(),
        think=body.think,
        video_path=str(path),
        user_id=_user_id(user),
        backend=body.backend,
        answer_shape=shapes.shape_dicts(shape) if shape else None,
    )
    return _entry(entry)


class SampleFrameOut(BaseModel):
    frame_number: int
    timestamp_sec: float
    jpeg_base64: str


class FramesResponse(BaseModel):
    t: float
    frames: list[SampleFrameOut]
    requested: int = Field(description="More than len(frames) when clipped")


@router.get(
    "/frames",
    response_model=FramesResponse,
    summary="The frames one sample would send, without asking a model",
)
def sample_frames(
    path: str,
    t: float,
    interval: float = 5.0,
    frames: int = 1,
    spacing: float | None = None,
    clip_length: float | None = Query(None, description="A clip's frames instead"),
    clip_fps: float = 4.0,
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> FramesResponse:
    video = readable_video(path)
    meta = video_info(str(video))
    clip = Clip(length=clip_length, fps=clip_fps) if clip_length else None
    check_sampling(
        Sampling(interval=interval, frames=frames, spacing=spacing, clip=clip),
        float(meta["duration"]),
    )
    if clip is not None:
        sent = vlm_frames.read_clip_from(str(video), t, clip.length, clip.fps)
        frames = len(
            vlm_sampling.clip_times(t, clip.length, clip.fps, meta["duration"])
        )
    else:
        offsets = None if frames == 1 else vlm_sampling.burst_offsets_for(frames)
        sent, _ = vlm_frames.read_sample_from(
            str(video), t, offsets, spacing or interval
        )
    return FramesResponse(
        t=t,
        frames=[
            SampleFrameOut(
                frame_number=f.frame_number,
                timestamp_sec=f.time,
                jpeg_base64=vlm_frames.thumbnail(f.jpeg),
            )
            for f in sent
        ],
        requested=frames,
    )


class EndpointState(BaseModel):
    base_url: str
    host: str
    class_: Literal["local", "remote"] = Field(alias="class")
    confirmed: bool
    sendable: bool = Field(
        True, description="Frames may go there now: local, or confirmed and allowed"
    )
    before_switch: bool = Field(
        False, description="Confirmed before remote servers needed switching on"
    )

    model_config = {"populate_by_name": True}


@router.get(
    "/endpoint",
    response_model=EndpointState,
    response_model_by_alias=True,
    summary="Where the model runs: on this computer or network, or remote",
    description="Works even when the model server is unreachable, so the badge "
    "always shows.",
)
def get_endpoint(
    base_url: str | None = None,
    db: Session = Depends(get_db),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> EndpointState:
    return EndpointState(**endpoint_state(db, base_url or default_ollama_base_url()))


class ConfirmRequest(BaseModel):
    base_url: str


@router.post(
    "/endpoints/confirm",
    status_code=204,
    summary="Agree that frames may be sent to this remote model server",
    description="Recorded for everyone on the server (FR-015).",
)
def confirm_endpoint(
    body: ConfirmRequest,
    db: Session = Depends(get_db),
    user: dict[str, Any] | None = Depends(validate_api_key),
) -> Response:
    key = normalise_base_url(body.base_url)
    if db.get(EndpointConfirmation, key) is None:
        db.add(
            EndpointConfirmation(
                base_url=key,
                confirmed_by=_user_id(user),
                confirmed_at=datetime.now(UTC),
            )
        )
        db.commit()
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

_RUN_STATUS = {
    "pending": "running",
    "running": "running",
    "retrying": "running",
    "completed": "done",
    "cancelled": "stopped",
    "failed": "failed",
}


def _load_job(job_id: str | None) -> Any:
    if not job_id:
        return None
    try:
        return get_storage_backend().load_job_metadata(job_id)
    except Exception:
        return None


def _entry_with_run(db: Session, entry: Any) -> Entry:
    """An entry; a run's follows its job, and is settled once the job is."""
    out = _entry(entry)
    if entry.kind != "run":
        return out
    job = _load_job(entry.job_id)
    if job is None:
        return out
    job_status = getattr(job.status, "value", job.status)
    status = _RUN_STATUS.get(str(job_status), "running")
    if status != "running" and status != entry.status:
        ask_history.finish(db, entry.id, status)
        out = _entry(entry)
    out.status = status  # type: ignore[assignment]
    out.job_status = str(job_status)
    out.progress = float(job.progress_percentage or 0)
    return out


class RunRequest(BaseModel):
    video_path: str
    prompt: str
    model: str
    base_url: str | None = None
    backend: BackendName = "ollama"
    answer_shape: ShapeIn | None = None
    think: bool = False
    enforce_shape: bool = Field(
        False,
        description="Force the answer shape through the server's structured "
        "output (D5); off, the shape only reads the reply into fields",
    )
    sampling: Sampling = Sampling()


@router.post(
    "/runs",
    response_model=Entry,
    status_code=201,
    summary="Run the question on the whole video (or its range) as a job",
    description="An ordinary vlm_annotation job, with its history entry. Stop it "
    "with POST /jobs/{job_id}/cancel: the answers so far are kept.",
)
def create_run(
    body: RunRequest,
    db: Session = Depends(get_db),
    user: dict[str, Any] | None = Depends(validate_api_key),
) -> Entry:
    require_llm()
    path = readable_video(body.video_path)
    meta = video_info(str(path))
    check_sampling(body.sampling, float(meta["duration"]))
    if not sample_points_for(body.sampling, float(meta["duration"])):
        raise _invalid("start", "These settings give no moments to sample.")
    check_clip_backend(body.sampling, body.backend)
    shape = valid_shape(body.answer_shape)
    base_url = resolve_base_url(body.backend, body.base_url)
    state = require_sendable(db, base_url)
    require_model(base_url, body.model, body.backend, db)

    prompt = body.prompt.replace("\r\n", "\n")
    vlm_config: dict[str, Any] = {
        **sampling_to_config(body.sampling),
        "prompt": prompt,
        "model": body.model,
        "base_url": base_url,
        "think": body.think,
    }
    # Only when not the default, so a run's config reads as it did before.
    if body.backend != "ollama":
        vlm_config["backend"] = body.backend
    if shape:
        vlm_config["answer_shape"] = shapes.shape_dicts(shape)
        if body.enforce_shape:
            vlm_config["enforce_shape"] = True
    config = {"vlm_annotation": vlm_config}
    job = BatchJob(
        # Read in place, never copied (spec 022).
        video_path=path,
        config=config,
        status=JobStatus.PENDING,
        selected_pipelines=["vlm_annotation"],
    )
    run = RunFolder.create(
        run_name(video=path),
        batch_id=None,
        pipelines=["vlm_annotation"],
        config=config,
    )
    run.add_video(job, path, size_bytes=path.stat().st_size)
    provider = get_storage_provider()
    provider.create_job_dir(job.job_id)
    job.storage_path = provider.get_absolute_path(job.job_id, "")
    get_storage_backend().save_job_metadata(job)

    entry = ask_history.create_entry(
        db,
        kind="run",
        prompt=prompt,
        model=body.model,
        base_url=base_url,
        endpoint_class=state["class"],
        sampling=body.sampling.model_dump(),
        think=body.think,
        video_path=str(path),
        job_id=job.job_id,
        user_id=_user_id(user),
        backend=body.backend,
        answer_shape=shapes.shape_dicts(shape) if shape else None,
    )
    return _entry_with_run(db, entry)


class Skipped(BaseModel):
    t: float
    reason: str


class RunAnswers(BaseModel):
    status: str
    planned: int
    done: int
    stopped_early: bool
    answers: list[Answer]
    skipped: list[Skipped]


# The last readable copy of each run's results, for a read that lands while the
# file is unreadable (a network share's hiccup; the rewrite itself is atomic).
_last_readable: dict[str, dict[str, Any]] = {}


def _answer_from_annotation(a: dict[str, Any]) -> Answer:
    label = a.get("label")
    status = a.get("status") or (
        "error" if isinstance(label, str) and label.startswith("ERROR:") else "answered"
    )
    numbers = [int(n) for n in a.get("context_frame_numbers") or []]
    if not numbers and a.get("frame_number") is not None:
        numbers = [int(a["frame_number"])]
    return Answer(
        t=float(a.get("timestamp_sec") or 0.0),
        frame_numbers=numbers,
        frame_times=[float(t) for t in a.get("frame_times") or []],
        status=status,
        label=None if status == "error" else label,
        reasoning=a.get("reasoning"),
        raw_response=a.get("raw_response"),
        error=label[len("ERROR: ") :]
        if status == "error" and isinstance(label, str)
        else None,
        total_time=a.get("total_time"),
        fields=a.get("fields"),
        fit=a.get("fit"),
        fit_problem=a.get("fit_problem"),
        clip_start=a.get("clip_start_sec"),
        clip_end=a.get("clip_end_sec"),
        clip_fps=a.get("clip_fps"),
    )


@router.get(
    "/runs/{job_id}/answers",
    response_model=RunAnswers,
    summary="A run's answers so far (polled while it goes)",
)
def run_answers(
    job_id: str,
    after: int = Query(0, ge=0),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> RunAnswers:
    job = _load_job(job_id)
    if job is None:
        raise APIError(
            status_code=404, code="JOB_NOT_FOUND", message=f"No run {job_id}"
        )
    job_status = str(getattr(job.status, "value", job.status))
    stem = Path(str(job.video_path)).stem
    path = job_folder(job) / f"{stem}_vlm_annotation.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        _last_readable[job_id] = data
    except (OSError, ValueError):
        data = _last_readable.get(job_id, {})

    annotations = [a for a in data.get("annotations") or [] if isinstance(a, dict)]
    progress = data.get("vlm_progress") or {}
    planned = progress.get("planned")
    if planned is None:
        planned = _planned_for(job)
    status = _RUN_STATUS.get(job_status, "running")
    # Stop takes effect between samples: until the call in progress returns
    # and the pipeline writes its last answer, the run is still going.
    if status == "stopped" and data and progress.get("finished") is False:
        status = "running"
    return RunAnswers(
        status=status,
        planned=int(planned or 0),
        done=int(progress.get("done") or len(annotations)) if data else 0,
        stopped_early=bool(progress.get("stopped_early")),
        answers=[_answer_from_annotation(a) for a in annotations[after:]],
        skipped=[
            Skipped(t=float(s.get("t", 0.0)), reason=str(s.get("reason", "")))
            for s in data.get("vlm_skipped") or []
            if isinstance(s, dict)
        ],
    )


def _planned_for(job: Any) -> int:
    """Before the first answer: the samples the run's config will take."""
    try:
        config = (job.config or {}).get("vlm_annotation", {})
        duration = float(vlm_frames.video_metadata(str(job.video_path))["duration"])
        return len(
            vlm_sampling.sample_points(
                duration,
                float(config.get("frame_interval_sec", 5.0)),
                float(config.get("start_sec") or 0.0),
                config.get("end_sec"),
                config.get("moments_sec"),
            )
        )
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


@router.get(
    "/entries/{entry_id}/export/table",
    summary="The entry's answers as the spec 025 table (CSV)",
    description="A run's entry redirects to its job's table.",
    response_class=Response,
)
def export_entry_table(
    entry_id: str,
    db: Session = Depends(get_db),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> Response:
    entry = _get_or_404(db, entry_id)
    if entry.kind == "run" and entry.job_id:
        return RedirectResponse(
            f"/api/v1/jobs/{entry.job_id}/export/table", status_code=307
        )
    out = io.StringIO()
    stem = Path(entry.video_path).stem
    write_vlm_answers_csv(
        entry.answers or [],
        video_stem=stem,
        model=entry.model,
        out=out,
        backend=entry.backend or "ollama",
    )
    name = sanitize_component(entry.name or stem, "answers")
    return Response(
        content=out.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{name}_answers.csv"'},
    )


def _wide_response(content: str, name: str) -> Response:
    return Response(
        content=content,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{name}_answers_wide.csv"'
        },
    )


def _not_shaped() -> APIError:
    return APIError(
        status_code=422,
        code="NOT_SHAPED",
        message="This question has no answer shape, so its answers have no fields.",
        hint="Download the table instead.",
    )


@router.get(
    "/entries/{entry_id}/export/wide",
    summary="A shaped question's answers, one column per field (CSV)",
    description="A run's entry redirects to its job's wide file.",
    response_class=Response,
)
def export_entry_wide(
    entry_id: str,
    db: Session = Depends(get_db),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> Response:
    entry = _get_or_404(db, entry_id)
    if not entry.answer_shape:
        raise _not_shaped()
    if entry.kind == "run" and entry.job_id:
        return RedirectResponse(
            f"/api/v1/jobs/{entry.job_id}/export/wide", status_code=307
        )
    stem = Path(entry.video_path).stem
    coco = vlm_coco_from_answers(
        entry.answers or [],
        video_stem=stem,
        model=entry.model,
        backend=entry.backend or "ollama",
    )
    out = io.StringIO()
    write_vlm_wide_csv(
        coco["annotations"], entry.answer_shape, video=entry.video_name, out=out
    )
    return _wide_response(
        out.getvalue(), sanitize_component(entry.name or stem, "answers")
    )


def export_job_wide(job: Any) -> Response:
    """The wide file for a vlm_annotation run (GET /jobs/{id}/export/wide)."""
    path = job_folder(job) / f"{Path(str(job.video_path)).stem}_vlm_annotation.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise APIError(
            status_code=404,
            code="NO_VLM_RESULTS",
            message="This run has no answers to export yet.",
        ) from e
    shape = data.get("vlm_answer_shape")
    if not shape:
        raise _not_shaped()
    out = io.StringIO()
    write_vlm_wide_csv(
        [a for a in data.get("annotations") or [] if isinstance(a, dict)],
        shape,
        video=Path(str(job.video_path)).name,
        out=out,
    )
    return _wide_response(
        out.getvalue(), sanitize_component(Path(str(job.video_path)).stem, "answers")
    )


# ---------------------------------------------------------------------------
# Suggest an answer shape (research R11)
# ---------------------------------------------------------------------------

SUGGEST_PROMPT = """\
A researcher asks a vision model this question about moments in a video:

---
{question}
---

Propose the answer's fields: one field for each thing the question asks, as a
researcher would want separate columns. Each field has a name in plain words, as
a column heading ("Adults", "Child smiling"), a kind (choice, yes_no, number or
text), values for a choice in plain words, and min and max for a number when
the question gives a range. Don't merge two things asked into one field."""

# The proposal's own shape, sent as the schema so the reply parses.
SUGGEST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "fields": {
            "type": "array",
            "minItems": 1,
            "maxItems": shapes.MAX_FIELDS,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "kind": {"type": "string", "enum": list(shapes.KINDS)},
                    "values": {"type": "array", "items": {"type": "string"}},
                    "min": {"type": "number"},
                    "max": {"type": "number"},
                },
                "required": ["name", "kind"],
            },
        }
    },
    "required": ["fields"],
}


class SuggestShapeRequest(BaseModel):
    prompt: str
    model: str | None = None
    base_url: str | None = None
    backend: BackendName = "ollama"


class SuggestShapeResponse(BaseModel):
    shape: ShapeIn | None = None
    made_by: Literal["rule", "model"] | None = None
    message: str


def _model_proposal(reply: str) -> list[dict[str, Any]] | None:
    parsed = shapes.first_object(reply)
    raw = parsed.get("fields") if parsed else None
    if not isinstance(raw, list):
        return None
    cleaned = []
    for item in raw:
        if not isinstance(item, dict):
            return None
        kind = item.get("kind")
        field = {"name": item.get("name"), "kind": kind}
        if kind == "choice":
            field["values"] = item.get("values")
        if kind == "number":
            for k in ("min", "max"):
                if item.get(k) is not None:
                    field[k] = item[k]
        cleaned.append(field)
    try:
        return shapes.shape_dicts(shapes.validate_shape(cleaned))
    except ValueError:
        return None


@router.post(
    "/suggest-shape",
    response_model=SuggestShapeResponse,
    summary="Suggest an answer shape for a question, for the user to edit",
    description="""
First by rule, from the options the question lists ("Return ONLY: A or B",
"(yes/no)", "on a scale of 1 to 5"). Only when the rule finds nothing and a
model is given, one text-only call asks the model: never any frames. The
suggestion is never applied on its own (FR-024).
""",
)
def suggest_shape(
    body: SuggestShapeRequest,
    db: Session = Depends(get_db),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> SuggestShapeResponse:
    by_rule = shapes.suggest_by_rule(body.prompt)
    if by_rule:
        return SuggestShapeResponse(
            shape=shapes.shape_dicts(by_rule),
            made_by="rule",
            message="Suggested from your question.",
        )
    if not body.model:
        return SuggestShapeResponse(
            message="Your question doesn't list its answers. Add fields yourself, "
            "or choose a model to suggest some."
        )
    require_llm()
    base_url = resolve_base_url(body.backend, body.base_url)
    require_sendable(db, base_url)
    defaults = VLMAnnotationPipeline({}).config
    client = client_for(
        body.backend,
        base_url,
        int(defaults["request_timeout_sec"]),
        _key(db, body.backend, base_url),
    )
    try:
        with gate_for(base_url, body.model).acquire("try"):
            result = client.chat(
                model=body.model,
                prompt=SUGGEST_PROMPT.format(question=body.prompt.strip()),
                images=[],
                think=False,
                keep_alive=defaults["keep_alive"],
                options=defaults["options"],
                max_retries=1,
                retry_backoff_sec=0,
                schema=SUGGEST_SCHEMA,
            )
    except KeyRefusedError as e:
        raise APIError(
            status_code=401, code="KEY_REFUSED", message=KEY_REFUSED_MESSAGE
        ) from e
    if result.error:
        raise APIError(
            status_code=502,
            code="VLM_CALL_FAILED",
            message=f"The model couldn't suggest fields: {result.error}",
        )
    proposal = _model_proposal(result.raw_text)
    if proposal is None:
        return SuggestShapeResponse(
            message="The model's suggestion wasn't a usable answer shape. "
            "Add fields yourself."
        )
    return SuggestShapeResponse(
        shape=proposal, made_by="model", message="Suggested by the model."
    )


# ---------------------------------------------------------------------------
# Remembered settings (spec 033: the chosen model server, per user)
# ---------------------------------------------------------------------------

ASK_SETTINGS = "ask_settings"


class AskSettings(BaseModel):
    server: str = Field(model_servers.DEFAULT_ID, description="A listed server's id")


@router.get(
    "/settings",
    response_model=AskSettings,
    summary="The caller's remembered Ask AI settings",
)
def get_ask_settings(
    db: Session = Depends(get_db),
    user: dict[str, Any] | None = Depends(validate_api_key),
) -> AskSettings:
    from ...database.models import UserPreference
    from ...run_settings import user_key

    row = db.get(UserPreference, (user_key(user), ASK_SETTINGS))
    value = row.value if row is not None and isinstance(row.value, dict) else {}
    server = str(value.get("server") or model_servers.DEFAULT_ID)
    if model_servers.get(db, server) is None:
        server = model_servers.DEFAULT_ID
    return AskSettings(server=server)


@router.put(
    "/settings",
    response_model=AskSettings,
    summary="Remember the caller's Ask AI settings",
)
def put_ask_settings(
    body: AskSettings,
    db: Session = Depends(get_db),
    user: dict[str, Any] | None = Depends(validate_api_key),
) -> AskSettings:
    from ...run_settings import _set_preference, user_key

    _set_preference(db, user_key(user), ASK_SETTINGS, body.model_dump())
    db.commit()
    return body
