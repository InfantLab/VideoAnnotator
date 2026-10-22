"""VLM prompt-workflow endpoints for VideoAnnotator API (spec 009).

Lets a researcher iterating on a vlm_annotation prompt test it against a
single frame (or burst) synchronously, and discover which models are
actually pulled on the configured Ollama server -- without submitting a
full job. Reuses vlm_pipeline.py's own frame-extraction/label-parsing code
and ollama_client.py's own model-calling code (FR-002), so a preview result
is representative of what a real job would actually produce.
"""

import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from ... import model_servers
from ...config_env import default_ollama_base_url
from ...database.database import get_db
from ...diagnostics.ollama import unreachable_hint
from ...pipelines.vlm_annotation import answer_shape as shapes
from ...pipelines.vlm_annotation import frames as vlm_frames
from ...pipelines.vlm_annotation.clients import KeyRefusedError, make_client
from ...pipelines.vlm_annotation.ollama_client import (
    OllamaUnavailableError,
    OllamaVLMClient,
)
from ...pipelines.vlm_annotation.openai_client import KEY_REFUSED_MESSAGE
from ...pipelines.vlm_annotation.vlm_pipeline import (
    DEFAULT_PROMPT,
    VLMAnnotationPipeline,
    _parse_label,
)
from ...prompt_library import record_use_quietly
from ...registry.pipeline_loader import extras_available
from ..errors import APIError
from ..middleware.auth import validate_api_key
from .exceptions import PipelineUnavailableException

# ollama_client.py defers its own `import ollama` (only installed under the
# `llm` extra, 004-extras-based-install), so importing it here is always
# safe -- but every endpoint below still needs its own extras_available()
# check so a missing `llm` extra surfaces as the standard 422
# PipelineUnavailableException instead of an unhandled ImportError.
VLM_REQUIRES_EXTRAS = ["llm"]

logger = logging.getLogger("videoannotator.api")

router = APIRouter()


class VlmEndpoint(BaseModel):
    host: str
    class_: str = Field(alias="class", description="local or remote")
    confirmed: bool

    model_config = {"populate_by_name": True}


class VlmModelsResponse(BaseModel):
    base_url: str
    models: list[str]
    endpoint: VlmEndpoint | None = None
    server: str = Field("default", description="The listed server's id")
    kind: str = Field("ollama", description="ollama or openai_compatible")


@router.get(
    "/models",
    response_model=VlmModelsResponse,
    response_model_by_alias=True,
    summary="List models available on the configured Ollama server",
    description="""
Lists the vision-language models currently pulled on the configured Ollama
server (FR-004). Distinguishes "server unreachable" (503) from "reachable
but zero models pulled" (200 with an empty list) -- both are valid states a
researcher needs to tell apart, not one generic failure (FR-005).

`endpoint` says whether the server is on this computer or local network
(`local`) or elsewhere (`remote`), decided by address (spec 027 R9).
""",
)
def list_vlm_models(
    base_url: str | None = None,
    server: str | None = Query(
        None, description="A listed model server's id (spec 033); `default` if none"
    ),
    db: Session = Depends(get_db),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> VlmModelsResponse:
    """List models pulled on the configured Ollama server, or on a listed one."""
    if not extras_available(VLM_REQUIRES_EXTRAS):
        raise PipelineUnavailableException("vlm_annotation", VLM_REQUIRES_EXTRAS)
    from .ask import endpoint_state, require_model_list

    kind = "ollama"
    server_id = "default"
    if server and server != model_servers.DEFAULT_ID:
        listed = model_servers.get(db, server)
        if listed is None:
            raise APIError(
                status_code=404,
                code="SERVER_NOT_FOUND",
                message="No such model server.",
            )
        base_url, kind, server_id = listed["base_url"], listed["kind"], listed["id"]
    base_url = base_url or default_ollama_base_url()

    if kind == "ollama":
        try:
            client = OllamaVLMClient(base_url=base_url, timeout=10)
            models = client.list_models()
        except OllamaUnavailableError as e:
            raise APIError(
                status_code=503,
                code="OLLAMA_UNREACHABLE",
                message=f"Cannot reach Ollama server at {base_url}",
                hint=f"{e} {unreachable_hint(base_url)}",
            ) from e
    else:
        models = require_model_list(db, kind, base_url)

    state = endpoint_state(db, base_url)
    return VlmModelsResponse(
        base_url=base_url,
        models=models,
        endpoint=VlmEndpoint(
            host=state["host"], class_=state["class"], confirmed=state["confirmed"]
        ),
        server=server_id,
        kind=kind,
    )


class PreviewFrame(BaseModel):
    """A frame the model was shown, small, so a client can show it too."""

    frame_number: int | None = None
    timestamp_sec: float | None = None
    jpeg_base64: str


class VlmPreviewResponse(BaseModel):
    label: str
    reasoning: str
    raw_response: str
    total_time: float
    load_time: float | None
    prompt_tokens: int
    resp_tokens: int
    tokens_per_sec: float | None
    frames: list[PreviewFrame] = []


def _throwaway_pipeline(base_url: str) -> VLMAnnotationPipeline:
    """A VLMAnnotationPipeline instance used only for its frame-extraction
    helpers (_get_video_metadata/_read_single/_read_burst) -- never
    .initialize()'d, so it never preflights/holds an Ollama client of its
    own. Preview makes its own OllamaVLMClient call directly below, since a
    preview's model/prompt may differ from any job's saved config."""
    return VLMAnnotationPipeline({"base_url": base_url})


@router.post(
    "/preview",
    response_model=VlmPreviewResponse,
    summary="Test a prompt against one frame (or burst) without creating a job",
    description="""
Runs `prompt` against `model` for a single frame -- either an uploaded
image, or a frame extracted from an already-uploaded video at
`timestamp_sec` -- and returns the label/reasoning synchronously (FR-001).
Nothing is persisted: no job, no annotation record (FR-001, spec's
Assumptions).

`sampling_mode="frame_burst"` requires `video_path`+`timestamp_sec` (an
uploaded still image can't be burst-sampled) and reuses the real pipeline's
own burst-window/clamping logic exactly (edge case).
""",
)
async def preview_vlm_prompt(
    image: UploadFile | None = File(
        None, description="A single frame image to test the prompt against"
    ),
    video_path: str | None = Form(
        None, description="Path to an already-uploaded video (alternative to `image`)"
    ),
    timestamp_sec: float | None = Form(
        None, description="Timestamp within `video_path` to extract the frame(s) from"
    ),
    prompt: str = Form(DEFAULT_PROMPT),
    model: str = Form(...),
    sampling_mode: str = Form("single_frame"),
    frame_interval_sec: float = Form(5.0),
    burst_offsets: str | None = Form(
        None, description="JSON-encoded list[int], e.g. '[-2,-1,0,1,2]'"
    ),
    burst_spacing_sec: float | None = Form(
        None,
        description="Seconds between burst frames; frame_interval_sec when omitted",
    ),
    think: bool = Form(False),
    base_url: str | None = Form(None),
    backend: str = Form("ollama", description="ollama or openai_compatible"),
    answer_shape: str | None = Form(
        None, description="JSON-encoded answer shape (spec 033), or none"
    ),
    db: Session = Depends(get_db),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> VlmPreviewResponse:
    """Test a prompt against a single frame or burst, synchronously."""
    if not extras_available(VLM_REQUIRES_EXTRAS):
        raise PipelineUnavailableException("vlm_annotation", VLM_REQUIRES_EXTRAS)
    from .ask import _key, require_sendable, resolve_base_url, valid_shape

    if backend not in ("ollama", "openai_compatible"):
        raise APIError(
            status_code=422,
            code="INVALID_BACKEND",
            message=f"Unknown model server kind '{backend}'",
            hint="Use 'ollama' or 'openai_compatible'",
        )
    try:
        shape_raw = json.loads(answer_shape) if answer_shape else None
    except ValueError as e:
        raise APIError(
            status_code=422,
            code="INVALID_ANSWER_SHAPE",
            message="The answer shape isn't JSON.",
        ) from e
    shape = valid_shape(shape_raw)
    base_url = resolve_base_url(backend, base_url)
    require_sendable(db, base_url)
    api_key = _key(db, backend, base_url)
    # Browsers send every newline of a multipart form field as CRLF, so the
    # same prompt typed in the viewer would otherwise reach the model (and
    # the prompt library) as different text from the one a job runs.
    prompt = prompt.replace("\r\n", "\n")

    if sampling_mode not in ("single_frame", "frame_burst"):
        raise APIError(
            status_code=422,
            code="INVALID_SAMPLING_MODE",
            message=f"Unknown sampling_mode '{sampling_mode}'",
            hint="Use 'single_frame' or 'frame_burst'",
        )

    have_image = image is not None
    have_video_ref = video_path is not None and timestamp_sec is not None

    if have_image == have_video_ref:
        raise APIError(
            status_code=422,
            code="INVALID_PREVIEW_SOURCE",
            message="Provide exactly one of: an uploaded `image`, or both "
            "`video_path` and `timestamp_sec`.",
        )

    if sampling_mode == "frame_burst" and have_image:
        raise APIError(
            status_code=422,
            code="BURST_REQUIRES_VIDEO_REFERENCE",
            message="frame_burst sampling needs multiple frames around a "
            "timestamp; it isn't supported for a single uploaded image.",
            hint="Provide video_path + timestamp_sec instead of image.",
        )

    image_bytes = await image.read() if image is not None else None
    # cv2 and the model call block for seconds to minutes: off the event loop,
    # so the server keeps answering everyone else meanwhile.
    return await run_in_threadpool(
        _answer_preview,
        image_bytes=image_bytes,
        video_path=video_path,
        timestamp_sec=timestamp_sec,
        prompt=prompt,
        model=model,
        sampling_mode=sampling_mode,
        frame_interval_sec=frame_interval_sec,
        burst_offsets=burst_offsets,
        burst_spacing_sec=burst_spacing_sec,
        think=think,
        base_url=base_url,
        user_id=str(_user["id"]) if _user and _user.get("id") else None,
        backend=backend,
        shape=shape,
        api_key=api_key,
    )


def _answer_preview(
    *,
    image_bytes: bytes | None,
    video_path: str | None,
    timestamp_sec: float | None,
    prompt: str,
    model: str,
    sampling_mode: str,
    frame_interval_sec: float,
    burst_offsets: str | None,
    burst_spacing_sec: float | None,
    think: bool,
    base_url: str,
    user_id: str | None,
    backend: str = "ollama",
    shape: list[shapes.Field] | None = None,
    api_key: str | None = None,
) -> VlmPreviewResponse:
    images: list[bytes]
    frame_numbers: list[int | None]
    fps = 0.0
    if image_bytes is not None:
        images = [image_bytes]
        frame_numbers = [None]
    else:
        frame_numbers, images, fps = _extract_video_frames(
            video_path=video_path,  # type: ignore[arg-type]
            timestamp_sec=timestamp_sec,  # type: ignore[arg-type]
            sampling_mode=sampling_mode,
            spacing_sec=burst_spacing_sec or frame_interval_sec,
            burst_offsets=burst_offsets,
            base_url=base_url,
        )

    if backend == "ollama":
        client: Any = OllamaVLMClient(base_url=base_url, timeout=240)
    else:
        client = make_client(backend, base_url, 240, api_key)
    default_config = VLMAnnotationPipeline({}).config
    extra = {"schema": shapes.to_json_schema(shape)} if shape else {}
    try:
        result = client.chat(
            model=model,
            prompt=prompt,
            images=images,
            think=think,
            keep_alive=default_config["keep_alive"],
            options=default_config["options"],
            max_retries=default_config["max_retries"],
            retry_backoff_sec=default_config["retry_backoff_sec"],
            **extra,
        )
    except KeyRefusedError as e:
        raise APIError(
            status_code=401, code="KEY_REFUSED", message=KEY_REFUSED_MESSAGE
        ) from e

    if result.error:
        if backend == "ollama":
            raise APIError(
                status_code=502,
                code="VLM_CALL_FAILED",
                message=f"Ollama call failed: {result.error}",
                hint=f"Check the model '{model}' is pulled and Ollama at "
                f"{base_url} is reachable.",
            )
        raise APIError(
            status_code=502,
            code="VLM_CALL_FAILED",
            message=f"The model server call failed: {result.error}",
            hint=f"Check the model '{model}' is on the server at {base_url}.",
        )

    record_use_quietly(prompt, model, "preview", user_id=user_id)
    label = (
        shapes.check_reply(shape, result.raw_text).label
        if shape
        else _parse_label(result.raw_text)
    )
    return VlmPreviewResponse(
        frames=[
            PreviewFrame(
                frame_number=n,
                timestamp_sec=round(n / fps, 3) if n is not None and fps else None,
                jpeg_base64=vlm_frames.thumbnail(image),
            )
            for n, image in zip(frame_numbers, images, strict=True)
        ],
        label=label,
        reasoning=result.thinking or result.raw_text,
        raw_response=result.raw_text,
        total_time=round(result.total_time, 4),
        load_time=None if result.load_time is None else round(result.load_time, 4),
        prompt_tokens=result.prompt_tokens,
        resp_tokens=result.resp_tokens,
        tokens_per_sec=None
        if result.tokens_per_sec is None
        else round(result.tokens_per_sec, 2),
    )


def _extract_video_frames(
    *,
    video_path: str,
    timestamp_sec: float,
    sampling_mode: str,
    spacing_sec: float,
    burst_offsets: str | None,
    base_url: str,
) -> tuple[list[int | None], list[bytes], float]:
    """Extract the frame(s) for a preview from an already-uploaded video,
    reusing VLMAnnotationPipeline's own extraction methods exactly (FR-002,
    edge cases around out-of-range timestamps and burst clamping)."""
    pipeline = _throwaway_pipeline(base_url)

    try:
        video_metadata = pipeline._get_video_metadata(video_path)
    except (ValueError, OSError) as e:
        raise APIError(
            status_code=404,
            code="VIDEO_NOT_FOUND",
            message=f"Could not open video at '{video_path}'",
            hint=str(e),
        ) from e

    duration = video_metadata["duration"]
    if timestamp_sec < 0 or timestamp_sec > duration:
        raise APIError(
            status_code=422,
            code="TIMESTAMP_OUT_OF_RANGE",
            message=f"timestamp_sec={timestamp_sec} is outside this video's "
            f"duration (0-{duration:.2f}s)",
        )

    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise APIError(
            status_code=404,
            code="VIDEO_NOT_FOUND",
            message=f"Could not open video at '{video_path}'",
        )

    try:
        _context_offsets: list[int] | None
        if sampling_mode == "frame_burst":
            offsets = (
                json.loads(burst_offsets)
                if burst_offsets
                else VLMAnnotationPipeline({}).config["burst_offsets"]
            )
            frame_numbers, images, _context_offsets = pipeline._read_burst(
                cap, video_metadata, timestamp_sec, spacing_sec, offsets
            )
        else:
            frame_numbers, images, _context_offsets = pipeline._read_single(
                cap, video_metadata, timestamp_sec
            )
    finally:
        cap.release()

    if not images:
        raise APIError(
            status_code=422,
            code="NO_READABLE_FRAME",
            message=f"Could not read any frame at t={timestamp_sec:.2f}s "
            f"in '{video_path}'",
        )
    return list(frame_numbers), images, video_metadata["fps"]
