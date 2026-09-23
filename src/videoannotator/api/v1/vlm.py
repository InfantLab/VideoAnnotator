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

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel

from ...pipelines.vlm_annotation.ollama_client import (
    OllamaUnavailableError,
    OllamaVLMClient,
)
from ...pipelines.vlm_annotation.vlm_pipeline import (
    DEFAULT_PROMPT,
    VLMAnnotationPipeline,
    _parse_label,
)
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

DEFAULT_BASE_URL = "http://127.0.0.1:11434"


class VlmModelsResponse(BaseModel):
    base_url: str
    models: list[str]


@router.get(
    "/models",
    response_model=VlmModelsResponse,
    summary="List models available on the configured Ollama server",
    description="""
Lists the vision-language models currently pulled on the configured Ollama
server (FR-004). Distinguishes "server unreachable" (503) from "reachable
but zero models pulled" (200 with an empty list) -- both are valid states a
researcher needs to tell apart, not one generic failure (FR-005).
""",
)
async def list_vlm_models(
    base_url: str = DEFAULT_BASE_URL,
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> VlmModelsResponse:
    """List models pulled on the configured Ollama server."""
    if not extras_available(VLM_REQUIRES_EXTRAS):
        raise PipelineUnavailableException("vlm_annotation", VLM_REQUIRES_EXTRAS)

    try:
        client = OllamaVLMClient(base_url=base_url, timeout=10)
        models = client.list_models()
    except OllamaUnavailableError as e:
        raise APIError(
            status_code=503,
            code="OLLAMA_UNREACHABLE",
            message=f"Cannot reach Ollama server at {base_url}",
            hint=str(e),
        ) from e
    return VlmModelsResponse(base_url=base_url, models=models)


class VlmPreviewResponse(BaseModel):
    label: str
    reasoning: str
    raw_response: str
    total_time: float
    load_time: float
    prompt_tokens: int
    resp_tokens: int
    tokens_per_sec: float


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
    think: bool = Form(False),
    base_url: str = Form(DEFAULT_BASE_URL),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> VlmPreviewResponse:
    """Test a prompt against a single frame or burst, synchronously."""
    if not extras_available(VLM_REQUIRES_EXTRAS):
        raise PipelineUnavailableException("vlm_annotation", VLM_REQUIRES_EXTRAS)

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

    images: list[bytes]
    if have_image:
        assert image is not None
        images = [await image.read()]
    else:
        images = _extract_video_frames(
            video_path=video_path,  # type: ignore[arg-type]
            timestamp_sec=timestamp_sec,  # type: ignore[arg-type]
            sampling_mode=sampling_mode,
            frame_interval_sec=frame_interval_sec,
            burst_offsets=burst_offsets,
            base_url=base_url,
        )

    client = OllamaVLMClient(base_url=base_url, timeout=240)
    default_config = VLMAnnotationPipeline({}).config
    result = client.chat(
        model=model,
        prompt=prompt,
        images=images,
        think=think,
        keep_alive=default_config["keep_alive"],
        options=default_config["options"],
        max_retries=default_config["max_retries"],
        retry_backoff_sec=default_config["retry_backoff_sec"],
    )

    if result.error:
        raise APIError(
            status_code=502,
            code="VLM_CALL_FAILED",
            message=f"Ollama call failed: {result.error}",
            hint=f"Check the model '{model}' is pulled and Ollama at "
            f"{base_url} is reachable.",
        )

    label = _parse_label(result.raw_text)
    return VlmPreviewResponse(
        label=label,
        reasoning=result.thinking or result.raw_text,
        raw_response=result.raw_text,
        total_time=round(result.total_time, 4),
        load_time=round(result.load_time, 4),
        prompt_tokens=result.prompt_tokens,
        resp_tokens=result.resp_tokens,
        tokens_per_sec=round(result.tokens_per_sec, 2),
    )


def _extract_video_frames(
    *,
    video_path: str,
    timestamp_sec: float,
    sampling_mode: str,
    frame_interval_sec: float,
    burst_offsets: str | None,
    base_url: str,
) -> list[bytes]:
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
        if sampling_mode == "frame_burst":
            offsets = (
                json.loads(burst_offsets)
                if burst_offsets
                else VLMAnnotationPipeline({}).config["burst_offsets"]
            )
            _frame_numbers, images, _context_offsets = pipeline._read_burst(
                cap, video_metadata, timestamp_sec, frame_interval_sec, offsets
            )
        else:
            _frame_numbers, images, _context_offsets = pipeline._read_single(
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
    return images
