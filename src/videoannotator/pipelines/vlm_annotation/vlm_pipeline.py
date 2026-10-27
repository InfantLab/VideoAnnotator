"""Local vision-language-model frame annotation pipeline.

Samples frames from a video at a fixed cadence and classifies/describes each
one with a VLM on a model server (Ollama, or any OpenAI-compatible server:
spec 033), using a user-supplied prompt
applied identically and independently to every sample point (see
ollama_client.py for why "independently" matters). Ported from
mother-infant-touch-detection's core/inference.py (single_frame mode) and
core/context_inference.py (frame_burst mode) — that repo's own research
roadmap treats the choice between those two sampling modes as an open
question, so both are supported here rather than picking one.
"""

import hashlib
import logging
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from videoannotator.config_env import default_ollama_base_url
from videoannotator.endpoints import classify_endpoint
from videoannotator.exporters.native_formats import (
    create_coco_annotation,
    create_coco_image_entry,
    export_coco_json,
    validate_coco_json,
)
from videoannotator.pipelines.base_pipeline import BasePipeline
from videoannotator.provenance import _SECRET_KEY, ModelRef

from . import answer_shape as shapes
from . import frames, sampling
from .clients import BACKENDS, KeyRefusedError, VLMClient, make_client
from .gate import gate_for
from .ollama_client import OllamaVLMClient, VLMCallResult

logger = logging.getLogger(__name__)

# Default prompt: ported verbatim from
# mother-infant-touch-detection/prompts/maternal_perspective.txt — Irene's
# current best-performing single-frame binary touch-detection prompt
# (docs/research_roadmap.md §3). Just a sensible default; any prompt works.
DEFAULT_PROMPT = (
    "Classify as TOUCH only if:\n"
    "The mother's hand or arm is actually in physical contact with the infant's body.\n"
    "\n"
    "Classify as NO_TOUCH if:\n"
    "The mother's hand or arm visually overlaps the infant in the image, but one is "
    "clearly in front of or behind the other (the apparent contact is camera "
    "perspective, not real contact).\n"
    "The mother's hand or arm is near but not actually touching the infant.\n"
    "\n"
    "Return ONLY: TOUCH or NO_TOUCH."
)

# Generic label extraction: matches the touch-detection labels out of the box,
# but falls back to a truncated raw response for prompts using different
# vocabulary — this pipeline isn't touch-detection-specific.
_LABEL_RE = re.compile(
    r"\b(NO_TOUCH|TOUCH|MATERNAL_TOUCH|INFANT_TOUCH|BOTH_TOUCH|YES|NO)\b",
    re.IGNORECASE,
)


def _parse_label(text: str | None) -> str:
    if not text or not text.strip():
        return "EMPTY"
    m = _LABEL_RE.search(text)
    if m:
        return m.group(1).upper()
    return re.sub(r"\s+", " ", text.strip())[:200]


class ConfigProblem(ValueError):
    """A vlm_annotation config that can't run, with the API's error code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


KEY_IN_CONFIG_MESSAGE = (
    "Keys don't go in a run's settings. Add the key in Settings → Model servers, "
    "or name an environment variable there."
)


def _secret_keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        found = [
            k
            for k, v in value.items()
            if isinstance(k, str) and _SECRET_KEY.search(k) and isinstance(v, str) and v
        ]
        for v in value.values():
            found += _secret_keys(v)
        return found
    if isinstance(value, list):
        return [k for v in value for k in _secret_keys(v)]
    return []


def check_config(config: dict[str, Any]) -> None:
    """The checks every route runs before a run starts (contracts/
    pipeline-config.md): at submission and again in `initialize()`."""
    if _secret_keys(config):
        raise ConfigProblem("KEY_IN_CONFIG", KEY_IN_CONFIG_MESSAGE)
    backend = config.get("backend") or "ollama"
    if backend not in BACKENDS:
        raise ConfigProblem(
            "INVALID_BACKEND",
            f"Unknown model server kind '{backend}': use ollama or openai_compatible.",
        )
    if backend == "openai_compatible" and not config.get("base_url"):
        raise ConfigProblem("MISSING_BASE_URL", "Say which model server to use.")
    if backend == "ollama" and config.get("sampling_mode") == "clip":
        raise ConfigProblem(
            "CLIP_NOT_SUPPORTED",
            "Ollama can't take video. Choose one frame or several, or a server "
            "that takes video.",
        )
    if config.get("answer_shape") is not None:
        try:
            shapes.validate_shape(config["answer_shape"])
        except ValueError as e:
            raise ConfigProblem("INVALID_ANSWER_SHAPE", str(e)) from e


def _server_key(base_url: str) -> str | None:
    """The key Settings → Model servers holds for this address, if any. Read
    through its own session: a run happens outside any request."""
    from videoannotator import model_servers

    return model_servers.key_for_url(base_url)


def _require_sendable(base_url: str) -> None:
    """Remote addresses need the switch and a confirmation, however the run
    arrived (spec 033 R5): checked before any frame is read."""
    if classify_endpoint(base_url).cls == "local":
        return
    from videoannotator import endpoint_policy
    from videoannotator.model_servers import ModelServerError

    try:
        endpoint_policy.require_sendable_for_url(base_url)
    except ModelServerError as e:
        raise ConfigProblem(e.code, str(e)) from e


def _server_capabilities(base_url: str) -> dict[str, Any] | None:
    from videoannotator import model_servers

    try:
        return model_servers.capabilities_for_url(base_url)
    except Exception:  # an older database: nothing known yet
        return None


def _record_structured_output(base_url: str, enforced: bool) -> None:
    """What a shaped run learnt about a listed server (R8): later calls go
    straight to describing the shape if it can't enforce it."""
    from videoannotator import model_servers

    try:
        model_servers.record_capabilities_for_url(base_url, structured_output=enforced)
    except Exception as e:
        logger.warning(f"Couldn't record what {base_url} supports: {e}")


def _rounded(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


class VLMAnnotationPipeline(BasePipeline):
    """Per-frame vision-language-model classification via a model server."""

    def __init__(self, config: dict[str, Any] | None = None):
        default_config: dict[str, Any] = {
            "prompt": DEFAULT_PROMPT,
            "backend": "ollama",  # or "openai_compatible" (spec 033)
            "base_url": "",  # empty, with Ollama: the server's default_ollama_base_url()
            "model": "qwen3.5:9b",
            "sampling_mode": "single_frame",  # or "frame_burst", or "clip"
            "frame_interval_sec": 5.0,
            "burst_offsets": [-2, -1, 0, 1, 2],
            "start_sec": 0.0,
            "end_sec": None,  # None: the end of the video
            "moments_sec": None,  # when set, replaces the interval grid
            "burst_spacing_sec": None,  # None: frame_interval_sec, as before
            "clip_length_sec": 2.0,  # spec 033: "clip" sends video
            "clip_fps": 4.0,
            "think": False,
            "request_timeout_sec": 240,
            "keep_alive": "2h",
            "max_retries": 3,
            "retry_backoff_sec": 10,
            "abort_after_consecutive_failures": 5,
            "options": {"temperature": 0, "top_p": 0.9},
            "answer_shape": None,  # a list of fields (spec 033 R8)
            "enforce_shape": False,  # send it as a schema (D5)
        }
        if config:
            default_config.update(config)
        # Resolved here so the output's provenance records the URL actually
        # used, and so an empty value from a config form means "server default".
        # Only Ollama has a default server; another kind must name one.
        if (default_config.get("backend") or "ollama") == "ollama":
            default_config["base_url"] = (
                default_config["base_url"] or default_ollama_base_url()
            )
        super().__init__(default_config)
        self.logger = logging.getLogger(__name__)
        self._client: VLMClient | None = None
        self._shape: list[shapes.Field] | None = None
        self._model_ref: ModelRef | None = None
        self._stop_check: Callable[[], bool] | None = None
        self._summary: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # BasePipeline interface

    @property
    def backend(self) -> str:
        return str(self.config.get("backend") or "ollama")

    def initialize(self) -> None:
        check_config(self.config)
        self.config["backend"] = self.backend
        shape = self.config.get("answer_shape")
        self._shape = shapes.validate_shape(shape) if shape is not None else None
        base_url = self.config["base_url"]
        _require_sendable(base_url)
        self._client = make_client(
            self.backend,
            base_url,
            int(self.config["request_timeout_sec"]),
            _server_key(base_url) if self.backend != "ollama" else None,
            ollama_class=OllamaVLMClient,
        )
        self._client.preflight(self.config["model"])
        if self._shape is not None and self.backend != "ollama":
            known = _server_capabilities(base_url)
            if known is not None and known.get("structured_output") is False:
                self._client.structured_output = False  # type: ignore[attr-defined]
        if self.config.get("sampling_mode") == "clip":
            self._check_takes_video()
        self.set_model_info(self.config["model"])
        details = self._client.model_details(self.config["model"])
        self._vlm_details = details
        if self.backend == "ollama":
            ref = ModelRef(
                self.config["model"],
                "ollama",
                details["digest"],
                "ollama-digest" if details["digest"] else "unknown",
                None if details["digest"] else "the server reported no digest",
            )
        else:
            note = "the server reported no digest"
            if details.get("root") and details["root"] != self.config["model"]:
                note += f"; it serves {details['root']}"
            ref = ModelRef(self.config["model"], self.backend, None, "unknown", note)
        self._model_ref = ref
        self._model_refs.append(ref)
        self.is_initialized = True
        self.logger.info(
            f"VLM Annotation Pipeline initialized: model={self.config['model']} "
            f"backend={self.backend} base_url={base_url}"
        )

    def provenance_vlm(self) -> dict[str, Any]:
        """The VLM-specific provenance fields (spec 017 FR-003, spec 033)."""
        details = getattr(self, "_vlm_details", {})
        endpoint = classify_endpoint(self.config["base_url"])
        record = {
            "prompt_sha256": hashlib.sha256(
                self.config["prompt"].encode("utf-8")
            ).hexdigest(),
            "model_digest": details.get("digest"),
            "quantization": details.get("quantization"),
            "base_url": self.config["base_url"],
            "endpoint_class": endpoint.cls,
            "endpoint_host": endpoint.host,
            "backend": self.backend,
        }
        if self._shape is not None:
            record["answer_shape_sha256"] = shapes.shape_sha256(self._shape)
        return record

    def _check_takes_video(self) -> None:
        """A clip run asks first, with a synthetic clip, whether the model
        takes video at all, and whether the server takes the clip's fps."""
        probe = getattr(self._client, "probe_video", None)
        takes_video = False
        if callable(probe):
            takes_video, _ = probe(self.config["model"])
        if not takes_video:
            raise ConfigProblem(
                "CLIP_NOT_SUPPORTED", "This model or server doesn't take video."
            )

    def _note_server_fingerprint(self, result: VLMCallResult) -> None:
        """An OpenAI-compatible server's fingerprint is the closest thing it
        gives to a revision (research R7); the first one seen is recorded."""
        ref = self._model_ref
        if ref is None or ref.revision or not result.system_fingerprint:
            return
        ref.revision = result.system_fingerprint
        ref.revision_kind = "server-fingerprint"
        ref.revision_note = None

    def set_stop_check(self, check: Callable[[], bool]) -> None:
        """Asked before each sample; when it says stop, the run keeps what it
        has and returns normally (spec 027 R7). Optional and duck-typed, like
        provenance_vlm: BasePipeline doesn't change."""
        self._stop_check = check

    def run_summary(self) -> dict[str, Any]:
        """How far the last `process()` got, for the provenance record."""
        return dict(self._summary)

    def cleanup(self) -> None:
        self._client = None
        self.is_initialized = False

    def process(
        self,
        video_path: str,
        start_time: float = 0.0,
        end_time: float | None = None,
        pps: float = 0.0,
        output_dir: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self.is_initialized:
            self.initialize()
        assert self._client is not None

        video_metadata = self._get_video_metadata(video_path)
        frame_interval_sec = float(self.config["frame_interval_sec"])
        sampling_mode = self.config["sampling_mode"]
        burst_spacing_sec = float(
            self.config.get("burst_spacing_sec") or frame_interval_sec
        )

        sample_points = self._planned_points(
            video_metadata["duration"], start_time, end_time
        )
        fps_video = float(video_metadata["fps"] or 0.0)

        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        annotations: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        # The shape reads replies into fields; it's sent as a schema only when
        # forced (D5): otherwise the question's own words ask for the shape.
        schema = (
            shapes.to_json_schema(self._shape)
            if self._shape and self.config.get("enforce_shape")
            else None
        )
        structured_output: str | None = None
        key_refused: str | None = None
        consecutive_failures = 0
        answered = 0
        done = 0
        stopped_early = False
        last_call_error: str | None = None
        last_frame_error: str | None = None
        abort_reason: str | None = None
        last_write = 0.0
        gate = gate_for(self.config["base_url"], self.config["model"])

        def progress(finished: bool) -> dict[str, Any]:
            extra: dict[str, Any] = {
                "vlm_progress": {
                    "planned": len(sample_points),
                    "done": done,
                    "finished": finished,
                    "stopped_early": stopped_early,
                },
                "vlm_skipped": skipped,
            }
            if self._shape is not None:
                extra["vlm_answer_shape"] = shapes.shape_dicts(self._shape)
            return extra

        try:
            for i, anchor_t in enumerate(sample_points):
                if self._stop_check is not None and self._stop_check():
                    stopped_early = True
                    self.logger.info(
                        f"Stopped after {done} of {len(sample_points)} samples"
                    )
                    break
                context_offsets: list[int] | None
                call_extra: dict[str, Any] = {"schema": schema} if schema else {}
                clip: dict[str, Any] = {}
                if sampling_mode == "clip":
                    frame_numbers, clip_jpegs, clip = self._read_clip(
                        cap, video_metadata, anchor_t
                    )
                    images: list[bytes] = []
                    context_offsets = None
                    if clip_jpegs:
                        call_extra.update(video=clip_jpegs, video_fps=clip["clip_fps"])
                elif sampling_mode == "frame_burst":
                    offsets = self.config["burst_offsets"]
                    frame_numbers, images, context_offsets = self._read_burst(
                        cap, video_metadata, anchor_t, burst_spacing_sec, offsets
                    )
                else:
                    frame_numbers, images, context_offsets = self._read_single(
                        cap, video_metadata, anchor_t
                    )

                if not images and not call_extra.get("video"):
                    self.logger.warning(
                        f"No readable frame at t={anchor_t:.2f}s in "
                        f"{video_metadata['video_id']}; skipping sample point"
                    )
                    consecutive_failures += 1
                    done += 1
                    last_frame_error = f"no readable frame at t={anchor_t:.2f}s"
                    skipped.append({"t": anchor_t, "reason": "no readable frame"})
                    if (
                        consecutive_failures
                        >= self.config["abort_after_consecutive_failures"]
                    ):
                        abort_reason = (
                            f"{consecutive_failures} consecutive unreadable frames"
                        )
                        self.logger.error(f"Aborting: {abort_reason}.")
                        break
                    continue

                try:
                    with gate.acquire("run"):
                        result = self._client.chat(
                            model=self.config["model"],
                            prompt=self.config["prompt"],
                            images=images,
                            think=self.config["think"],
                            keep_alive=self.config["keep_alive"],
                            options=self.config["options"],
                            max_retries=self.config["max_retries"],
                            retry_backoff_sec=self.config["retry_backoff_sec"],
                            **call_extra,
                        )
                except KeyRefusedError as e:
                    # No sample after a refused key would fare better (FR-006).
                    key_refused = str(e)
                    self.logger.error(f"Stopping: {key_refused}")
                    break
                done += 1

                fit: dict[str, Any] = {}
                if result.error:
                    consecutive_failures += 1
                    last_call_error = result.error
                    label = f"ERROR: {result.error}"
                else:
                    consecutive_failures = 0
                    answered += 1
                    self._note_server_fingerprint(result)
                    if self._shape is not None:
                        checked = shapes.check_reply(self._shape, result.raw_text)
                        label = checked.label
                        fit = {
                            "fields": checked.fields,
                            "fit": checked.fit,
                            "fit_problem": checked.fit_problem,
                        }
                        structured_output = (
                            result.structured_output or structured_output
                        )
                    else:
                        label = _parse_label(result.raw_text)

                center_frame = (
                    min(frame_numbers, key=lambda n: abs(n - anchor_t * fps_video))
                    if clip
                    else frame_numbers[len(frame_numbers) // 2]
                )
                annotations.append(
                    create_coco_annotation(
                        annotation_id=i + 1,
                        image_id=f"{video_metadata['video_id']}_frame_{center_frame:06d}",
                        category_id=1,
                        bbox=[0, 0, video_metadata["width"], video_metadata["height"]],
                        video_id=video_metadata["video_id"],
                        timestamp_sec=anchor_t,
                        frame_number=center_frame,
                        sampling_mode=sampling_mode,
                        context_frame_offsets=context_offsets,
                        context_frame_numbers=frame_numbers,
                        frame_times=[
                            round(n / video_metadata["fps"], 3)
                            if video_metadata["fps"]
                            else anchor_t
                            for n in frame_numbers
                        ],
                        status="error" if result.error else "answered",
                        label=label,
                        reasoning=result.thinking or result.raw_text,
                        raw_response=result.raw_text,
                        model=self.config["model"],
                        backend=self.backend,
                        base_url=self.config["base_url"],
                        prompt=self.config["prompt"],
                        total_time=round(result.total_time, 4),
                        load_time=_rounded(result.load_time, 4),
                        prompt_tokens=result.prompt_tokens,
                        resp_tokens=result.resp_tokens,
                        tokens_per_sec=_rounded(result.tokens_per_sec, 2),
                        **fit,
                        **clip,
                    )
                )
                # Answers as they come (R6): at most once a second.
                if output_dir and time.monotonic() - last_write >= 1.0:
                    self._save_coco_annotations(
                        annotations,
                        output_dir,
                        video_metadata,
                        extra=progress(False),
                        validate=False,
                    )
                    last_write = time.monotonic()

                if (
                    consecutive_failures
                    >= self.config["abort_after_consecutive_failures"]
                ):
                    abort_reason = f"{consecutive_failures} consecutive call failures"
                    self.logger.error(f"Aborting: {abort_reason}.")
                    break
        finally:
            cap.release()

        self._summary = {
            "samples_planned": len(sample_points),
            "samples_answered": answered,
            "stopped_early": stopped_early,
        }
        if structured_output:
            self._summary["structured_output"] = structured_output
            if self.backend != "ollama":
                _record_structured_output(
                    self.config["base_url"], structured_output == "enforced"
                )
        if output_dir and (annotations or skipped or stopped_early or key_refused):
            self._save_coco_annotations(
                annotations, output_dir, video_metadata, extra=progress(True)
            )
        if key_refused:
            raise RuntimeError(key_refused)

        # Kept in the output folder above for diagnosis, but a run that gave up
        # or got no answer at all is a failure, not a result. One the user
        # stopped is a result: what it has so far.
        last_error = last_call_error or last_frame_error
        if abort_reason:
            raise RuntimeError(
                f"VLM annotation aborted after {abort_reason}; last error: {last_error}"
            )
        if sample_points and answered == 0 and not stopped_early:
            raise RuntimeError(
                f"No sample point got an answer from the model; last error: "
                f"{last_error}"
            )

        self.logger.info(
            f"VLM annotation complete: {len(annotations)} sample points "
            f"({sampling_mode}) for {video_metadata['video_id']}"
        )
        return annotations

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "vlm_annotation",
            "description": "Per-frame vision-language-model classification/reasoning",
            "properties": {
                "label": {"type": "string"},
                "reasoning": {"type": "string"},
                "raw_response": {"type": "string"},
                "timestamp_sec": {"type": "number"},
                "sampling_mode": {"type": "string"},
                "model": {"type": "string"},
            },
        }

    # ------------------------------------------------------------------
    # Video / frame helpers

    def _get_video_metadata(self, video_path: str) -> dict[str, Any]:
        return frames.video_metadata(video_path)

    def _planned_points(
        self, duration: float, start_time: float = 0.0, end_time: float | None = None
    ) -> list[float]:
        """The config's points (the viewer draws the same ones), narrowed by
        `process()`'s own range when a caller passes one; the job runner passes
        the whole video."""
        points = sampling.sample_points(
            duration,
            float(self.config["frame_interval_sec"]),
            float(self.config.get("start_sec") or 0.0),
            self.config.get("end_sec"),
            self.config.get("moments_sec"),
        )
        last = duration if end_time is None else end_time
        return [t for t in points if start_time <= t <= last]

    def _read_single(
        self, cap, video_metadata: dict[str, Any], anchor_t: float
    ) -> tuple[list[int], list[bytes], None]:
        sent = frames.read_sample(
            cap, video_metadata["fps"], video_metadata["duration"], anchor_t, None, 0.0
        )
        return [f.frame_number for f in sent], [f.jpeg for f in sent], None

    def _read_burst(
        self,
        cap,
        video_metadata: dict[str, Any],
        anchor_t: float,
        spacing_sec: float,
        offsets: list[int],
    ) -> tuple[list[int], list[bytes], list[int]]:
        sent = frames.read_sample(
            cap,
            video_metadata["fps"],
            video_metadata["duration"],
            anchor_t,
            offsets,
            spacing_sec,
        )
        return (
            [f.frame_number for f in sent],
            [f.jpeg for f in sent],
            [f.offset for f in sent],
        )

    def _read_clip(
        self, cap, video_metadata: dict[str, Any], anchor_t: float
    ) -> tuple[list[int], list[bytes], dict[str, Any]]:
        length = float(self.config["clip_length_sec"])
        clip_fps = float(self.config["clip_fps"])
        sent = frames.read_clip(
            cap,
            video_metadata["fps"],
            video_metadata["duration"],
            anchor_t,
            length,
            clip_fps,
        )
        start, end = sampling.clip_span(anchor_t, length, video_metadata["duration"])
        return (
            [f.frame_number for f in sent],
            [f.jpeg for f in sent],
            {"clip_start_sec": start, "clip_end_sec": end, "clip_fps": clip_fps},
        )

    # ------------------------------------------------------------------
    # Output

    def _save_coco_annotations(
        self,
        annotations: list[dict[str, Any]],
        output_dir: str,
        video_metadata: dict[str, Any],
        extra: dict[str, Any] | None = None,
        validate: bool = True,
    ) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        images = []
        image_ids = set()
        for ann in annotations:
            image_id = ann["image_id"]
            if image_id not in image_ids:
                image_ids.add(image_id)
                images.append(
                    create_coco_image_entry(
                        image_id=image_id,
                        width=video_metadata["width"],
                        height=video_metadata["height"],
                        file_name=f"frame_{ann.get('frame_number', 0):06d}.jpg",
                        video_id=video_metadata["video_id"],
                        frame_number=ann.get("frame_number", 0),
                        timestamp=ann.get("timestamp_sec", 0.0),
                    )
                )

        categories = [
            {
                "id": 1,
                "name": "vlm_annotation",
                "supercategory": "frame_classification",
                "description": "Per-frame vision-language-model classification",
            }
        ]

        coco_path = output_path / f"{video_metadata['video_id']}_vlm_annotation.json"
        export_coco_json(
            annotations, images, str(coco_path), categories, extra, validate
        )
        if not validate:
            return

        validation_result = validate_coco_json(str(coco_path), "vlm_annotation")
        if validation_result.is_valid:
            self.logger.info(f"VLM annotation COCO validation successful: {coco_path}")
        else:
            self.logger.warning(
                "VLM annotation COCO validation warnings: "
                f"{', '.join(validation_result.warnings)}"
            )
