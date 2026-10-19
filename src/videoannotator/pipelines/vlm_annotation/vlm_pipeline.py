"""Local vision-language-model frame annotation pipeline.

Samples frames from a video at a fixed cadence and classifies/describes each
one with a locally-hosted VLM through Ollama, using a user-supplied prompt
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
from pathlib import Path
from typing import Any

from videoannotator.config_env import default_ollama_base_url
from videoannotator.exporters.native_formats import (
    create_coco_annotation,
    create_coco_image_entry,
    export_coco_json,
    validate_coco_json,
)
from videoannotator.pipelines.base_pipeline import BasePipeline
from videoannotator.provenance import ModelRef

from . import frames, sampling
from .ollama_client import OllamaVLMClient

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


class VLMAnnotationPipeline(BasePipeline):
    """Per-frame vision-language-model classification via a local Ollama server."""

    def __init__(self, config: dict[str, Any] | None = None):
        default_config: dict[str, Any] = {
            "prompt": DEFAULT_PROMPT,
            "base_url": "",  # empty: the server's default_ollama_base_url()
            "model": "qwen3.5:9b",
            "sampling_mode": "single_frame",  # or "frame_burst"
            "frame_interval_sec": 5.0,
            "burst_offsets": [-2, -1, 0, 1, 2],
            "start_sec": 0.0,
            "end_sec": None,  # None: the end of the video
            "moments_sec": None,  # when set, replaces the interval grid
            "burst_spacing_sec": None,  # None: frame_interval_sec, as before
            "think": False,
            "request_timeout_sec": 240,
            "keep_alive": "2h",
            "max_retries": 3,
            "retry_backoff_sec": 10,
            "abort_after_consecutive_failures": 5,
            "options": {"temperature": 0, "top_p": 0.9},
        }
        if config:
            default_config.update(config)
        # Resolved here so the output's provenance records the URL actually
        # used, and so an empty value from a config form means "server default".
        default_config["base_url"] = (
            default_config["base_url"] or default_ollama_base_url()
        )
        super().__init__(default_config)
        self.logger = logging.getLogger(__name__)
        self._client: OllamaVLMClient | None = None

    # ------------------------------------------------------------------
    # BasePipeline interface

    def initialize(self) -> None:
        self._client = OllamaVLMClient(
            base_url=self.config["base_url"],
            timeout=int(self.config["request_timeout_sec"]),
        )
        self._client.preflight(self.config["model"])
        self.set_model_info(self.config["model"])
        details = self._client.model_details(self.config["model"])
        self._vlm_details = details
        self._model_refs.append(
            ModelRef(
                self.config["model"],
                "ollama",
                details["digest"],
                "ollama-digest" if details["digest"] else "unknown",
                None if details["digest"] else "the server reported no digest",
            )
        )
        self.is_initialized = True
        self.logger.info(
            f"VLM Annotation Pipeline initialized: model={self.config['model']} "
            f"backend=ollama base_url={self.config['base_url']}"
        )

    def provenance_vlm(self) -> dict[str, Any]:
        """The VLM-specific provenance fields (spec 017 FR-003)."""
        details = getattr(self, "_vlm_details", {})
        return {
            "prompt_sha256": hashlib.sha256(
                self.config["prompt"].encode("utf-8")
            ).hexdigest(),
            "model_digest": details.get("digest"),
            "quantization": details.get("quantization"),
            "base_url": self.config["base_url"],
        }

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

        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        annotations: list[dict[str, Any]] = []
        consecutive_failures = 0
        answered = 0
        last_call_error: str | None = None
        last_frame_error: str | None = None
        abort_reason: str | None = None
        try:
            for i, anchor_t in enumerate(sample_points):
                context_offsets: list[int] | None
                if sampling_mode == "frame_burst":
                    offsets = self.config["burst_offsets"]
                    frame_numbers, images, context_offsets = self._read_burst(
                        cap, video_metadata, anchor_t, burst_spacing_sec, offsets
                    )
                else:
                    frame_numbers, images, context_offsets = self._read_single(
                        cap, video_metadata, anchor_t
                    )

                if not images:
                    self.logger.warning(
                        f"No readable frame at t={anchor_t:.2f}s in "
                        f"{video_metadata['video_id']}; skipping sample point"
                    )
                    consecutive_failures += 1
                    last_frame_error = f"no readable frame at t={anchor_t:.2f}s"
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

                result = self._client.chat(
                    model=self.config["model"],
                    prompt=self.config["prompt"],
                    images=images,
                    think=self.config["think"],
                    keep_alive=self.config["keep_alive"],
                    options=self.config["options"],
                    max_retries=self.config["max_retries"],
                    retry_backoff_sec=self.config["retry_backoff_sec"],
                )

                if result.error:
                    consecutive_failures += 1
                    last_call_error = result.error
                    label = f"ERROR: {result.error}"
                else:
                    consecutive_failures = 0
                    answered += 1
                    label = _parse_label(result.raw_text)

                center_frame = frame_numbers[len(frame_numbers) // 2]
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
                        label=label,
                        reasoning=result.thinking or result.raw_text,
                        raw_response=result.raw_text,
                        model=self.config["model"],
                        backend="ollama",
                        base_url=self.config["base_url"],
                        prompt=self.config["prompt"],
                        total_time=round(result.total_time, 4),
                        load_time=round(result.load_time, 4),
                        prompt_tokens=result.prompt_tokens,
                        resp_tokens=result.resp_tokens,
                        tokens_per_sec=round(result.tokens_per_sec, 2),
                    )
                )

                if (
                    consecutive_failures
                    >= self.config["abort_after_consecutive_failures"]
                ):
                    abort_reason = f"{consecutive_failures} consecutive call failures"
                    self.logger.error(f"Aborting: {abort_reason}.")
                    break
        finally:
            cap.release()

        if output_dir and annotations:
            self._save_coco_annotations(annotations, output_dir, video_metadata)

        # Kept in the output folder above for diagnosis, but a run that stopped
        # early or got no answer at all is a failure, not a result.
        last_error = last_call_error or last_frame_error
        if abort_reason:
            raise RuntimeError(
                f"VLM annotation aborted after {abort_reason}; last error: {last_error}"
            )
        if sample_points and answered == 0:
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

    # ------------------------------------------------------------------
    # Output

    def _save_coco_annotations(
        self,
        annotations: list[dict[str, Any]],
        output_dir: str,
        video_metadata: dict[str, Any],
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
        export_coco_json(annotations, images, str(coco_path), categories)

        validation_result = validate_coco_json(str(coco_path), "vlm_annotation")
        if validation_result.is_valid:
            self.logger.info(f"VLM annotation COCO validation successful: {coco_path}")
        else:
            self.logger.warning(
                "VLM annotation COCO validation warnings: "
                f"{', '.join(validation_result.warnings)}"
            )
