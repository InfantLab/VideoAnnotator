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

import logging
import re
from pathlib import Path
from typing import Any

from videoannotator.exporters.native_formats import (
    create_coco_annotation,
    create_coco_image_entry,
    export_coco_json,
    validate_coco_json,
)
from videoannotator.pipelines.base_pipeline import BasePipeline

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
            "base_url": "http://127.0.0.1:11434",
            "model": "qwen3.5:9b",
            "sampling_mode": "single_frame",  # or "frame_burst"
            "frame_interval_sec": 5.0,
            "burst_offsets": [-2, -1, 0, 1, 2],
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
        self.is_initialized = True
        self.logger.info(
            f"VLM Annotation Pipeline initialized: model={self.config['model']} "
            f"backend=ollama base_url={self.config['base_url']}"
        )

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
        end_time = end_time if end_time is not None else video_metadata["duration"]
        frame_interval_sec = float(self.config["frame_interval_sec"])
        sampling_mode = self.config["sampling_mode"]

        sample_points = self._sample_points(start_time, end_time, frame_interval_sec)

        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        annotations: list[dict[str, Any]] = []
        consecutive_failures = 0
        try:
            for i, anchor_t in enumerate(sample_points):
                if sampling_mode == "frame_burst":
                    offsets = self.config["burst_offsets"]
                    frame_numbers, images, context_offsets = self._read_burst(
                        cap, video_metadata, anchor_t, frame_interval_sec, offsets
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
                    if (
                        consecutive_failures
                        >= self.config["abort_after_consecutive_failures"]
                    ):
                        self.logger.error(
                            f"Aborting: {consecutive_failures} consecutive "
                            "unreadable frames."
                        )
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
                    label = f"ERROR: {result.error}"
                else:
                    consecutive_failures = 0
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
                    self.logger.error(
                        f"Aborting: {consecutive_failures} consecutive call failures."
                    )
                    break
        finally:
            cap.release()

        if output_dir and annotations:
            self._save_coco_annotations(annotations, output_dir, video_metadata)

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
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        return {
            "video_id": Path(video_path).stem,
            "filepath": video_path,
            "width": width,
            "height": height,
            "fps": fps,
            "duration": duration,
            "total_frames": total_frames,
        }

    @staticmethod
    def _sample_points(
        start_time: float, end_time: float, frame_interval_sec: float
    ) -> list[float]:
        points = []
        t = start_time
        while t <= end_time:
            points.append(round(t, 3))
            t += frame_interval_sec
        return points

    @staticmethod
    def _seek_frame_jpeg(
        cap, fps: float, timestamp_sec: float
    ) -> tuple[int, bytes] | None:
        import cv2

        frame_number = max(0, round(timestamp_sec * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            return None
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            return None
        return frame_number, buf.tobytes()

    def _read_single(
        self, cap, video_metadata: dict[str, Any], anchor_t: float
    ) -> tuple[list[int], list[bytes], None]:
        result = self._seek_frame_jpeg(cap, video_metadata["fps"], anchor_t)
        if result is None:
            return [], [], None
        frame_number, jpeg_bytes = result
        return [frame_number], [jpeg_bytes], None

    def _read_burst(
        self,
        cap,
        video_metadata: dict[str, Any],
        anchor_t: float,
        frame_interval_sec: float,
        offsets: list[int],
    ) -> tuple[list[int], list[bytes], list[int]]:
        duration = video_metadata["duration"]
        frame_numbers: list[int] = []
        images: list[bytes] = []
        used_offsets: list[int] = []
        for off in offsets:
            t = anchor_t + off * frame_interval_sec
            if t < 0 or t > duration:
                continue
            result = self._seek_frame_jpeg(cap, video_metadata["fps"], t)
            if result is None:
                continue
            frame_number, jpeg_bytes = result
            frame_numbers.append(frame_number)
            images.append(jpeg_bytes)
            used_offsets.append(off)
        return frame_numbers, images, used_offsets

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
