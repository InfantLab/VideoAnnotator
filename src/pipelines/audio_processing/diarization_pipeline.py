"""
Speaker diarization pipeline using PyAnnote.

Handles speaker segmentation and identification with timestamps.
"""

import os
from typing import Any

from ..base_pipeline import BasePipeline

try:
    from pyannote.audio import Pipeline as PyAnnotePipeline

    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False


class DiarizationPipeline(BasePipeline):
    """
    Speaker diarization pipeline using PyAnnote.

    Outputs RTTM-compatible timestamped speaker turns.
    """

    @property
    def output_format(self) -> str:
        """Return the output format for this pipeline."""
        return "rttm"

    def __init__(self, config: dict[str, Any] | None = None):
        default_config = {
            "model": "pyannote/speaker-diarization-3.1",
            "min_speakers": 1,
            "max_speakers": 10,
            "use_auth_token": True,
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
        self.diarization_model = None

    @property
    def pipeline_name(self) -> str:
        return "speaker_diarization"

    def initialize(self) -> None:
        """Initialize PyAnnote diarization model."""
        if self.is_initialized:
            return

        if not PYANNOTE_AVAILABLE:
            raise ImportError(
                "PyAnnote not available. Install with: pip install pyannote.audio"
            )

        # Check for HuggingFace token
        hf_token = os.getenv("HF_AUTH_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token and self.config["use_auth_token"]:
            raise ValueError(
                "HuggingFace token required for PyAnnote models. "
                "Set HF_AUTH_TOKEN environment variable or disable with use_auth_token=False"
            )

        self.logger.info(f"Loading diarization model: {self.config['model']}")

        if hf_token:
            self.diarization_model = PyAnnotePipeline.from_pretrained(
                self.config["model"], use_auth_token=hf_token
            )
        else:
            self.diarization_model = PyAnnotePipeline.from_pretrained(
                self.config["model"]
            )

        self.is_initialized = True
        self.logger.info("DiarizationPipeline initialized")

    def process(
        self,
        video_path: str,
        start_time: float = 0.0,
        end_time: float | None = None,
        pps: float = 0.0,
        output_dir: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Process video for speaker diarization.

        Returns:
            List of speaker turns with timestamps
        """
        if not self.is_initialized:
            self.initialize()

        # For diarization, we process the full audio file
        # Extract audio from video if needed (simplified - in production would use ffmpeg)
        audio_path = video_path  # Assume video path can be used directly

        # Create metadata
        from pathlib import Path

        metadata = {"video_id": Path(video_path).stem, "filepath": video_path}

        # Apply diarization
        diarization = self.diarization_model(
            audio_path,
            min_speakers=self.config["min_speakers"],
            max_speakers=self.config["max_speakers"],
        )

        # Convert to RTTM format
        turns = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turn_data = {
                "file_id": metadata.get("video_id", "unknown"),
                "start_time": turn.start,
                "duration": turn.duration,
                "end_time": turn.end,  # For convenience
                "speaker_id": speaker,
                "confidence": 1.0,  # PyAnnote doesn't provide confidence by default
                "pipeline": self.pipeline_name,
                "format": self.output_format,
            }
            turns.append(turn_data)

        self.logger.info(f"Diarization complete: {len(turns)} speaker turns")
        return turns

    def get_schema(self) -> dict[str, Any]:
        """Return schema for diarization output."""
        return {
            "type": "array",
            "description": "Speaker diarization results in RTTM format",
            "items": {
                "type": "object",
                "properties": {
                    "file_id": {"type": "string", "description": "File identifier"},
                    "start_time": {
                        "type": "number",
                        "description": "Start time in seconds",
                    },
                    "duration": {
                        "type": "number",
                        "description": "Duration in seconds",
                    },
                    "end_time": {
                        "type": "number",
                        "description": "End time in seconds",
                    },
                    "speaker_id": {
                        "type": "string",
                        "description": "Speaker identifier",
                    },
                    "confidence": {"type": "number", "description": "Confidence score"},
                    "pipeline": {
                        "type": "string",
                        "description": "Source pipeline name",
                    },
                    "format": {"type": "string", "description": "Output format"},
                },
                "required": [
                    "file_id",
                    "start_time",
                    "duration",
                    "speaker_id",
                    "pipeline",
                    "format",
                ],
            },
        }

    def cleanup(self) -> None:
        """Cleanup diarization resources."""
        if self.diarization_model is not None:
            del self.diarization_model
            self.diarization_model = None
        self.is_initialized = False
        self.logger.info("DiarizationPipeline cleanup complete")
