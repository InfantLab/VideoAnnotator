"""Base pipeline interface for all video annotation processors."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

# Note: After standards migration, base schemas are no longer used
# Pipelines now return native format dictionaries (COCO, WebVTT, RTTM, etc.)
from version import create_annotation_metadata, get_model_info


class BasePipeline(ABC):
    """Base class for all annotation pipelines."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize pipeline with configuration."""
        self.config = config or {}
        self.name = self.__class__.__name__.lower()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_initialized = False
        self._model_info = None  # Will be set by individual pipelines

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the pipeline (load models, etc.)."""

    @abstractmethod
    def process(
        self,
        video_path: str,
        start_time: float = 0.0,
        end_time: float | None = None,
        pps: float = 0.0,  # predictions per second, 0 = once per segment
        output_dir: str | None = None,
    ) -> list[dict[str, Any]]:
        """Process video segment and return annotations.

        Args:
            video_path: Path to video file
            start_time: Segment start time in seconds
            end_time: Segment end time in seconds (None = full video)
            pps: Predictions per second (0 = once per segment)
            output_dir: Optional output directory for saving results

        Returns:
            List of annotation dictionaries in native formats (COCO, WebVTT, RTTM, etc.)
        """

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources."""

    @abstractmethod
    def get_schema(self) -> dict[str, Any]:
        """Return the JSON schema for this pipeline's output."""

    def validate_video_path(self, video_path: str) -> Path:
        """Validate and return Path object for video."""
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {video_path}")
        return path

    def get_video_info(self, video_path: str) -> dict[str, Any]:
        """Get basic video information."""
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0

            return {
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "duration": duration,
            }
        finally:
            cap.release()

    def set_model_info(self, model_name: str, model_path: str | None = None) -> None:
        """Set model information for this pipeline."""
        self._model_info = get_model_info(model_name, model_path)
        self.logger.info(f"Model info set: {model_name}")

    def create_output_metadata(
        self, video_metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Create comprehensive metadata for pipeline outputs."""
        return create_annotation_metadata(
            pipeline_name=self.__class__.__name__,
            model_info=self._model_info,
            processing_params=self.config,
            video_metadata=video_metadata,
        )

    def __enter__(self):
        """Context manager entry."""
        if not self.is_initialized:
            self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
