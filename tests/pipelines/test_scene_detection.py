"""Modern Scene Detection Pipeline Tests.

Tests current scene detection and video analysis capabilities. Living
documentation for scene detection functionality.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from videoannotator.pipelines.scene_detection.scene_pipeline import (
    SceneDetectionPipeline,
)


@pytest.mark.unit
class TestSceneDetectionPipeline:
    """Core functionality tests for scene detection pipeline."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization with custom configuration."""
        config = {"threshold": 25.0, "min_scene_length": 3.0, "clip_model": "ViT-L/14"}
        pipeline = SceneDetectionPipeline(config)

        assert pipeline.config["threshold"] == 25.0
        assert pipeline.config["min_scene_length"] == 3.0
        assert pipeline.config["clip_model"] == "ViT-L/14"

    def test_default_configuration(self):
        """Test pipeline with default configuration values."""
        pipeline = SceneDetectionPipeline()

        # Verify current default configuration
        assert pipeline.config["threshold"] == 30.0
        assert pipeline.config["min_scene_length"] == 2.0
        assert pipeline.config["clip_model"] == "ViT-B-32"
        assert pipeline.config["use_gpu"]

    def test_initialization_lifecycle(self):
        """Test pipeline initialization and cleanup lifecycle."""
        pipeline = SceneDetectionPipeline()

        # Should initialize successfully
        pipeline.initialize()
        assert pipeline.is_initialized

        # Should cleanup properly
        pipeline.cleanup()
        assert not pipeline.is_initialized

    def test_schema_format(self):
        """Test that pipeline returns correct schema format."""
        pipeline = SceneDetectionPipeline()
        schema = pipeline.get_schema()

        # Should return schema structure for scene detection
        assert "type" in schema
        assert isinstance(schema, dict)


@pytest.mark.performance
class TestSceneDetectionPerformance:
    """Performance and efficiency tests."""

    def test_processing_efficiency(self):
        """Test that pipeline processes efficiently without memory leaks."""
        pipeline = SceneDetectionPipeline()

        # Mock processing to test pipeline overhead
        with patch("cv2.VideoCapture") as mock_cap:
            mock_instance = Mock()
            mock_instance.read.return_value = (
                True,
                np.zeros((480, 640, 3), dtype=np.uint8),
            )
            mock_instance.get.return_value = 30.0  # FPS
            mock_cap.return_value = mock_instance

            pipeline.initialize()

            # Should process without errors
            with patch.object(pipeline, "_detect_scene_boundaries") as mock_detect:
                mock_detect.return_value = []  # Mock empty results
                result = pipeline.process("test_video.mp4")
                assert isinstance(result, list)

            pipeline.cleanup()
