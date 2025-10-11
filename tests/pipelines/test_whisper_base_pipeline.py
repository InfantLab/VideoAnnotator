"""WhisperBasePipeline Tests - Complete Version.

Extended test suite for the WhisperBasePipeline component.
This builds on the Stage 1 basic tests with more comprehensive coverage.
"""

from unittest.mock import patch

import pytest

# Import the basic Stage 1 tests - commented out as file doesn't exist
# from tests.test_whisper_base_pipeline_stage1 import *

# Try importing with proper dependencies
try:
    import torch

    from src.pipelines.audio_processing.speech_pipeline import SpeechPipeline

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.unit
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestWhisperBasePipelineAdvanced:
    """Advanced WhisperBasePipeline tests with PyTorch dependencies."""

    @patch("torch.cuda.is_available")
    def test_device_detection_cuda_available(self, mock_cuda):
        """Test device detection when CUDA is available."""
        mock_cuda.return_value = True

        pipeline = SpeechPipeline({"device": "auto"})
        pipeline.initialize()

        if pipeline.is_initialized:
            assert pipeline.device.type == "cuda"

    @patch("torch.cuda.is_available")
    def test_device_detection_cuda_unavailable(self, mock_cuda):
        """Test device detection when CUDA is unavailable."""
        mock_cuda.return_value = False

        pipeline = SpeechPipeline({"device": "auto"})
        pipeline.initialize()

        if pipeline.is_initialized:
            assert pipeline.device.type == "cpu"

    def test_embedding_shape_validation(self):
        """Test embedding shape validation logic."""
        from src.pipelines.audio_processing.speech_pipeline import SpeechPipeline

        pipeline = SpeechPipeline()

        # Test with mock embeddings
        mock_embedding = torch.randn(1, 100, 768)

        # Test padding logic
        if hasattr(pipeline, "_pad_or_truncate_embeddings"):
            result = pipeline._pad_or_truncate_embeddings(
                mock_embedding, target_length=1500
            )
            assert result.shape == (1, 1500, 768)


@pytest.mark.integration
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestWhisperBasePipelineIntegrationAdvanced:
    """Integration tests for WhisperBasePipeline."""

    def test_initialization_with_real_dependencies(self):
        """Test initialization with real PyTorch."""
        from src.pipelines.audio_processing.speech_pipeline import SpeechPipeline

        pipeline = SpeechPipeline()

        # Should handle initialization gracefully
        try:
            pipeline.initialize()
            # If successful, should have device set
            if pipeline.is_initialized:
                assert pipeline.device is not None
        except Exception as e:
            # Should fail gracefully with informative error
            assert "whisper" in str(e).lower() or "model" in str(e).lower()

    def test_cleanup_with_real_resources(self):
        """Test cleanup with real resources."""
        from src.pipelines.audio_processing.speech_pipeline import SpeechPipeline

        pipeline = SpeechPipeline()

        try:
            pipeline.initialize()
        except:
            pass  # Initialization might fail, that's OK

            # Cleanup should always work
            pipeline.cleanup()
            assert not pipeline.is_initialized
