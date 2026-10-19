"""Speech pipeline device choice: CUDA, CPU, and the fall back without a driver."""

from unittest.mock import patch

import pytest

# The tests patch torch.cuda, so torch must be importable.
try:
    import torch  # noqa: F401

    from videoannotator.pipelines.audio_processing.speech_pipeline import SpeechPipeline

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.unit
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestWhisperBasePipelineAdvanced:
    """Advanced WhisperBasePipeline tests with PyTorch dependencies."""

    # The model load is patched out: these tests are about device choice, and a
    # real load on a machine without CUDA falls back to CPU (tested below).
    @patch.object(SpeechPipeline, "_load_whisper_model")
    @patch("torch.cuda.is_available")
    def test_device_detection_cuda_available(self, mock_cuda, _mock_load):
        """Test device detection when CUDA is available."""
        mock_cuda.return_value = True

        pipeline = SpeechPipeline({"device": "auto"})
        pipeline.initialize()

        assert pipeline.device.type == "cuda"

    @patch("torch.cuda.is_available", return_value=True)
    def test_auto_falls_back_to_cpu_without_nvidia_driver(self, _mock_cuda):
        """CUDA-built torch on a machine with no NVIDIA driver falls back to CPU."""
        no_driver = RuntimeError(
            "Found no NVIDIA driver on your system. Please check that you have an "
            "NVIDIA GPU and installed a driver"
        )
        with patch.object(
            SpeechPipeline, "_load_whisper_model", side_effect=[no_driver, None]
        ):
            pipeline = SpeechPipeline({"device": "auto"})
            pipeline.initialize()

        assert pipeline.device.type == "cpu"
        assert pipeline.fell_back_to_cpu

    @patch.object(SpeechPipeline, "_load_whisper_model")
    @patch("torch.cuda.is_available")
    def test_device_detection_cuda_unavailable(self, mock_cuda, _mock_load):
        """Test device detection when CUDA is unavailable."""
        mock_cuda.return_value = False

        pipeline = SpeechPipeline({"device": "auto"})
        pipeline.initialize()

        assert pipeline.device.type == "cpu"
