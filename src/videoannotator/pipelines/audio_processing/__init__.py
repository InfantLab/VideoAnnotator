"""Audio processing pipelines for VideoAnnotator.

This module provides modular audio pipelines:
- AudioPipelineModular: Modular coordinator for multiple independent audio pipelines (recommended)
- SpeechPipeline: Speech recognition using Whisper on faster-whisper (inherits from BasePipeline)
- DiarizationPipeline: Speaker diarization using PyAnnote (inherits from BasePipeline)
- WhisperBasePipeline: Base pipeline for Whisper-based audio processing tasks (inherits from BasePipeline)

For backwards compatibility, AudioPipelineModular is also available as AudioPipeline.
"""

from .audio_pipeline_modular import AudioPipelineModular
from .diarization_pipeline import DiarizationPipeline
from .speech_pipeline import SpeechPipeline
from .whisper_base_pipeline import WhisperBasePipeline

try:
    import onnxruntime

    # faster-whisper installs onnxruntime and torchmetrics (via pyannote) imports
    # it. Its official Windows builds send telemetry by default (constitution I).
    onnxruntime.disable_telemetry_events()
except Exception:  # not installed, or a build without the call
    pass

# For backwards compatibility
AudioPipeline = AudioPipelineModular

__all__ = [
    "AudioPipeline",  # Backwards compatibility alias
    "AudioPipelineModular",
    "DiarizationPipeline",
    "SpeechPipeline",
    "WhisperBasePipeline",
]
