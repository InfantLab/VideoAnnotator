"""
Audio processing pipelines for VideoAnnotator.

This module provides modular audio pipelines:
- AudioPipelineModular: Modular coordinator for multiple independent audio pipelines (recommended)
- SpeechPipeline: Speech recognition using OpenAI Whisper (inherits from BasePipeline)  
- DiarizationPipeline: Speaker diarization using PyAnnote (inherits from BasePipeline)

For backwards compatibility, AudioPipelineModular is also available as AudioPipeline.
"""

from .audio_pipeline_modular import AudioPipelineModular
from .speech_pipeline import SpeechPipeline
from .diarization_pipeline import DiarizationPipeline

# For backwards compatibility
AudioPipeline = AudioPipelineModular

__all__ = [
    "AudioPipeline",  # Backwards compatibility alias
    "AudioPipelineModular", 
    "SpeechPipeline",
    "DiarizationPipeline",
]
