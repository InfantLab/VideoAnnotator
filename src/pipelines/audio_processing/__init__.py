"""
Audio processing pipelines for VideoAnnotator.

This module provides separate pipelines for different audio processing tasks:
- DiarizationPipeline: Speaker diarization using PyAnnote
- SpeechPipeline: Speech recognition using OpenAI Whisper
- AudioPipeline: Comprehensive audio analysis including transcription, diarization, and feature extraction
"""

from .audio_pipeline import AudioPipeline, AudioPipelineConfig

__all__ = [
    "DiarizationPipeline",
    "DiarizationPipelineConfig",
    "AudioPipeline",
    "AudioPipelineConfig",
], AudioPipelineConfig

__all__ = ["AudioPipeline", "AudioPipelineConfig"]
