"""
Audio Processing Pipeline Module

This module provides comprehensive audio analysis capabilities including:
- Speech transcription and diarization
- Audio event detection and classification
- Music detection and analysis
- Audio quality assessment
- Speech emotion analysis
"""

from .audio_pipeline import AudioPipeline, AudioPipelineConfig

__all__ = ['AudioPipeline', 'AudioPipelineConfig']
