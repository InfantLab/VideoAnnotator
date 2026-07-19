"""Audio processing pipelines for VideoAnnotator.

This module provides modular audio pipelines:
- AudioPipelineModular: Modular coordinator for multiple independent audio pipelines (recommended)
- SpeechPipeline: Speech recognition using OpenAI Whisper (inherits from BasePipeline)
- DiarizationPipeline: Speaker diarization using PyAnnote (inherits from BasePipeline)
- LAIONVoicePipeline: Voice emotion analysis using LAION Empathic Insight models (inherits from BasePipeline)
- WhisperBasePipeline: Base pipeline for Whisper-based audio processing tasks (inherits from BasePipeline)

For backwards compatibility, AudioPipelineModular is also available as AudioPipeline.
"""

from .audio_pipeline_modular import AudioPipelineModular
from .diarization_pipeline import DiarizationPipeline
from .speech_pipeline import SpeechPipeline
from .whisper_base_pipeline import WhisperBasePipeline

# For backwards compatibility
AudioPipeline = AudioPipelineModular

__all__ = [
    "AudioPipeline",  # Backwards compatibility alias
    "AudioPipelineModular",
    "DiarizationPipeline",
    "LAIONVoicePipeline",
    "SpeechPipeline",
    "WhisperBasePipeline",
]


# LAIONVoicePipeline needs the `audio-laion` extras group (transformers,
# huggingface-hub) — a strict superset of `audio`'s deps, not a subset —
# so it can't be imported eagerly alongside the rest of this package
# without breaking a plain `[audio]` install. See 004-extras-based-install.
def __getattr__(name: str) -> object:
    if name != "LAIONVoicePipeline":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .laion_voice_pipeline import LAIONVoicePipeline

    return LAIONVoicePipeline
