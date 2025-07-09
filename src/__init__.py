"""
VideoAnnotator Core Module

This package contains the core functionality for the VideoAnnotator toolkit.
"""

from .pipelines import (
    BasePipeline,
    SceneDetectionPipeline,
    PersonTrackingPipeline,
    FaceAnalysisPipeline,
    AudioPipeline
)

__version__ = "1.0.0"

__all__ = [
    "BasePipeline",
    "SceneDetectionPipeline",
    "PersonTrackingPipeline", 
    "FaceAnalysisPipeline",
    "AudioProcessingPipeline"
]
