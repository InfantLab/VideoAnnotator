"""
VideoAnnotator Pipeline Modules

This package contains modular pipeline implementations for video annotation tasks.
"""

from .base_pipeline import BasePipeline
from .scene_detection.scene_pipeline_legacy import SceneDetectionPipeline
from .person_tracking.person_pipeline import PersonTrackingPipeline
from .face_analysis.face_pipeline import FaceAnalysisPipeline
from .audio_processing.audio_pipeline import AudioPipeline

__all__ = [
    "BasePipeline",
    "SceneDetectionPipeline", 
    "PersonTrackingPipeline",
    "FaceAnalysisPipeline",
    "AudioPipeline"
]
