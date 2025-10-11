"""VideoAnnotator Pipeline Modules.

This package contains modular pipeline implementations for video
annotation tasks.
"""

from .audio_processing import AudioPipeline
from .base_pipeline import BasePipeline
from .face_analysis.face_pipeline import FaceAnalysisPipeline
from .face_analysis.laion_face_pipeline import LAIONFacePipeline
from .person_tracking.person_pipeline import PersonTrackingPipeline
from .scene_detection.scene_pipeline import SceneDetectionPipeline

__all__ = [
    "AudioPipeline",
    "BasePipeline",
    "FaceAnalysisPipeline",
    "LAIONFacePipeline",
    "PersonTrackingPipeline",
    "SceneDetectionPipeline",
]
