"""
Video Annotation Data Schemas

This module defines the standardized data schemas for all annotation pipelines.
All annotations follow a common structure for interoperability.
"""

from .base_schema import AnnotationBase, VideoMetadata
from .scene_schema import SceneAnnotation, SceneSegment
from .person_schema import PersonDetection, PersonTrajectory, PoseKeypoints
from .face_schema import FaceDetection, FaceEmotion, FaceGaze
from .audio_schema import SpeechSegment, SpeakerDiarization, AudioFeatures

__all__ = [
    "AnnotationBase",
    "VideoMetadata", 
    "SceneAnnotation",
    "SceneSegment",
    "PersonDetection",
    "PersonTrajectory", 
    "PoseKeypoints",
    "FaceDetection",
    "FaceEmotion",
    "FaceGaze",
    "SpeechSegment",
    "SpeakerDiarization",
    "AudioFeatures"
]
