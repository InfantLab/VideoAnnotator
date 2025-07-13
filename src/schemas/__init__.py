"""
VideoAnnotator Industry Standards Integration

This module provides schemas aligned with industry standards:
- COCO JSON format (PyTorch, YOLO, detectron2)
- CVAT/Datumaro format (annotation tools)  
- Label Studio format (research/enterprise)
- WebVTT format (video captions)
- RTTM format (speaker diarization)

Usage:
    # For ML training (COCO format)
    from src.schemas import COCOPersonDetection, COCOPoseKeypoints
    
    # For annotation tools (CVAT format)  
    from src.schemas import CVATPersonDetection
    
    # For quick prototyping (flexible JSON)
    from src.schemas import PersonDetection, PoseKeypoints
    
    # For audio processing
    from src.schemas import WebVTTTranscription, RTTMSpeakerDiarization
"""

# Import from the new industry standards module
from .industry_standards import (
    # COCO formats (recommended for ML)
    COCOPersonDetection,
    COCOPoseKeypoints,
    COCOFaceDetection, 
    COCOSceneDetection,
    
    # Annotation tool formats
    CVATPersonDetection,
    LabelStudioPersonDetection,
    
    # Audio formats
    WebVTTTranscription,
    RTTMSpeakerDiarization,
    AudioSetClassification,
    
    # Simple formats (prototyping & backwards compatibility)
    PersonDetection,
    PoseKeypoints,
    FaceDetection,
    FaceEmotion, 
    SpeechRecognition,
    SpeakerDiarization,
    SceneAnnotation,
    
    # Export functions
    export_to_coco_json,
    export_to_cvat_xml,
    export_to_labelstudio_json,
    export_audio_transcription_webvtt,
    export_speaker_diarization_rttm,
    
    # Conversion utilities
    to_cvat_format,
    to_labelstudio_format,
    from_dict,
    
    # Migration helpers
    get_recommended_schema,
    migrate_legacy_annotation
)

# Legacy compatibility (DEPRECATED - will be removed in v2.0)
import warnings

try:
    from .base_schema import AnnotationBase as LegacyAnnotationBase, VideoMetadata
    from .scene_schema import SceneSegment
    from .person_schema import PersonTrajectory  
    from .face_schema import FaceGaze
    from .audio_schema import SpeechSegment, AudioFeatures
    
    # Alias legacy names to new standards
    AnnotationBase = LegacyAnnotationBase
    
    def _deprecated_warning(old_name, new_name):
        warnings.warn(
            f"{old_name} is deprecated. Use {new_name} instead. "
            f"Legacy schemas will be removed in v2.0.", 
            DeprecationWarning, 
            stacklevel=3
        )
    
    # Provide backward compatibility with warnings
    class PersonDetectionLegacy(PersonDetection):
        def __init__(self, *args, **kwargs):
            _deprecated_warning("PersonDetectionLegacy", "PersonDetection")
            super().__init__(*args, **kwargs)
    
    class PoseKeypointsLegacy(PoseKeypoints):
        def __init__(self, *args, **kwargs):
            _deprecated_warning("PoseKeypointsLegacy", "PoseKeypoints")
            super().__init__(*args, **kwargs)
            
except ImportError:
    # Legacy files not available - this is expected after full migration
    VideoMetadata = None
    SceneSegment = None
    PersonTrajectory = None
    FaceGaze = None
    SpeechSegment = None
    AudioFeatures = None

__all__ = [
    # Modern industry standards (recommended)
    "COCOPersonDetection",
    "COCOPoseKeypoints", 
    "COCOFaceDetection",
    "COCOSceneDetection",
    "CVATPersonDetection",
    "LabelStudioPersonDetection", 
    "WebVTTTranscription",
    "RTTMSpeakerDiarization",
    "AudioSetClassification",
    
    # Simple schemas (prototyping)
    "PersonDetection",
    "PoseKeypoints", 
    "FaceDetection",
    "FaceEmotion",
    "SpeechRecognition",
    "SpeakerDiarization", 
    "SceneAnnotation",
    
    # Export functions
    "export_to_coco_json",
    "export_to_cvat_xml", 
    "export_to_labelstudio_json",
    "export_audio_transcription_webvtt",
    "export_speaker_diarization_rttm",
    "to_cvat_format",
    "to_labelstudio_format",
    "from_dict",
    
    # Migration utilities
    "get_recommended_schema",
    "migrate_legacy_annotation",
    
    # Legacy compatibility (DEPRECATED)
    "AnnotationBase",
    "VideoMetadata",
    "SceneSegment", 
    "PersonTrajectory",
    "FaceGaze",
    "SpeechSegment",
    "AudioFeatures"
]
