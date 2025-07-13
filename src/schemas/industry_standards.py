"""
VideoAnnotator Industry Standards Integration

This is the new unified entry point for all VideoAnnotator schemas.
All schemas follow industry standards (COCO, CVAT, Label Studio, WebVTT, RTTM).

Usage:
    from src.schemas.industry_standards import (
        COCOPersonDetection,
        COCOPoseKeypoints, 
        COCOFaceDetection,
        CVATAnnotation,
        WebVTTTranscription,
        RTTMSpeakerDiarization
    )
"""

# Core industry standard base classes
from .standards_compatible_schemas import (
    COCOBaseAnnotation,
    COCOBoundingBox, 
    COCOKeypoints,
    COCOImageAnnotation,
    CVATAnnotation,
    CVATTrack,
    LabelStudioResult,
    UniversalPersonDetection,
    UniversalPoseKeypoints,
    export_to_coco_json,
    export_to_cvat_xml,
    export_to_labelstudio_json
)

# Audio format integrations
from .audio_standards import (
    WebVTTEntry,
    RTTMEntry,
    AudioSetEntry,
    AudioStandardsExporter,
    export_audio_transcription_webvtt,
    export_speaker_diarization_rttm,
    export_audio_events_audioset
)

# COCO extensions for video-specific needs
from .coco_extensions import (
    VIDEOANNOTATOR_CATEGORIES,
    COCOVideoImage,
    COCOVideoAnnotation,
    COCOPersonAnnotation,
    COCOFaceAnnotation, 
    COCOSceneAnnotation,
    COCOVideoDataset,
    create_coco_info,
    create_default_licenses,
    get_categories_for_pipeline,
    convert_yolo_bbox_to_coco,
    convert_yolo_keypoints_to_coco,
    calculate_bbox_area,
    create_video_coco_dataset,
    create_scene_annotation,
    create_person_annotation,
    create_face_annotation
)

# Simplified schemas for quick prototyping
from .simple_schemas import (
    BaseAnnotation as SimpleBaseAnnotation,
    PersonDetection as SimplePersonDetection,
    PoseKeypoints as SimplePoseKeypoints, 
    FaceDetection as SimpleFaceDetection,
    FaceEmotion as SimpleFaceEmotion,
    SpeechRecognition as SimpleSpeechRecognition,
    SpeakerDiarization as SimpleSpeakerDiarization,
    AudioEvent as SimpleAudioEvent,
    SceneAnnotation as SimpleSceneAnnotation,
    ObjectDetection as SimpleObjectDetection,
    to_cvat_format,
    to_labelstudio_format,
    from_dict
)


# ============================================================================
# Recommended Usage Patterns
# ============================================================================

# For COCO-compatible outputs (PyTorch, YOLO, etc.)
COCOPersonDetection = COCOPersonAnnotation
COCOPoseKeypoints = COCOKeypoints
COCOFaceDetection = COCOFaceAnnotation
COCOSceneDetection = COCOSceneAnnotation

# For annotation tools (CVAT, Label Studio)
CVATPersonDetection = CVATAnnotation
LabelStudioPersonDetection = LabelStudioResult

# For audio processing (WebVTT captions, RTTM diarization)
WebVTTTranscription = WebVTTEntry
RTTMSpeakerDiarization = RTTMEntry
AudioSetClassification = AudioSetEntry

# For quick prototyping (flexible JSON)
PersonDetection = SimplePersonDetection
PoseKeypoints = SimplePoseKeypoints
FaceDetection = SimpleFaceDetection
FaceEmotion = SimpleFaceEmotion
SpeechRecognition = SimpleSpeechRecognition
SpeakerDiarization = SimpleSpeakerDiarization
SceneAnnotation = SimpleSceneAnnotation


# ============================================================================
# Migration Helpers
# ============================================================================

def get_recommended_schema(use_case: str):
    """Get the recommended schema for a specific use case."""
    
    recommendations = {
        "pytorch_training": {
            "person": COCOPersonDetection,
            "pose": COCOPoseKeypoints,
            "face": COCOFaceDetection,
            "scene": COCOSceneDetection
        },
        "annotation_tool": {
            "person": CVATPersonDetection,
            "pose": CVATAnnotation,
            "face": CVATAnnotation,
            "scene": CVATAnnotation
        },
        "research_export": {
            "person": LabelStudioPersonDetection,
            "pose": LabelStudioResult,
            "face": LabelStudioResult,
            "scene": LabelStudioResult
        },
        "audio_processing": {
            "transcription": WebVTTTranscription,
            "diarization": RTTMSpeakerDiarization,
            "classification": AudioSetClassification
        },
        "prototyping": {
            "person": PersonDetection,
            "pose": PoseKeypoints,
            "face": FaceDetection,
            "emotion": FaceEmotion,
            "speech": SpeechRecognition,
            "speaker": SpeakerDiarization,
            "scene": SceneAnnotation
        }
    }
    
    return recommendations.get(use_case, recommendations["prototyping"])


def migrate_legacy_annotation(legacy_dict: dict, annotation_type: str) -> any:
    """Convert legacy annotation dictionaries to new schemas."""
    
    # Detect legacy format and convert
    if legacy_dict.get("type") == "person_bbox":
        return PersonDetection(**legacy_dict)
    elif legacy_dict.get("type") == "pose_keypoints":
        return PoseKeypoints(**legacy_dict)
    elif legacy_dict.get("type") == "face_detection":
        return FaceDetection(**legacy_dict)
    elif legacy_dict.get("type") == "facial_emotion":
        return FaceEmotion(**legacy_dict)
    elif legacy_dict.get("type") == "speech_recognition":
        return SpeechRecognition(**legacy_dict)
    elif legacy_dict.get("type") == "speaker_diarization":
        return SpeakerDiarization(**legacy_dict)
    elif legacy_dict.get("type") == "scene_detection":
        return SceneAnnotation(**legacy_dict)
    else:
        # Fallback to simple base annotation
        return from_dict(legacy_dict)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Core COCO formats (recommended for ML workflows)
    "COCOPersonDetection",
    "COCOPoseKeypoints", 
    "COCOFaceDetection",
    "COCOSceneDetection",
    
    # Annotation tool formats
    "CVATPersonDetection",
    "LabelStudioPersonDetection",
    
    # Audio formats
    "WebVTTTranscription",
    "RTTMSpeakerDiarization", 
    "AudioSetClassification",
    
    # Simple formats (prototyping)
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
    "export_audio_events_audioset",
    
    # Conversion utilities
    "to_cvat_format",
    "to_labelstudio_format",
    "from_dict",
    
    # Migration helpers
    "get_recommended_schema",
    "migrate_legacy_annotation"
]
