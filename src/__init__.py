"""
VideoAnnotator - Modern Video Annotation Pipeline

A comprehensive toolkit for video analysis including scene detection,
person tracking, face analysis, and audio processing.
"""

from .version import (
    __author__,
    __license__,
    __version__,
    __version_info__,
    create_annotation_metadata,
    get_version_info,
    print_version_info,
)

# Heavy pipeline imports commented out to prevent pytest collection hangs
# Uncomment when needed for production use
# from .pipelines import (
#     BasePipeline,
#     SceneDetectionPipeline,
#     PersonTrackingPipeline,
#     FaceAnalysisPipeline,
#     AudioPipeline,
# )

# Note: After standards migration, these schemas are no longer used
# Pipelines now return native format dictionaries (COCO, WebVTT, RTTM, etc.)
# from .schemas import (
#     AnnotationBase,
#     VideoMetadata,
#     SceneSegment,
#     SceneAnnotation,
#     PersonDetection,
#     FaceDetection,
#     SpeechSegment,
# )

__all__ = [
    # Version information
    "__version__",
    "__version_info__",
    "__author__",
    "__license__",
    "get_version_info",
    "print_version_info",
    "create_annotation_metadata",
    # Pipelines
    "BasePipeline",
    "SceneDetectionPipeline",
    "PersonTrackingPipeline",
    "FaceAnalysisPipeline",
    "AudioPipeline",
    # Note: Schemas no longer exported after standards migration
    # "AnnotationBase",
    # "VideoMetadata",
    # "SceneSegment",
    # "SceneAnnotation",
    # "PersonDetection",
    # "FaceDetection",
    # "SpeechSegment",
]
