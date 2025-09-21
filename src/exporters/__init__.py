"""
Native format exporters using established FOSS libraries.

This module provides direct integration with industry-standard libraries
instead of custom schema implementations:

- pycocotools: Official COCO format support
- webvtt-py: WebVTT subtitle/caption format
- pyannote.core: RTTM speaker diarization format
- praatio: TextGrid format for speech analysis

Usage:
    from src.exporters import (
        export_coco_json,
        export_webvtt_captions,
        export_rttm_diarization,
        export_textgrid_speech,
        auto_export_annotations
    )
"""

from .native_formats import (
    # COCO format functions
    create_coco_annotation,
    create_coco_keypoints_annotation,
    create_coco_image_entry,
    export_coco_json,
    validate_coco_json,
    ValidationResult,
    # Audio format functions
    export_webvtt_captions,
    export_rttm_diarization,
    export_textgrid_speech,
    # Utility functions
    auto_export_annotations,
)

__all__ = [
    # COCO format
    "create_coco_annotation",
    "create_coco_keypoints_annotation",
    "create_coco_image_entry",
    "export_coco_json",
    "validate_coco_json",
    "ValidationResult",
    # Audio formats
    "export_webvtt_captions",
    "export_rttm_diarization",
    "export_textgrid_speech",
    # Auto-export
    "auto_export_annotations",
]
