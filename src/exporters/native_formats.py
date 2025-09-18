"""
Native FOSS library integration for VideoAnnotator.

This module uses established FOSS libraries directly instead of custom schema wrappers:
- pycocotools: Official COCO format support
- webvtt-py: WebVTT subtitle/caption format
- pyannote.core: RTTM speaker diarization format
- praatio: TextGrid format for speech analysis
- audformat: Comprehensive audio annotation library

Usage:
    from src.exporters.native_formats import (
        export_coco_json,
        export_webvtt_captions,
        export_rttm_diarization,
        export_textgrid_speech
    )
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, NamedTuple, TYPE_CHECKING

# Import version information
from version import __version__

if TYPE_CHECKING:
    try:
        from pycocotools.coco import COCO
    except ImportError:
        COCO = None

logger = logging.getLogger(__name__)


class ValidationResult(NamedTuple):
    """Result of COCO validation."""

    is_valid: bool
    warnings: List[str]
    errors: List[str]


# Import native FOSS libraries
try:
    from pycocotools.coco import COCO

    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    # Define a placeholder for type annotations when pycocotools is not available
    COCO = None
    logger.warning("pycocotools not available. Install with: pip install pycocotools")

try:
    import webvtt

    WEBVTT_AVAILABLE = True
except ImportError:
    WEBVTT_AVAILABLE = False
    logger.warning("webvtt-py not available. Install with: pip install webvtt-py")

try:
    from pyannote.core import Annotation, Segment

    PYANNOTE_CORE_AVAILABLE = True
except ImportError:
    PYANNOTE_CORE_AVAILABLE = False
    logger.warning("pyannote.core not available. Install with: pip install pyannote.core")

try:
    import praatio

    PRAATIO_AVAILABLE = True
except ImportError:
    PRAATIO_AVAILABLE = False
    logger.warning("praatio not available. Install with: pip install praatio")


# ============================================================================
# COCO Format (Native pycocotools)
# ============================================================================


def create_coco_image_entry(
    image_id: Union[int, str], width: int, height: int, file_name: str, **kwargs
) -> Dict[str, Any]:
    """Create COCO image entry using native format."""
    image = {"id": image_id, "width": width, "height": height, "file_name": file_name}

    # Add any additional fields (video_id, frame_number, timestamp, etc.)
    image.update(kwargs)
    return image


def create_coco_annotation(
    annotation_id: int,
    image_id: Union[int, str],
    category_id: int,
    bbox: List[float],
    area: Optional[float] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Create COCO annotation using native format (no custom classes)."""
    if area is None:
        area = bbox[2] * bbox[3]  # width * height

    annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,  # [x, y, width, height]
        "area": area,
        "iscrowd": 0,
    }

    # Add any additional fields (score, track_id, etc.)
    annotation.update(kwargs)
    return annotation


def create_coco_keypoints_annotation(
    annotation_id: int,
    image_id: Union[int, str],
    category_id: int,
    keypoints: List[float],  # Flattened [x1,y1,v1,x2,y2,v2,...]
    bbox: List[float],
    num_keypoints: int,
    **kwargs,
) -> Dict[str, Any]:
    """Create COCO keypoints annotation using native format."""
    annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "keypoints": keypoints,
        "num_keypoints": num_keypoints,
        "bbox": bbox,
        "area": bbox[2] * bbox[3],
        "iscrowd": 0,
    }

    annotation.update(kwargs)
    return annotation


def export_coco_json(
    annotations: List[Dict[str, Any]],
    images: List[Dict[str, Any]],
    output_path: str,
    categories: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Any]:
    """
    Export to COCO JSON format using native structure.

    Args:
        annotations: List of COCO annotation dictionaries
        images: List of COCO image dictionaries
        output_path: Path to save COCO JSON file
        categories: List of category dictionaries (default: person category)

    Returns:
        COCO object for validation and further use
    """
    if not PYCOCOTOOLS_AVAILABLE:
        raise ImportError("pycocotools required for COCO export")

    if categories is None:
        categories = [{"id": 1, "name": "person", "supercategory": "person"}]

    # Create native COCO format
    coco_data = {
        "info": {
            "description": "VideoAnnotator COCO Export",
            "version": __version__,
            "year": 2025,
            "contributor": "VideoAnnotator",
            "date_created": "2025-01-01T00:00:00Z",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    # Save to file
    with open(output_path, "w") as f:
        json.dump(coco_data, f, indent=2)

    # Load and validate with official COCO API
    try:
        coco = COCO(output_path)
        logger.info(f"COCO dataset exported and validated: {output_path}")
        logger.info(f"Images: {len(images)}, Annotations: {len(annotations)}")
        return coco
    except Exception as e:
        logger.error(f"COCO validation failed: {e}")
        raise


def validate_coco_json(coco_path: str, context: str = "") -> ValidationResult:
    """
    Validate COCO JSON file using native pycocotools.

    Args:
        coco_path: Path to COCO JSON file
        context: Context for logging (e.g., "face_analysis")

    Returns:
        ValidationResult with validation status and any warnings/errors
    """
    if not PYCOCOTOOLS_AVAILABLE:
        return ValidationResult(False, [], ["pycocotools not available"])

    warnings = []
    errors = []

    try:
        # Load with official COCO API
        coco = COCO(coco_path)

        # Basic validation checks
        if len(coco.getImgIds()) == 0:
            warnings.append("No images found in COCO dataset")

        if len(coco.getAnnIds()) == 0:
            warnings.append("No annotations found in COCO dataset")

        # Check category IDs are valid
        cat_ids = coco.getCatIds()
        if not cat_ids:
            warnings.append("No categories defined")

        logger.info(f"COCO validation successful for {context}: {coco_path}")
        return ValidationResult(True, warnings, errors)

    except Exception as e:
        error_msg = f"COCO validation failed for {context}: {str(e)}"
        logger.error(error_msg)
        errors.append(error_msg)
        return ValidationResult(False, warnings, errors)


# ============================================================================
# WebVTT Format (Native webvtt-py)
# ============================================================================


def export_webvtt_captions(speech_segments: List[Dict[str, Any]], output_path: str) -> None:
    """
    Export speech transcription to WebVTT format using native library.

    Args:
        speech_segments: List of dicts with 'start', 'end', 'text' keys
        output_path: Path to save WebVTT file
    """
    if not WEBVTT_AVAILABLE:
        raise ImportError("webvtt-py required for WebVTT export")

    # Create WebVTT using native library
    captions = webvtt.WebVTT()

    for segment in speech_segments:
        # Convert seconds to WebVTT time format
        start_time = _seconds_to_webvtt_time(segment["start"])
        end_time = _seconds_to_webvtt_time(segment["end"])

        caption = webvtt.Caption(start=start_time, end=end_time, text=segment["text"])
        captions.append(caption)

    # Save using native library
    captions.save(output_path)
    logger.info(f"WebVTT captions exported: {output_path}")


def export_webvtt(
    segments: List[Dict[str, Any]],
    output_path: Union[str, Path],
    include_metadata: bool = True
) -> bool:
    """
    Export results in WebVTT format.
    
    Args:
        segments: List of segment dictionaries with start_time, end_time, and optional text/speaker data
        output_path: Path to save the WebVTT file
        include_metadata: Whether to include extra metadata in WebVTT comments
        
    Returns:
        True if export was successful, False otherwise
    """
    if not WEBVTT_AVAILABLE:
        logger.error("WebVTT export failed: webvtt-py library not available")
        return False
    
    try:
        output_path = Path(output_path)
        vtt = webvtt.WebVTT()
        
        for segment in segments:
            # Required fields
            start_time = segment.get("start_time", 0)
            end_time = segment.get("end_time", 0)
            
            # Format times as WebVTT timestamps (HH:MM:SS.mmm)
            start_str = _format_timestamp(start_time)
            end_str = _format_timestamp(end_time)
            
            # Get content text
            text = segment.get("transcription", "")
            if not text and "text" in segment:
                text = segment["text"]
                
            # Add speaker information if available
            speaker_id = segment.get("speaker_id", segment.get("speaker", ""))
            if speaker_id:
                text = f"<v {speaker_id}>{text}"
            
            # Add emotions if available and metadata is enabled
            emotions = segment.get("emotions", {})
            if emotions and include_metadata:
                emotion_text = ", ".join([
                    f"{emotion}({data['score']:.2f})" 
                    for emotion, data in emotions.items()
                    if isinstance(data, dict) and "score" in data
                ])
                if emotion_text:
                    text += f"\nEMOTIONS: {emotion_text}"
            
            # Create caption
            caption = webvtt.Caption(
                start=start_str,
                end=end_str,
                text=text
            )
            vtt.captions.append(caption)
        
        # Add header information
        vtt.file = str(output_path)
        
        # Save the WebVTT file
        vtt.save()
        logger.info(f"WebVTT file saved to: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to export WebVTT: {e}")
        return False


def _seconds_to_webvtt_time(seconds: float) -> str:
    """Convert seconds to WebVTT time format (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def _format_timestamp(seconds: float) -> str:
    """Format seconds as WebVTT timestamp (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


# ============================================================================
# RTTM Format (Native pyannote.core)
# ============================================================================


def export_rttm_diarization(
    speaker_segments: List[Dict[str, Any]], output_path: str, uri: str = "video"
) -> None:
    """
    Export speaker diarization to RTTM format using pyannote.core.

    Args:
        speaker_segments: List of dicts with 'start', 'end', 'speaker_id' keys
        output_path: Path to save RTTM file
        uri: URI identifier for the recording
    """
    if not PYANNOTE_CORE_AVAILABLE:
        raise ImportError("pyannote.core required for RTTM export")

    # Create Annotation using native pyannote.core
    diarization = Annotation(uri=uri)

    for segment in speaker_segments:
        segment_obj = Segment(segment["start"], segment["end"])
        diarization[segment_obj] = segment["speaker_id"]

    # Export to RTTM using native method
    with open(output_path, "w") as f:
        diarization.write_rttm(f)

    logger.info(f"RTTM diarization exported: {output_path}")


# ============================================================================
# TextGrid Format (Native praatio)
# ============================================================================


def export_textgrid_speech(
    speech_segments: List[Dict[str, Any]], output_path: str, tier_name: str = "speech"
) -> None:
    """
    Export speech transcription to TextGrid format using praatio.

    Args:
        speech_segments: List of dicts with 'start', 'end', 'text' keys
        output_path: Path to save TextGrid file
        tier_name: Name of the tier in TextGrid
    """
    if not PRAATIO_AVAILABLE:
        raise ImportError("praatio required for TextGrid export")

    # Calculate total duration
    max_time = max(segment["end"] for segment in speech_segments) if speech_segments else 0

    # Create TextGrid using native praatio
    tg = praatio.textgrid.Textgrid()

    # Create interval tier
    tier = praatio.textgrid.IntervalTier(tier_name, [], 0, max_time)

    # Add speech intervals
    for segment in speech_segments:
        tier.insertEntry((segment["start"], segment["end"], segment["text"]))

    # Add tier to TextGrid
    tg.addTier(tier)

    # Save using native method
    tg.save(output_path)
    logger.info(f"TextGrid exported: {output_path}")


# ============================================================================
# Utility Functions
# ============================================================================


def validate_coco_format(coco_file_path: str) -> bool:
    """Validate COCO format using official pycocotools."""
    if not PYCOCOTOOLS_AVAILABLE:
        logger.warning("pycocotools not available for validation")
        return False

    try:
        coco = COCO(coco_file_path)
        # Validate that we can access basic COCO properties
        _ = coco.getImgIds()
        _ = coco.getAnnIds()
        logger.info(f"COCO format validation successful: {coco_file_path}")
        return True
    except Exception as e:
        logger.error(f"COCO format validation failed: {e}")
        return False


# ============================================================================
# Format Detection and Auto-Export
# ============================================================================


def auto_export_annotations(
    annotations: List[Dict[str, Any]], output_dir: str, base_name: str = "annotations"
) -> Dict[str, str]:
    """
    Auto-export annotations to multiple native formats.

    Returns:
        Dictionary mapping format names to output file paths
    """
    output_paths = {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Separate annotations by type
    person_annotations = []
    keypoint_annotations = []
    speech_segments = []
    speaker_segments = []

    for ann in annotations:
        ann_type = ann.get("type", "")
        if "person" in ann_type or "bbox" in ann_type:
            person_annotations.append(ann)
        elif "keypoint" in ann_type or "pose" in ann_type:
            keypoint_annotations.append(ann)
        elif "speech" in ann_type or "transcript" in ann_type:
            speech_segments.append(ann)
        elif "speaker" in ann_type or "diarization" in ann_type:
            speaker_segments.append(ann)

    # Export COCO format for person/keypoint annotations
    if person_annotations or keypoint_annotations:
        # Combine and convert to COCO format
        coco_annotations = []
        coco_images = set()

        for i, ann in enumerate(person_annotations + keypoint_annotations):
            # Create COCO annotation from VideoAnnotator format
            coco_ann = create_coco_annotation(
                annotation_id=i + 1,
                image_id=ann.get("image_id", f"frame_{i}"),
                category_id=1,  # person
                bbox=ann.get("bbox", [0, 0, 100, 100]),
                score=ann.get("confidence", ann.get("score", 1.0)),
            )
            coco_annotations.append(coco_ann)

            # Track unique images
            coco_images.add(ann.get("image_id", f"frame_{i}"))

        # Create image entries
        images = [
            create_coco_image_entry(
                image_id=img_id,
                width=1920,  # Default, should be provided
                height=1080,
                file_name=f"{img_id}.jpg",
            )
            for img_id in sorted(coco_images)
        ]

        coco_path = output_dir / f"{base_name}.json"
        export_coco_json(coco_annotations, images, str(coco_path))
        output_paths["coco"] = str(coco_path)

    # Export WebVTT for speech
    if speech_segments:
        webvtt_path = output_dir / f"{base_name}.vtt"
        export_webvtt_captions(speech_segments, str(webvtt_path))
        output_paths["webvtt"] = str(webvtt_path)

        # Also export TextGrid
        textgrid_path = output_dir / f"{base_name}.TextGrid"
        export_textgrid_speech(speech_segments, str(textgrid_path))
        output_paths["textgrid"] = str(textgrid_path)

    # Export RTTM for speaker diarization
    if speaker_segments:
        rttm_path = output_dir / f"{base_name}.rttm"
        export_rttm_diarization(speaker_segments, str(rttm_path))
        output_paths["rttm"] = str(rttm_path)

    logger.info(f"Auto-exported annotations to {len(output_paths)} formats")
    return output_paths
