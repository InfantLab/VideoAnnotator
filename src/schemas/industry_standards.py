"""
VideoAnnotator Industry Standards Schemas

This module has been migrated to use native formats via src.exporters.native_formats.
Most functionality has been moved to direct integration with:
- pycocotools (COCO format)
- webvtt-py (WebVTT format) 
- pyannote.core (RTTM format)
- praatio (TextGrid format)

This file is preserved for compatibility but most imports have been disabled.
"""

# Placeholder for compatibility
class IndustryStandardsPlaceholder:
    """Placeholder class for migrated functionality."""
    pass

# Legacy compatibility - these are no longer used
def get_coco_format_for_pipeline(pipeline_type: str):
    """Legacy function - use native_formats.py instead."""
    return None

def get_webvtt_format_for_audio():
    """Legacy function - use native_formats.py instead."""
    return None
