"""
VideoAnnotator Schemas Package

After the standards migration, most functionality has been moved to:
- src.exporters.native_formats for direct FOSS library integration
- Direct use of pycocotools, webvtt-py, pyannote.core, praatio

This __init__.py is preserved for backward compatibility.
"""

# Minimal compatibility exports
__all__ = [
    "IndustryStandardsPlaceholder",
]

# Import from the new industry standards module
# Note: After migration, most of these are handled by native_formats.py
try:
    from .industry_standards import IndustryStandardsPlaceholder
except ImportError:
    # Fallback if industry_standards is not available
    IndustryStandardsPlaceholder = None


# Legacy compatibility message
def get_migration_info():
    """Return information about the standards migration."""
    return {
        "message": "Schemas have been migrated to native formats",
        "location": "src.exporters.native_formats",
        "formats": ["COCO", "WebVTT", "RTTM", "TextGrid"],
        "libraries": ["pycocotools", "webvtt-py", "pyannote.core", "praatio"],
    }
