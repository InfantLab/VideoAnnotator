"""
Face Analysis Pipeline

This module provides face detection, recognition, and emotion analysis capabilities.
"""

from .face_pipeline import FaceAnalysisPipeline
from .laion_face_pipeline import LAIONFacePipeline

# Optional OpenFace 3.0 pipeline
try:
    from .openface3_pipeline import OpenFace3Pipeline

    OPENFACE3_AVAILABLE = True
except ImportError:
    OpenFace3Pipeline = None
    OPENFACE3_AVAILABLE = False

__all__ = [
    "OPENFACE3_AVAILABLE",
    "FaceAnalysisPipeline",
    "LAIONFacePipeline",
    "OpenFace3Pipeline",
]
