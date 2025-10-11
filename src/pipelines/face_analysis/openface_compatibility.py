"""OpenFace 3.0 compatibility layer for VideoAnnotator.

This module provides compatibility interfaces for OpenFace 3.0 when the
actual OpenFace library is not installed. Instead of generating fake
data, it raises clear errors indicating that OpenFace needs to be
installed.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Module-level flag to indicate this is a compatibility layer
IS_COMPATIBILITY_LAYER = True


class OpenFaceNotAvailableError(Exception):
    """Exception raised when OpenFace functionality is not available."""

    pass


class FaceDetector:
    """OpenFace 3.0 FaceDetector compatibility interface."""

    def __init__(
        self,
        model_path: str | None = None,
        confidence_threshold: float = 0.5,
        device: str = "cpu",
    ):
        """Initialize the FaceDetector compatibility interface."""
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        logger.warning("Using OpenFace FaceDetector compatibility layer")

    def detect(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Detect faces in image."""
        raise OpenFaceNotAvailableError(
            "OpenFace 3.0 is not installed. Please install it from "
            "https://github.com/CMU-MultiComp-Lab/OpenFace-3.0"
        )


class LandmarkDetector:
    """OpenFace 3.0 LandmarkDetector compatibility interface."""

    def __init__(
        self,
        model_path: str | None = None,
        model_type: str = "98_point",
        enable_3d: bool = False,
        device: str = "cpu",
    ):
        """Initialize the LandmarkDetector compatibility interface."""
        self.model_path = model_path
        self.model_type = model_type
        self.enable_3d = enable_3d
        self.device = device
        self.num_landmarks = 98 if "98" in model_type else 68
        logger.warning("Using OpenFace LandmarkDetector compatibility layer")

    def detect(self, image: np.ndarray, face_bbox: dict[str, Any]) -> dict[str, Any]:
        """Detect facial landmarks."""
        raise OpenFaceNotAvailableError(
            "OpenFace 3.0 is not installed. Please install it from "
            "https://github.com/CMU-MultiComp-Lab/OpenFace-3.0"
        )


class ActionUnitAnalyzer:
    """OpenFace 3.0 ActionUnitAnalyzer compatibility interface."""

    def __init__(self, device: str = "cpu"):
        """Initialize the ActionUnitAnalyzer compatibility interface."""
        self.device = device
        logger.warning("Using OpenFace ActionUnitAnalyzer compatibility layer")

    def analyze(
        self, face_image: np.ndarray, landmarks: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze Action Units."""
        raise OpenFaceNotAvailableError(
            "OpenFace 3.0 is not installed. Please install it from "
            "https://github.com/CMU-MultiComp-Lab/OpenFace-3.0"
        )


class HeadPoseEstimator:
    """OpenFace 3.0 HeadPoseEstimator compatibility interface."""

    def __init__(self, device: str = "cpu"):
        """Initialize the HeadPoseEstimator compatibility interface."""
        self.device = device
        logger.warning("Using OpenFace HeadPoseEstimator compatibility layer")

    def estimate(
        self, face_image: np.ndarray, landmarks: dict[str, Any]
    ) -> dict[str, Any]:
        """Estimate head pose."""
        raise OpenFaceNotAvailableError(
            "OpenFace 3.0 is not installed. Please install it from "
            "https://github.com/CMU-MultiComp-Lab/OpenFace-3.0"
        )


class GazeEstimator:
    """OpenFace 3.0 GazeEstimator compatibility interface."""

    def __init__(self, device: str = "cpu"):
        """Initialize the GazeEstimator compatibility interface."""
        self.device = device
        logger.warning("Using OpenFace GazeEstimator compatibility layer")

    def estimate(
        self, face_image: np.ndarray, landmarks: dict[str, Any]
    ) -> dict[str, Any]:
        """Estimate gaze direction."""
        raise OpenFaceNotAvailableError(
            "OpenFace 3.0 is not installed. Please install it from "
            "https://github.com/CMU-MultiComp-Lab/OpenFace-3.0"
        )


class EmotionRecognizer:
    """OpenFace 3.0 EmotionRecognizer compatibility interface."""

    def __init__(self, device: str = "cpu"):
        """Initialize the EmotionRecognizer compatibility interface."""
        self.device = device
        logger.warning("Using OpenFace EmotionRecognizer compatibility layer")

    def recognize(
        self, face_image: np.ndarray, landmarks: dict[str, Any]
    ) -> dict[str, Any]:
        """Recognize emotions."""
        raise OpenFaceNotAvailableError(
            "OpenFace 3.0 is not installed. Please install it from "
            "https://github.com/CMU-MultiComp-Lab/OpenFace-3.0"
        )


class FaceTracker:
    """OpenFace 3.0 FaceTracker compatibility interface."""

    def __init__(self, max_faces: int = 5):
        """Initialize the FaceTracker compatibility interface."""
        self.max_faces = max_faces
        logger.warning("Using OpenFace FaceTracker compatibility layer")

    def track(self, detection: dict[str, Any]) -> int:
        """Track faces."""
        raise OpenFaceNotAvailableError(
            "OpenFace 3.0 is not installed. Please install it from "
            "https://github.com/CMU-MultiComp-Lab/OpenFace-3.0"
        )


def patch_scipy_compatibility():
    """Patch SciPy compatibility issues with OpenFace.

    OpenFace uses scipy.integrate.simps which was deprecated and removed
    in SciPy 1.14.0+. This function provides a compatibility layer.
    """
    try:
        import scipy.integrate

        # Check if simps is missing (SciPy 1.14.0+)
        if not hasattr(scipy.integrate, "simps"):
            logger.info("Applying scipy.integrate.simps compatibility patch")
            # Use simpson instead of simps
            scipy.integrate.simps = scipy.integrate.simpson
            logger.info("Successfully patched scipy.integrate.simps")
        else:
            logger.debug("scipy.integrate.simps already available, no patch needed")

    except ImportError as e:
        logger.warning(f"Failed to apply scipy compatibility patch: {e}")
    except Exception as e:
        logger.error(f"Unexpected error applying scipy patch: {e}")


def get_default_model_paths():
    """Get default model paths for OpenFace.

    Returns:
        dict: Dictionary with model type as key and path as value
    """
    return {
        "face_detector": "./weights/Alignment_RetinaFace.pth",
        "landmark_68": "./weights/Landmark_68.pkl",
        "landmark_98": "./weights/Landmark_98.pkl",
        "action_units": "./weights/ActionUnits.pkl",
        "head_pose": "./weights/HeadPose.pkl",
        "gaze": "./weights/Gaze.pkl",
        "emotion": "./weights/Emotion.pkl",
    }
