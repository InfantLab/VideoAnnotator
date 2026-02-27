"""Unit tests for pipelines/face_analysis/openface_compatibility.py."""

import numpy as np
import pytest

from videoannotator.pipelines.face_analysis.openface_compatibility import (
    IS_COMPATIBILITY_LAYER,
    ActionUnitAnalyzer,
    EmotionRecognizer,
    FaceDetector,
    FaceTracker,
    GazeEstimator,
    HeadPoseEstimator,
    LandmarkDetector,
    OpenFaceNotAvailableError,
    get_default_model_paths,
    patch_scipy_compatibility,
)

# ---------------------------------------------------------------------------
# Module-level flag
# ---------------------------------------------------------------------------


class TestModuleFlag:
    def test_is_compatibility_layer(self):
        assert IS_COMPATIBILITY_LAYER is True


# ---------------------------------------------------------------------------
# OpenFaceNotAvailableError
# ---------------------------------------------------------------------------


class TestOpenFaceNotAvailableError:
    def test_is_exception(self):
        assert issubclass(OpenFaceNotAvailableError, Exception)

    def test_can_raise_with_message(self):
        with pytest.raises(OpenFaceNotAvailableError, match="not installed"):
            raise OpenFaceNotAvailableError("not installed")


# ---------------------------------------------------------------------------
# FaceDetector
# ---------------------------------------------------------------------------


class TestFaceDetector:
    def test_init_defaults(self):
        det = FaceDetector()
        assert det.model_path is None
        assert det.confidence_threshold == 0.5
        assert det.device == "cpu"

    def test_init_custom(self):
        det = FaceDetector(model_path="/m.pth", confidence_threshold=0.8, device="cuda")
        assert det.model_path == "/m.pth"
        assert det.device == "cuda"

    def test_detect_raises(self):
        det = FaceDetector()
        with pytest.raises(OpenFaceNotAvailableError):
            det.detect(np.zeros((100, 100, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# LandmarkDetector
# ---------------------------------------------------------------------------


class TestLandmarkDetector:
    def test_init_defaults(self):
        det = LandmarkDetector()
        assert det.model_type == "98_point"
        assert det.num_landmarks == 98
        assert det.enable_3d is False

    def test_init_68_point(self):
        det = LandmarkDetector(model_type="68_point")
        assert det.num_landmarks == 68

    def test_detect_raises(self):
        det = LandmarkDetector()
        with pytest.raises(OpenFaceNotAvailableError):
            det.detect(np.zeros((100, 100, 3), dtype=np.uint8), {})


# ---------------------------------------------------------------------------
# ActionUnitAnalyzer
# ---------------------------------------------------------------------------


class TestActionUnitAnalyzer:
    def test_init(self):
        au = ActionUnitAnalyzer(device="cuda")
        assert au.device == "cuda"

    def test_analyze_raises(self):
        au = ActionUnitAnalyzer()
        with pytest.raises(OpenFaceNotAvailableError):
            au.analyze(np.zeros((64, 64, 3), dtype=np.uint8), {})


# ---------------------------------------------------------------------------
# HeadPoseEstimator
# ---------------------------------------------------------------------------


class TestHeadPoseEstimator:
    def test_init(self):
        hp = HeadPoseEstimator()
        assert hp.device == "cpu"

    def test_estimate_raises(self):
        hp = HeadPoseEstimator()
        with pytest.raises(OpenFaceNotAvailableError):
            hp.estimate(np.zeros((64, 64, 3), dtype=np.uint8), {})


# ---------------------------------------------------------------------------
# GazeEstimator
# ---------------------------------------------------------------------------


class TestGazeEstimator:
    def test_init(self):
        ge = GazeEstimator()
        assert ge.device == "cpu"

    def test_estimate_raises(self):
        ge = GazeEstimator()
        with pytest.raises(OpenFaceNotAvailableError):
            ge.estimate(np.zeros((64, 64, 3), dtype=np.uint8), {})


# ---------------------------------------------------------------------------
# EmotionRecognizer
# ---------------------------------------------------------------------------


class TestEmotionRecognizer:
    def test_init(self):
        er = EmotionRecognizer()
        assert er.device == "cpu"

    def test_recognize_raises(self):
        er = EmotionRecognizer()
        with pytest.raises(OpenFaceNotAvailableError):
            er.recognize(np.zeros((64, 64, 3), dtype=np.uint8), {})


# ---------------------------------------------------------------------------
# FaceTracker
# ---------------------------------------------------------------------------


class TestFaceTracker:
    def test_init_default(self):
        ft = FaceTracker()
        assert ft.max_faces == 5

    def test_init_custom(self):
        ft = FaceTracker(max_faces=10)
        assert ft.max_faces == 10

    def test_track_raises(self):
        ft = FaceTracker()
        with pytest.raises(OpenFaceNotAvailableError):
            ft.track({})


# ---------------------------------------------------------------------------
# patch_scipy_compatibility
# ---------------------------------------------------------------------------


class TestPatchScipy:
    def test_real_scipy_runs_without_error(self):
        """The real patch function should not raise on the installed scipy."""
        patch_scipy_compatibility()  # should succeed silently

    def test_scipy_has_simps_after_patch(self):
        """After patching, scipy.integrate.simps should be available."""
        import scipy.integrate

        patch_scipy_compatibility()
        assert hasattr(scipy.integrate, "simps") or hasattr(scipy.integrate, "simpson")


# ---------------------------------------------------------------------------
# get_default_model_paths
# ---------------------------------------------------------------------------


class TestGetDefaultModelPaths:
    def test_returns_dict(self):
        paths = get_default_model_paths()
        assert isinstance(paths, dict)

    def test_expected_keys(self):
        paths = get_default_model_paths()
        expected = {
            "face_detector",
            "landmark_68",
            "landmark_98",
            "action_units",
            "head_pose",
            "gaze",
            "emotion",
        }
        assert set(paths.keys()) == expected

    def test_all_values_are_strings(self):
        paths = get_default_model_paths()
        for v in paths.values():
            assert isinstance(v, str)
