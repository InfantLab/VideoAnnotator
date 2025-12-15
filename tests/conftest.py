"""Shared pytest fixtures and environment patches for VideoAnnotator tests."""

import pytest


# --- Speech Pipeline Robustness Fixture ---
@pytest.fixture(autouse=True)
def patch_speech_pipeline_cuda(monkeypatch):
    """Respect actual CUDA availability instead of forcing True.

    Previous version forced torch.cuda.is_available() -> True which caused
    GPU-only code paths (e.g., Whisper model allocation) to execute on systems
    without CUDA support leading to failures. We now:
      - Return the real torch.cuda.is_available() result
      - Allow an opt-in override by setting TEST_FORCE_CUDA=1 in env
    This keeps tests deterministic while remaining safe on CPU-only machines.
    """
    import os

    try:
        import torch

        if os.environ.get("TEST_FORCE_CUDA") == "1":
            monkeypatch.setattr("torch.cuda.is_available", lambda: True, raising=False)
        else:
            # Wrap original to avoid accidental mutation; always boolean
            orig = torch.cuda.is_available
            monkeypatch.setattr(
                "torch.cuda.is_available", lambda: bool(orig()), raising=False
            )
    except Exception:
        # If torch not present or any issue, leave unpatched (defaults to no CUDA)
        pass
    yield


# --- Test Environment Robustness Fixtures ---
import os


@pytest.fixture(autouse=True)
def disable_auth_for_tests(monkeypatch):
    """Disable API authentication for tests.

    v1.3.0 introduced secure-by-default authentication. For tests,
    we disable it unless explicitly testing authentication behavior.
    Individual tests can override this by setting AUTH_REQUIRED=true.
    """
    monkeypatch.setenv("AUTH_REQUIRED", "false")
    yield


@pytest.fixture(autouse=True)
def patch_hf_token(monkeypatch):
    """Patch HuggingFace token for tests: use real token if available, else fake."""
    real_token = os.environ.get("HF_AUTH_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    token = real_token if real_token else "FAKE_TOKEN_FOR_TESTING"
    monkeypatch.setenv("HF_AUTH_TOKEN", token)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", token)
    yield


@pytest.fixture(autouse=True)
def patch_pipeline_availability(monkeypatch):
    """Patch pipeline availability flags to True for all tests unless.

    explicitly testing absence.
    """
    # Patch for speech pipeline (whisper)
    try:
        monkeypatch.setattr(
            "videoannotator.pipelines.audio_processing.speech_pipeline.WHISPER_AVAILABLE",
            True,
            raising=False,
        )
    except Exception:
        pass
    # Patch for diarization pipeline (pyannote)
    try:
        monkeypatch.setattr(
            "videoannotator.pipelines.audio_processing.diarization_pipeline.PYANNOTE_AVAILABLE",
            True,
            raising=False,
        )
    except Exception:
        pass
    yield


@pytest.fixture(autouse=True, scope="session")
def test_storage_env(tmp_path_factory):
    """Configure isolated storage environment for tests.

    Sets STORAGE_ROOT and VIDEOANNOTATOR_DB_PATH to temporary directories
    to prevent tests from polluting the main application storage.
    """
    # Create temp directories
    tmp_path = tmp_path_factory.mktemp("storage_env")
    storage_root = tmp_path / "storage" / "jobs"
    storage_root.mkdir(parents=True, exist_ok=True)

    db_path = tmp_path / "test_videoannotator.db"

    # Set environment variables
    os.environ["STORAGE_ROOT"] = str(storage_root)
    os.environ["VIDEOANNOTATOR_DB_PATH"] = str(db_path)
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"

    # Patch the database engine to use the new path
    # We can't easily reload modules because of circular dependencies and Base class issues
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.pool import StaticPool

        import videoannotator.database.database as db_module

        # Dispose old engine
        if hasattr(db_module, "engine"):
            db_module.engine.dispose()

        # Create new engine
        new_engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False,
        )

        # Update module globals
        db_module.engine = new_engine
        db_module.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=new_engine
        )

        # Also update migrations module if it imported engine
        try:
            import videoannotator.database.migrations as migrations_module

            migrations_module.engine = new_engine
        except ImportError:
            pass

    except ImportError:
        pass

    # Clear caches to ensure new env vars are picked up
    # We use try-except to avoid import errors if modules aren't available yet
    try:
        from videoannotator.api.database import get_storage_backend

        get_storage_backend.cache_clear()
    except ImportError:
        pass

    try:
        from videoannotator.storage.manager import get_storage_provider

        get_storage_provider.cache_clear()
    except ImportError:
        pass

    yield

    # Cleanup is handled by tmp_path fixture, but we should clear caches again
    try:
        from videoannotator.api.database import get_storage_backend

        get_storage_backend.cache_clear()
    except ImportError:
        pass

    try:
        from videoannotator.storage.manager import get_storage_provider

        get_storage_provider.cache_clear()
    except ImportError:
        pass


import tempfile
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest


@pytest.fixture
def temp_video_file():
    """Create a temporary video file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        # Create a simple test video using OpenCV
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(f.name, fourcc, 30.0, (640, 480))

        # Write a few frames
        for i in range(90):  # 3 seconds at 30fps
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add some variation to frames
            frame[:, :, i % 3] = (i * 5) % 255
            out.write(frame)

        out.release()

        yield Path(f.name)

        # Cleanup
        Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def temp_audio_file():
    """Provide an audio file for testing.

    Uses real test audio if available, otherwise generates synthetic audio.
    For integration tests, real audio with speech is required.
    """
    # Check for real test audio first
    real_audio = Path(__file__).parent / "fixtures" / "audio" / "test.wav"

    if real_audio.exists():
        # Use real test audio (don't delete)
        yield real_audio
        return

    # Fall back to synthetic audio for unit tests
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        # Create a simple test audio file
        import scipy.io.wavfile as wavfile

        sample_rate = 16000
        duration = 3.0  # 3 seconds
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Generate a simple sine wave (NOT speech - for unit tests only)
        frequency = 440  # A4 note
        audio_data = np.sin(2 * np.pi * frequency * t)

        # Convert to int16
        audio_data = (audio_data * 32767).astype(np.int16)

        wavfile.write(f.name, sample_rate, audio_data)
        temp_path = Path(f.name)
    # File is now closed
    yield temp_path
    # Cleanup
    temp_path.unlink(missing_ok=True)


@pytest.fixture
def multi_speaker_audio_file():
    """Provide audio with multiple speakers for diarization testing.

    Uses real test audio if available, required for integration tests.
    """
    real_audio = Path(__file__).parent / "fixtures" / "audio" / "test.wav"

    if real_audio.exists():
        yield real_audio
    else:
        pytest.skip(
            "Multi-speaker audio not available - add tests/fixtures/audio/test.wav"
        )


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_yolo_model():
    """Mock YOLO model for testing."""
    model = Mock()
    model.predict.return_value = [Mock()]
    model.predict.return_value[0].boxes = Mock()
    model.predict.return_value[0].boxes.data = np.array(
        [
            [100, 100, 200, 200, 0.9, 0]  # x1, y1, x2, y2, confidence, class
        ]
    )
    return model


@pytest.fixture
def mock_whisper_model():
    """Mock Whisper model for testing."""
    model = Mock()
    model.transcribe.return_value = {
        "text": "Hello world this is a test",
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "Hello world"},
            {"start": 2.0, "end": 4.0, "text": "this is a test"},
        ],
    }
    return model


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "scene_detection": {
            "threshold": 0.3,
            "min_scene_length": 1.0,
            "use_adaptive_threshold": True,
        },
        "person_tracking": {
            "model_name": "yolo11s",
            "confidence_threshold": 0.5,
            "iou_threshold": 0.7,
        },
        "face_analysis": {
            "backends": ["deepface"],
            "detection_confidence": 0.7,
            "enable_landmarks": True,
        },
        "audio_processing": {
            "whisper_model": "base",
            "sample_rate": 16000,
            "chunk_duration": 30.0,
        },
    }


@pytest.fixture
def sample_scene_results():
    """Sample scene detection results for testing."""
    return {
        "scenes": [
            {
                "scene_id": "scene_001",
                "start_time": 0.0,
                "end_time": 5.0,
                "scene_type": "indoor",
                "confidence": 0.85,
            },
            {
                "scene_id": "scene_002",
                "start_time": 5.0,
                "end_time": 10.0,
                "scene_type": "outdoor",
                "confidence": 0.92,
            },
        ],
        "total_duration": 10.0,
        "total_scenes": 2,
    }


@pytest.fixture
def sample_person_results():
    """Sample person tracking results for testing."""
    return {
        "tracks": [
            {
                "track_id": "person_001",
                "duration": 8.0,
                "detections": [
                    {
                        "timestamp": 1.0,
                        "bounding_box": [100, 100, 200, 200],
                        "confidence": 0.9,
                    }
                ],
            }
        ],
        "total_tracks": 1,
        "total_detections": 240,
    }


@pytest.fixture
def sample_face_results():
    """Sample face analysis results for testing."""
    return {
        "faces": [
            {
                "face_id": "face_001",
                "timestamp": 1.0,
                "bounding_box": [50, 50, 100, 100],
                "confidence": 0.8,
            }
        ],
        "face_tracks": [
            {"track_id": "face_track_001", "duration": 5.0, "faces": ["face_001"]}
        ],
        "total_faces": 1,
    }


@pytest.fixture
def sample_audio_results():
    """Sample audio processing results for testing."""
    return {
        "duration": 10.0,
        "sample_rate": 16000,
        "speech_transcription": {
            "text": "Hello world this is a test",
            "language": "en",
            "confidence": 0.9,
        },
        "speaker_diarization": {
            "num_speakers": 2,
            "segments": [
                {"start_time": 0.0, "end_time": 5.0, "speaker_id": "speaker_001"},
                {"start_time": 5.0, "end_time": 10.0, "speaker_id": "speaker_002"},
            ],
        },
    }


# Test markers for different test categories
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.performance = pytest.mark.performance
pytest.mark.slow = pytest.mark.slow


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "performance: mark test as a performance test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add unit marker to unit tests
        if "unit" in item.nodeid or "Unit" in item.name:
            item.add_marker(pytest.mark.unit)

        # Add integration marker to integration tests
        if "integration" in item.nodeid or "Integration" in item.name:
            item.add_marker(pytest.mark.integration)

        # Add performance marker to performance tests
        if "performance" in item.nodeid or "Performance" in item.name:
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)

        # Add slow marker to slow tests
        if "slow" in item.nodeid or any(
            keyword in item.name.lower() for keyword in ["batch", "large", "stress"]
        ):
            item.add_marker(pytest.mark.slow)
