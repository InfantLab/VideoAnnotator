"""Integration tests for configuration validation in job submission.

Tests the integration of ConfigValidator into the POST /api/v1/jobs/ endpoint,
ensuring invalid configurations are rejected with helpful error messages before
job creation.
"""

import io
import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from videoannotator.api.database import reset_storage_backend, set_database_path
from videoannotator.api.main import create_app


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = Path(f.name)

    set_database_path(db_path)
    reset_storage_backend()

    yield db_path

    # Cleanup
    reset_storage_backend()
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def client(temp_db):
    """Create test client with temporary database."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_video_file():
    """Create a minimal video file for testing."""
    from tests.fixtures.synthetic_video import synthetic_video_bytes_avi

    return io.BytesIO(synthetic_video_bytes_avi())


class TestJobSubmissionValidation:
    """Test configuration validation in job submission endpoint."""

    def test_submit_job_with_valid_config(self, client, sample_video_file):
        """Test job submission with valid configuration succeeds."""
        config = {
            "confidence_threshold": 0.8,
            "iou_threshold": 0.5,
        }
        pipelines = "person_tracking,scene_detection"

        files = {"video": ("test.avi", sample_video_file, "video/avi")}
        data = {
            "config": json.dumps(config),
            "selected_pipelines": pipelines,
        }

        response = client.post("/api/v1/jobs/", files=files, data=data)
        assert response.status_code == 201

        job_data = response.json()
        assert job_data["config"] == config
        assert job_data["selected_pipelines"] == ["person_tracking", "scene_detection"]
        assert job_data["status"] == "pending"

    def test_submit_job_with_invalid_confidence_threshold(
        self, client, sample_video_file
    ):
        """Test job submission with out-of-range confidence_threshold is rejected."""
        config = {
            "confidence_threshold": 1.5,  # Invalid: > 1.0
        }
        pipelines = "person_tracking"

        files = {"video": ("test.avi", sample_video_file, "video/avi")}
        data = {
            "config": json.dumps(config),
            "selected_pipelines": pipelines,
        }

        response = client.post("/api/v1/jobs/", files=files, data=data)
        assert response.status_code == 400

        error_data = response.json()
        detail = error_data["detail"].lower()
        assert "validation failed" in detail
        assert "confidence" in detail or "1.5" in detail or "range" in detail

    def test_submit_job_with_invalid_iou_threshold(self, client, sample_video_file):
        """Test job submission with negative iou_threshold is rejected."""
        config = {
            "iou_threshold": -0.1,  # Invalid: negative
        }
        pipelines = "person_tracking"

        files = {"video": ("test.avi", sample_video_file, "video/avi")}
        data = {
            "config": json.dumps(config),
            "selected_pipelines": pipelines,
        }

        response = client.post("/api/v1/jobs/", files=files, data=data)
        assert response.status_code == 400

        error_data = response.json()
        detail = error_data["detail"].lower()
        assert "validation failed" in detail
        assert "iou" in detail or "-0.1" in detail or "range" in detail

    def test_submit_job_with_unknown_pipeline(self, client, sample_video_file):
        """Test job submission with unknown pipeline is rejected."""
        config = {"confidence_threshold": 0.8}
        pipelines = "unknown_pipeline"

        files = {"video": ("test.avi", sample_video_file, "video/avi")}
        data = {
            "config": json.dumps(config),
            "selected_pipelines": pipelines,
        }

        response = client.post("/api/v1/jobs/", files=files, data=data)
        assert response.status_code == 400

        error_data = response.json()
        assert "validation failed" in error_data["detail"].lower()
        assert "unknown pipeline" in error_data["detail"].lower()
        # Should suggest available pipelines
        assert "person_tracking" in error_data["detail"].lower()

    def test_submit_job_with_multiple_unknown_pipelines(
        self, client, sample_video_file
    ):
        """Test job submission with multiple unknown pipelines shows all errors."""
        config = {"confidence_threshold": 0.8}
        pipelines = "unknown1,unknown2"

        files = {"video": ("test.avi", sample_video_file, "video/avi")}
        data = {
            "config": json.dumps(config),
            "selected_pipelines": pipelines,
        }

        response = client.post("/api/v1/jobs/", files=files, data=data)
        assert response.status_code == 400

        error_data = response.json()
        assert "validation failed" in error_data["detail"].lower()
        assert "unknown1" in error_data["detail"]
        assert "unknown2" in error_data["detail"]

    def test_submit_job_without_config_succeeds(self, client, sample_video_file):
        """Test job submission without config uses defaults (valid)."""
        pipelines = "person_tracking"

        files = {"video": ("test.avi", sample_video_file, "video/avi")}
        data = {"selected_pipelines": pipelines}

        response = client.post("/api/v1/jobs/", files=files, data=data)
        assert response.status_code == 201

        job_data = response.json()
        assert job_data["config"] == {}
        assert job_data["selected_pipelines"] == ["person_tracking"]

    def test_submit_job_without_pipelines_skips_validation(
        self, client, sample_video_file
    ):
        """Test job submission without pipelines skips validation."""
        config = {
            "confidence_threshold": 2.0,  # Invalid but not validated
        }

        files = {"video": ("test.avi", sample_video_file, "video/avi")}
        data = {"config": json.dumps(config)}

        response = client.post("/api/v1/jobs/", files=files, data=data)
        # Should succeed because validation only runs when pipelines are specified
        assert response.status_code == 201

    def test_submit_job_with_empty_config_and_pipelines(
        self, client, sample_video_file
    ):
        """Test job submission with empty config and pipelines specified (uses defaults)."""
        config = {}
        pipelines = "scene_detection"

        files = {"video": ("test.avi", sample_video_file, "video/avi")}
        data = {
            "config": json.dumps(config),
            "selected_pipelines": pipelines,
        }

        response = client.post("/api/v1/jobs/", files=files, data=data)
        assert response.status_code == 201  # Empty config is valid (defaults)

        job_data = response.json()
        assert job_data["config"] == {}
        assert job_data["selected_pipelines"] == ["scene_detection"]

    def test_submit_job_with_mixed_valid_invalid_pipelines(
        self, client, sample_video_file
    ):
        """Test job submission with mix of valid and invalid pipelines is rejected."""
        config = {"confidence_threshold": 0.8}
        pipelines = "person_tracking,invalid_pipeline"

        files = {"video": ("test.avi", sample_video_file, "video/avi")}
        data = {
            "config": json.dumps(config),
            "selected_pipelines": pipelines,
        }

        response = client.post("/api/v1/jobs/", files=files, data=data)
        assert response.status_code == 400

        error_data = response.json()
        assert "validation failed" in error_data["detail"].lower()
        assert "invalid_pipeline" in error_data["detail"]

    def test_submit_job_with_wrong_type_config(self, client, sample_video_file):
        """Test job submission with wrong type for config field is rejected."""
        config = {
            "confidence_threshold": "not_a_number",  # Should be float
        }
        pipelines = "person_tracking"

        files = {"video": ("test.avi", sample_video_file, "video/avi")}
        data = {
            "config": json.dumps(config),
            "selected_pipelines": pipelines,
        }

        response = client.post("/api/v1/jobs/", files=files, data=data)
        assert response.status_code == 400

        error_data = response.json()
        assert "validation failed" in error_data["detail"].lower()

    def test_submit_job_with_multiple_validation_errors(
        self, client, sample_video_file
    ):
        """Test job submission with multiple validation errors shows all."""
        config = {
            "confidence_threshold": 1.5,  # Too high
            "iou_threshold": -0.5,  # Negative
            "max_persons": 0,  # Too low
        }
        pipelines = "person_tracking"

        files = {"video": ("test.avi", sample_video_file, "video/avi")}
        data = {
            "config": json.dumps(config),
            "selected_pipelines": pipelines,
        }

        response = client.post("/api/v1/jobs/", files=files, data=data)
        assert response.status_code == 400

        error_data = response.json()
        detail = error_data["detail"].lower()
        assert "validation failed" in detail
        # Should mention multiple errors - check for multiple semicolons or multiple mentions of "range"
        assert detail.count(";") >= 2 or detail.count("range") >= 2

    def test_submit_job_with_all_pipeline_types(self, client, sample_video_file):
        """Test job submission with all valid pipeline types succeeds."""
        config = {
            "confidence_threshold": 0.7,
            "threshold": 0.5,
            "sample_rate": 16000,
        }
        pipelines = "person_tracking,scene_detection,face_analysis,audio_processing"

        files = {"video": ("test.avi", sample_video_file, "video/avi")}
        data = {
            "config": json.dumps(config),
            "selected_pipelines": pipelines,
        }

        response = client.post("/api/v1/jobs/", files=files, data=data)
        assert response.status_code == 201

        job_data = response.json()
        assert len(job_data["selected_pipelines"]) == 4
        assert "person_tracking" in job_data["selected_pipelines"]
        assert "scene_detection" in job_data["selected_pipelines"]
        assert "face_analysis" in job_data["selected_pipelines"]
        assert "audio_processing" in job_data["selected_pipelines"]

    def test_error_response_structure(self, client, sample_video_file):
        """Test that validation error responses have proper structure."""
        config = {"confidence_threshold": 2.0}
        pipelines = "person_tracking"

        files = {"video": ("test.avi", sample_video_file, "video/avi")}
        data = {
            "config": json.dumps(config),
            "selected_pipelines": pipelines,
        }

        response = client.post("/api/v1/jobs/", files=files, data=data)
        assert response.status_code == 400

        error_data = response.json()
        # Check error envelope structure
        assert "error" in error_data or "detail" in error_data
        assert isinstance(error_data, dict)

    def test_validation_hint_included_in_response(self, client, sample_video_file):
        """Test that validation errors include helpful hints."""
        config = {"confidence_threshold": 1.5}
        pipelines = "person_tracking"

        files = {"video": ("test.avi", sample_video_file, "video/avi")}
        data = {
            "config": json.dumps(config),
            "selected_pipelines": pipelines,
        }

        response = client.post("/api/v1/jobs/", files=files, data=data)
        assert response.status_code == 400

        error_data = response.json()
        detail = error_data["detail"]
        # Check that hint about valid range is included
        assert "0" in detail or "1" in detail  # Range hint should mention 0-1
