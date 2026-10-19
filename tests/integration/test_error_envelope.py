"""Integration tests for ErrorEnvelope format across all API endpoints."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client."""
    from videoannotator.api.main import create_app

    app = create_app()
    return TestClient(app)


class TestErrorEnvelopeFormat:
    """Test that all API endpoints return consistent ErrorEnvelope format."""

    def test_404_error_uses_error_envelope(self, client):
        """Test that 404 errors return ErrorEnvelope format."""
        # Request non-existent job
        response = client.get("/api/v1/jobs/nonexistent-job-id")

        assert response.status_code == 404
        data = response.json()

        # Verify ErrorEnvelope structure
        assert "error" in data
        error = data["error"]

        # Check all required fields
        assert "code" in error
        assert "message" in error
        assert "timestamp" in error

        # Check optional but expected fields
        assert "hint" in error

        # Verify specific error details
        assert error["code"] == "JOB_NOT_FOUND"
        assert "nonexistent-job-id" in error["message"]
        assert error["hint"]  # Should have helpful hint

    def test_400_validation_error_uses_error_envelope(self, client, tmp_path):
        """Test that 400 validation errors return ErrorEnvelope with field details."""
        # Create a test video file
        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"fake video data")

        # Submit job with invalid config (invalid JSON)
        with open(video_path, "rb") as video_file:
            response = client.post(
                "/api/v1/jobs/",
                data={
                    "config": "invalid json {{{",  # Invalid JSON
                },
                files={"video": video_file},
            )

        assert response.status_code == 400
        data = response.json()

        # Verify ErrorEnvelope structure
        assert "error" in data
        error = data["error"]

        # Check all required fields
        assert "code" in error
        assert "message" in error
        assert "timestamp" in error
        assert "hint" in error

        # Verify specific error details
        assert error["code"] == "INVALID_REQUEST"
        assert "hint" in error
        assert len(error["hint"]) > 0  # Should have helpful hint

    def test_pipeline_not_found_uses_error_envelope(self, client):
        """Test that pipeline not found errors return ErrorEnvelope."""
        # Request non-existent pipeline
        response = client.get("/api/v1/pipelines/nonexistent-pipeline")

        assert response.status_code == 404
        data = response.json()

        # Verify ErrorEnvelope structure
        assert "error" in data
        error = data["error"]

        # Check all required fields
        assert "code" in error
        assert "message" in error
        assert "timestamp" in error
        assert "hint" in error

        # Verify specific error details
        assert error["code"] == "PIPELINE_NOT_FOUND"
        assert "nonexistent-pipeline" in error["message"]

    def test_error_envelope_timestamp_format(self, client):
        """Test that ErrorEnvelope timestamps are in ISO format."""
        # Request non-existent job to get an error
        response = client.get("/api/v1/jobs/test-job-id")

        assert response.status_code == 404
        data = response.json()

        # Verify timestamp format
        timestamp = data["error"]["timestamp"]
        assert "T" in timestamp  # ISO format includes 'T' separator
        assert len(timestamp) > 0

        # Verify it's parseable as ISO datetime
        from datetime import datetime

        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        assert parsed is not None

    def test_error_envelope_consistency_across_endpoints(self, client):
        """Test that ErrorEnvelope format is consistent across different endpoints."""
        # Test multiple endpoints to ensure consistency
        endpoints_to_test = [
            ("/api/v1/jobs/nonexistent", 404),
            ("/api/v1/pipelines/nonexistent", 404),
        ]

        errors = []
        for endpoint, expected_status in endpoints_to_test:
            response = client.get(endpoint)
            assert response.status_code == expected_status

            data = response.json()
            assert "error" in data

            error = data["error"]
            errors.append(error)

            # Verify all have same structure
            assert "code" in error
            assert "message" in error
            assert "timestamp" in error

        # All errors should have the same core keys
        for error in errors[1:]:
            # Core keys should be present
            assert "code" in error
            assert "message" in error
            assert "timestamp" in error

    def test_helpful_hints_are_present(self, client):
        """Test that errors include helpful hints."""
        # Test job not found
        response = client.get("/api/v1/jobs/test-id")
        assert response.status_code == 404
        data = response.json()
        assert "hint" in data["error"]
        # Check for either the old or new hint format
        assert (
            "GET /api/v1/jobs" in data["error"]["hint"]
            or "videoannotator job" in data["error"]["hint"]
        )

        # Test pipeline not found
        response = client.get("/api/v1/pipelines/test-pipeline")
        assert response.status_code == 404
        data = response.json()
        assert "hint" in data["error"]
        assert "videoannotator pipelines" in data["error"]["hint"]
