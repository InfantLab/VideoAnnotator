"""Integration tests for job cancellation API endpoint.

Tests the POST /api/v1/jobs/{id}/cancel endpoint with various job states.
"""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from videoannotator.api.database import reset_storage_backend
from videoannotator.api.main import app
from videoannotator.batch.types import JobStatus

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_db():
    """Reset database before each test."""
    reset_storage_backend()
    yield
    reset_storage_backend()


class TestJobCancellationAPI:
    """Integration tests for job cancellation endpoint."""

    def test_cancel_pending_job(self, tmp_path):
        """Test cancelling a job that hasn't started yet."""
        # Create a test video file
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video content")

        # Submit a job
        with open(video_file, "rb") as f:
            response = client.post(
                "/api/v1/jobs/",
                files={"video": ("test.mp4", f, "video/mp4")},
                data={"config": "{}"},
            )

        assert response.status_code == 201
        job_id = response.json()["id"]

        # Cancel the job
        cancel_response = client.post(f"/api/v1/jobs/{job_id}/cancel")

        assert cancel_response.status_code == 200
        data = cancel_response.json()
        assert data["id"] == job_id
        assert data["status"] == JobStatus.CANCELLED.value
        assert data["error_message"] == "Job cancelled by user request"

        # Verify job status is updated in database
        status_response = client.get(f"/api/v1/jobs/{job_id}")
        assert status_response.status_code == 200
        assert status_response.json()["status"] == JobStatus.CANCELLED.value

    def test_cancel_already_cancelled_job_idempotent(self, tmp_path):
        """Test cancelling an already cancelled job is idempotent."""
        # Create and submit job
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video content")

        with open(video_file, "rb") as f:
            response = client.post(
                "/api/v1/jobs/",
                files={"video": ("test.mp4", f, "video/mp4")},
                data={"config": "{}"},
            )

        job_id = response.json()["id"]

        # Cancel the job twice
        first_cancel = client.post(f"/api/v1/jobs/{job_id}/cancel")
        second_cancel = client.post(f"/api/v1/jobs/{job_id}/cancel")

        # Both should succeed
        assert first_cancel.status_code == 200
        assert second_cancel.status_code == 200

        # Both should return the same CANCELLED status
        assert first_cancel.json()["status"] == JobStatus.CANCELLED.value
        assert second_cancel.json()["status"] == JobStatus.CANCELLED.value

    def test_cancel_nonexistent_job_returns_404(self):
        """Test cancelling a non-existent job returns 404."""
        fake_job_id = "00000000-0000-0000-0000-000000000000"
        response = client.post(f"/api/v1/jobs/{fake_job_id}/cancel")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_cannot_cancel_completed_job(self, tmp_path, monkeypatch):
        """Test that completed jobs cannot be cancelled."""
        # Create and submit job
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video content")

        with open(video_file, "rb") as f:
            response = client.post(
                "/api/v1/jobs/",
                files={"video": ("test.mp4", f, "video/mp4")},
                data={"config": "{}"},
            )

        job_id = response.json()["id"]

        # Manually mark job as completed in database
        from api.database import get_storage_backend

        storage = get_storage_backend()
        job = storage.load_job_metadata(job_id)
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now()
        storage.save_job_metadata(job)

        # Try to cancel
        response = client.post(f"/api/v1/jobs/{job_id}/cancel")

        # Should return 409 Conflict
        assert response.status_code == 409
        assert "already complete" in response.json()["detail"].lower()

    def test_cannot_cancel_failed_job(self, tmp_path):
        """Test that failed jobs cannot be cancelled."""
        # Create and submit job
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video content")

        with open(video_file, "rb") as f:
            response = client.post(
                "/api/v1/jobs/",
                files={"video": ("test.mp4", f, "video/mp4")},
                data={"config": "{}"},
            )

        job_id = response.json()["id"]

        # Manually mark job as failed in database
        from api.database import get_storage_backend

        storage = get_storage_backend()
        job = storage.load_job_metadata(job_id)
        job.status = JobStatus.FAILED
        job.error_message = "Processing failed"
        job.completed_at = datetime.now()
        storage.save_job_metadata(job)

        # Try to cancel
        response = client.post(f"/api/v1/jobs/{job_id}/cancel")

        # Should return 409 Conflict
        assert response.status_code == 409
        assert "already complete" in response.json()["detail"].lower()

    def test_cancel_running_job(self, tmp_path):
        """Test cancelling a job that is currently running."""
        # Create and submit job
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video content")

        with open(video_file, "rb") as f:
            response = client.post(
                "/api/v1/jobs/",
                files={"video": ("test.mp4", f, "video/mp4")},
                data={"config": "{}"},
            )

        job_id = response.json()["id"]

        # Manually mark job as running in database
        from api.database import get_storage_backend

        storage = get_storage_backend()
        job = storage.load_job_metadata(job_id)
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()
        storage.save_job_metadata(job)

        # Cancel the job
        response = client.post(f"/api/v1/jobs/{job_id}/cancel")

        # Should succeed
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == JobStatus.CANCELLED.value
        assert data["error_message"] == "Job cancelled by user request"

    def test_cancelled_job_has_completed_at_timestamp(self, tmp_path):
        """Test that cancelled jobs have a completed_at timestamp set."""
        # Create and submit job
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video content")

        with open(video_file, "rb") as f:
            response = client.post(
                "/api/v1/jobs/",
                files={"video": ("test.mp4", f, "video/mp4")},
                data={"config": "{}"},
            )

        job_id = response.json()["id"]

        # Cancel the job
        response = client.post(f"/api/v1/jobs/{job_id}/cancel")

        assert response.status_code == 200
        data = response.json()

        # Should have a completed_at timestamp
        assert data["completed_at"] is not None

    def test_cancel_updates_error_message(self, tmp_path):
        """Test that cancellation sets an appropriate error message."""
        # Create and submit job
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video content")

        with open(video_file, "rb") as f:
            response = client.post(
                "/api/v1/jobs/",
                files={"video": ("test.mp4", f, "video/mp4")},
                data={"config": "{}"},
            )

        job_id = response.json()["id"]

        # Cancel the job
        response = client.post(f"/api/v1/jobs/{job_id}/cancel")

        assert response.status_code == 200
        data = response.json()

        # Should have cancellation message
        assert data["error_message"] == "Job cancelled by user request"

    def test_cancel_job_multiple_pipelines(self, tmp_path):
        """Test cancelling a job with multiple pipelines."""
        # Create and submit job with multiple pipelines
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video content")

        with open(video_file, "rb") as f:
            response = client.post(
                "/api/v1/jobs/",
                files={"video": ("test.mp4", f, "video/mp4")},
                data={
                    "config": "{}",
                    "selected_pipelines": '["laion_face_detection", "whisper_transcription"]',
                },
            )

        assert response.status_code == 201
        job_id = response.json()["id"]

        # Cancel the job
        response = client.post(f"/api/v1/jobs/{job_id}/cancel")

        assert response.status_code == 200
        assert response.json()["status"] == JobStatus.CANCELLED.value
