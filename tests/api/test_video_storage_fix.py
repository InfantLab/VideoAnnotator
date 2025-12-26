"""Test that video files are properly stored in persistent storage.

This test verifies the fix for the issue where videos uploaded via API
were stored in temporary directories that could be cleaned up before
background processing started, causing pipeline failures.
"""

import io
import os
import tempfile
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from videoannotator.api.database import (
    get_storage_backend,
    reset_storage_backend,
    set_database_path,
)
from videoannotator.api.main import create_app


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    prev_db_path = os.environ.get("VIDEOANNOTATOR_DB_PATH")
    prev_database_url = os.environ.get("DATABASE_URL")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = Path(f.name)

    set_database_path(db_path)
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"

    # Ensure caches pick up the per-test DB.
    reset_storage_backend()
    try:
        from videoannotator.storage.manager import get_storage_provider

        get_storage_provider.cache_clear()
    except Exception:
        pass

    yield db_path

    reset_storage_backend()
    try:
        from videoannotator.storage.manager import get_storage_provider

        get_storage_provider.cache_clear()
    except Exception:
        pass

    # Restore prior env to avoid cross-test contamination.
    if prev_db_path is None:
        os.environ.pop("VIDEOANNOTATOR_DB_PATH", None)
    else:
        os.environ["VIDEOANNOTATOR_DB_PATH"] = prev_db_path

    if prev_database_url is None:
        os.environ.pop("DATABASE_URL", None)
    else:
        os.environ["DATABASE_URL"] = prev_database_url

    reset_storage_backend()
    try:
        from videoannotator.storage.manager import get_storage_provider

        get_storage_provider.cache_clear()
    except Exception:
        pass
    time.sleep(0.1)
    try:
        if db_path.exists():
            db_path.unlink()
    except PermissionError:
        pass


@pytest.fixture
def client(temp_db):
    """Create a test client for the FastAPI app with temporary database."""
    app = create_app()
    with TestClient(app) as client:
        yield client


@pytest.fixture
def storage():
    """Get the storage backend."""
    return get_storage_backend()


@pytest.fixture
def sample_video_bytes() -> bytes:
    """Create sample video bytes for upload testing."""
    from tests.fixtures.synthetic_video import synthetic_video_bytes_avi

    data = synthetic_video_bytes_avi()
    assert len(data) > 0
    return data


class TestVideoPersistentStorage:
    """Test video file persistence in job storage."""

    def test_video_stored_in_persistent_location(
        self, client, sample_video_bytes, storage
    ):
        """Test that uploaded video is copied to persistent storage."""
        # Submit a job with a video file
        files = {
            "video": (
                "test_video.avi",
                io.BytesIO(sample_video_bytes),
                "video/avi",
            )
        }
        response = client.post("/api/v1/jobs/", files=files)

        assert response.status_code == 201
        job_data = response.json()
        job_id = job_data["id"]
        video_path = Path(job_data["video_path"])
        storage_path = Path(job_data["storage_path"])

        # Verify video_path is inside storage_path (persistent location)
        assert video_path.is_relative_to(storage_path), (
            f"Video path {video_path} should be inside storage path {storage_path}"
        )

        # Verify video file actually exists at the persistent location
        assert video_path.exists(), f"Video file should exist at {video_path}"

        # Verify the file is readable
        with open(video_path, "rb") as f:
            content = f.read()
            assert len(content) > 0, "Video file should not be empty"

        # Load job from database
        job = storage.load_job_metadata(job_id)
        assert job.video_path == video_path
        assert job.storage_path == storage_path

    def test_temp_directory_cleaned_up(self, client, storage):
        """Test that temporary upload directory is cleaned up after copying."""
        # Create a test video
        from tests.fixtures.synthetic_video import synthetic_video_bytes_avi

        video_file = io.BytesIO(synthetic_video_bytes_avi())

        files = {"video": ("cleanup_test.avi", video_file, "video/avi")}
        response = client.post("/api/v1/jobs/", files=files)

        assert response.status_code == 201
        job_data = response.json()

        # Video should be in persistent storage, not /tmp
        video_path = job_data["video_path"]
        assert not video_path.startswith("/tmp/tmp"), (
            f"Video should not be in temp directory: {video_path}"
        )

    def test_video_available_for_background_processing(
        self, client, sample_video_bytes, storage
    ):
        """Test that video file is accessible when background worker processes job."""
        # Submit a job
        files = {
            "video": (
                "worker_test.avi",
                io.BytesIO(sample_video_bytes),
                "video/avi",
            )
        }
        response = client.post("/api/v1/jobs/", files=files)

        assert response.status_code == 201
        job_data = response.json()
        job_id = job_data["id"]

        # Simulate what the background worker does:
        # Load job from database
        job = storage.load_job_metadata(job_id)

        # Verify video file exists (critical check from job_processor.py line 93)
        assert job.video_path.exists(), (
            f"Background worker should find video at {job.video_path}"
        )

        # Verify file is readable (critical for pipeline processing)
        with open(job.video_path, "rb") as f:
            content = f.read()
            assert len(content) > 0

    def test_multiple_jobs_isolated_storage(self, client, sample_video_bytes):
        """Test that each job gets its own isolated storage directory."""
        # Submit first job
        files1 = {"video": ("video1.avi", io.BytesIO(sample_video_bytes), "video/avi")}
        response1 = client.post("/api/v1/jobs/", files=files1)
        job1 = response1.json()

        # Submit second job with a fresh stream
        files2 = {"video": ("video2.avi", io.BytesIO(sample_video_bytes), "video/avi")}
        response2 = client.post("/api/v1/jobs/", files=files2)
        job2 = response2.json()

        # Both jobs should succeed
        assert response1.status_code == 201
        assert response2.status_code == 201

        # Storage paths should be different
        storage_path1 = Path(job1["storage_path"])
        storage_path2 = Path(job2["storage_path"])
        assert storage_path1 != storage_path2

        # Video paths should be different
        video_path1 = Path(job1["video_path"])
        video_path2 = Path(job2["video_path"])
        assert video_path1 != video_path2

        # Both files should exist
        assert video_path1.exists()
        assert video_path2.exists()
