"""Integration tests for storage path functionality (v1.3.0).

Tests verify that:
1. Job creation populates storage_path field
2. Storage directories are created when needed
3. Artifacts can be stored in job-specific directories
4. Storage paths persist through database round-trips
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from batch.types import BatchJob, JobStatus
from storage.config import ensure_job_storage_path, get_job_storage_path
from storage.sqlite_backend import SQLiteStorageBackend


class TestStoragePathsIntegration:
    """Integration tests for storage path functionality."""

    def setup_method(self):
        """Set up test environment with temporary storage."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.db_path = self.temp_dir / "test_jobs.db"
        self.storage = SQLiteStorageBackend(self.db_path)

    def teardown_method(self):
        """Clean up test environment."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_job_creation_populates_storage_path(self):
        """Test that creating a job sets storage_path field."""
        # Create a job with storage_path
        job = BatchJob(
            video_path=Path("/test/video.mp4"),
            status=JobStatus.PENDING,
            config={},
        )

        # Set storage path as would be done in API
        job.storage_path = get_job_storage_path(job.job_id)

        # Verify storage_path is populated
        assert job.storage_path is not None
        assert isinstance(job.storage_path, Path)
        assert job.job_id in str(job.storage_path)

    def test_storage_path_persists_to_database(self):
        """Test that storage_path is saved to and loaded from database."""
        # Create job with storage_path
        job = BatchJob(
            video_path=Path("/test/video.mp4"),
            status=JobStatus.PENDING,
            config={"test": "data"},
        )
        job.storage_path = get_job_storage_path(job.job_id)

        # Save to database
        self.storage.save_job_metadata(job)

        # Load from database
        loaded_job = self.storage.load_job_metadata(job.job_id)

        # Verify storage_path persisted
        assert loaded_job.storage_path is not None
        assert loaded_job.storage_path == job.storage_path
        assert isinstance(loaded_job.storage_path, Path)

    def test_ensure_creates_job_storage_directory(self):
        """Test that ensure_job_storage_path creates directories."""
        import uuid

        # Use unique job ID to avoid conflicts with other tests
        job_id = f"test-job-{uuid.uuid4()}"

        # Get path
        job_path = get_job_storage_path(job_id)

        # Ensure path creates directory
        ensured_path = ensure_job_storage_path(job_id)
        assert ensured_path.exists()
        assert ensured_path.is_dir()
        assert ensured_path == job_path

        # Verify idempotency - calling again should work fine
        ensured_again = ensure_job_storage_path(job_id)
        assert ensured_again == ensured_path
        assert ensured_again.exists()

    def test_artifacts_can_be_stored_in_job_directory(self):
        """Test that artifacts can be written to job storage directory."""
        job_id = "artifact-test-job"
        job_path = ensure_job_storage_path(job_id)

        # Write a test artifact
        artifact_file = job_path / "test_output.json"
        artifact_file.write_text('{"result": "success"}')

        # Verify artifact exists
        assert artifact_file.exists()
        assert artifact_file.read_text() == '{"result": "success"}'

    def test_multiple_jobs_have_isolated_storage(self):
        """Test that different jobs have separate storage directories."""
        job1 = BatchJob(video_path=Path("/test/video1.mp4"), status=JobStatus.PENDING)
        job2 = BatchJob(video_path=Path("/test/video2.mp4"), status=JobStatus.PENDING)

        job1.storage_path = get_job_storage_path(job1.job_id)
        job2.storage_path = get_job_storage_path(job2.job_id)

        # Paths should be different
        assert job1.storage_path != job2.storage_path

        # Create directories
        path1 = ensure_job_storage_path(job1.job_id)
        path2 = ensure_job_storage_path(job2.job_id)

        # Write different artifacts
        (path1 / "data.txt").write_text("job1 data")
        (path2 / "data.txt").write_text("job2 data")

        # Verify isolation
        assert (path1 / "data.txt").read_text() == "job1 data"
        assert (path2 / "data.txt").read_text() == "job2 data"

    def test_storage_path_updated_on_job_update(self):
        """Test that storage_path can be updated when job is updated."""
        # Create and save job without storage_path
        job = BatchJob(
            video_path=Path("/test/video.mp4"),
            status=JobStatus.PENDING,
            config={},
        )
        self.storage.save_job_metadata(job)

        # Update job with storage_path
        job.storage_path = get_job_storage_path(job.job_id)
        job.status = JobStatus.RUNNING
        self.storage.save_job_metadata(job)

        # Load and verify storage_path was updated
        loaded_job = self.storage.load_job_metadata(job.job_id)
        assert loaded_job.storage_path is not None
        assert loaded_job.storage_path == job.storage_path
        assert loaded_job.status == JobStatus.RUNNING

    def test_storage_path_with_real_workflow(self):
        """Test complete workflow: create job, set storage, process, persist."""
        # Simulate API job creation
        job = BatchJob(
            video_path=Path("/videos/test.mp4"),
            status=JobStatus.PENDING,
            config={"pipeline": "face_detection"},
            selected_pipelines=["face_detection"],
        )

        # Step 1: Set storage path (as API would do)
        job.storage_path = get_job_storage_path(job.job_id)

        # Step 2: Save to database
        self.storage.save_job_metadata(job)

        # Step 3: Worker picks up job (simulated by loading)
        worker_job = self.storage.load_job_metadata(job.job_id)
        assert worker_job.storage_path is not None

        # Step 4: Create working directory
        work_dir = ensure_job_storage_path(worker_job.job_id)
        assert work_dir.exists()

        # Step 5: Write processing results
        result_file = work_dir / "face_detections.json"
        result_file.write_text('{"faces": []}')

        # Step 6: Update job status
        worker_job.status = JobStatus.COMPLETED
        self.storage.save_job_metadata(worker_job)

        # Step 7: Verify final state
        final_job = self.storage.load_job_metadata(job.job_id)
        assert final_job.status == JobStatus.COMPLETED
        assert final_job.storage_path is not None
        assert (Path(final_job.storage_path) / "face_detections.json").exists()

    def test_storage_path_consistency_across_operations(self):
        """Test that storage_path remains consistent through multiple operations."""
        job = BatchJob(
            video_path=Path("/test/video.mp4"),
            status=JobStatus.PENDING,
            config={},
        )

        # Set initial storage path
        original_path = get_job_storage_path(job.job_id)
        job.storage_path = original_path

        # Save, load, update multiple times
        self.storage.save_job_metadata(job)
        loaded1 = self.storage.load_job_metadata(job.job_id)
        assert loaded1.storage_path == original_path

        loaded1.status = JobStatus.RUNNING
        self.storage.save_job_metadata(loaded1)
        loaded2 = self.storage.load_job_metadata(job.job_id)
        assert loaded2.storage_path == original_path

        loaded2.status = JobStatus.COMPLETED
        self.storage.save_job_metadata(loaded2)
        loaded3 = self.storage.load_job_metadata(job.job_id)
        assert loaded3.storage_path == original_path

    def test_idempotent_directory_creation(self):
        """Test that ensure_job_storage_path is idempotent."""
        job_id = "idempotent-test-job"

        # Create directory multiple times
        path1 = ensure_job_storage_path(job_id)
        path2 = ensure_job_storage_path(job_id)
        path3 = ensure_job_storage_path(job_id)

        # All should return same path
        assert path1 == path2 == path3

        # Directory should exist and be usable
        assert path1.exists()
        test_file = path1 / "test.txt"
        test_file.write_text("test")
        assert test_file.exists()

    def test_storage_path_serialization_in_batch_job(self):
        """Test that BatchJob correctly serializes/deserializes storage_path."""
        job = BatchJob(
            video_path=Path("/test/video.mp4"),
            status=JobStatus.PENDING,
            config={},
        )
        job.storage_path = get_job_storage_path(job.job_id)

        # Serialize to dict
        job_dict = job.to_dict()
        assert "storage_path" in job_dict
        assert job_dict["storage_path"] is not None
        assert isinstance(job_dict["storage_path"], str)

        # Deserialize from dict
        restored_job = BatchJob.from_dict(job_dict)
        assert restored_job.storage_path is not None
        assert isinstance(restored_job.storage_path, Path)
        assert restored_job.storage_path == job.storage_path
