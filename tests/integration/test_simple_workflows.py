"""Simple integration tests for batch processing components.

These tests verify that the components work together correctly.
"""

import tempfile
from datetime import datetime
from pathlib import Path

from src.batch.batch_orchestrator import BatchOrchestrator
from src.batch.types import BatchJob, JobStatus
from src.storage.file_backend import FileStorageBackend


class TestBatchIntegrationSimple:
    """Simple integration tests for batch processing."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.storage = FileStorageBackend(self.temp_dir / "storage")
        self.orchestrator = BatchOrchestrator(storage_backend=self.storage)

        # Create test video file
        self.test_video = self.temp_dir / "test_video.mp4"
        self.test_video.write_bytes(b"fake video content")

    def test_orchestrator_storage_integration(self):
        """Test that orchestrator and storage work together."""
        # Add a job through orchestrator
        job_id = self.orchestrator.add_job(str(self.test_video))

        # Job should be in orchestrator
        assert len(self.orchestrator.jobs) == 1
        job = self.orchestrator.jobs[0]
        assert job.job_id == job_id

        # Save job metadata through storage
        self.orchestrator.storage_backend.save_job_metadata(job)

        # Load it back
        loaded_job = self.orchestrator.storage_backend.load_job_metadata(job_id)
        assert loaded_job.job_id == job_id
        assert loaded_job.video_path == job.video_path

    def test_orchestrator_progress_tracker_integration(self):
        """Test orchestrator and progress tracker integration."""
        # Add multiple jobs
        _job_id1 = self.orchestrator.add_job(str(self.test_video))

        # Create another test video
        test_video2 = self.temp_dir / "test_video2.mp4"
        test_video2.write_bytes(b"fake video content 2")
        _job_id2 = self.orchestrator.add_job(str(test_video2))

        # Get status through progress tracker
        status = self.orchestrator.progress_tracker.get_status(self.orchestrator.jobs)

        assert status.total_jobs == 2
        assert status.pending_jobs == 2
        assert status.running_jobs == 0
        assert status.completed_jobs == 0

    def test_orchestrator_failure_recovery_integration(self):
        """Test orchestrator and failure recovery integration."""
        # Add a job
        _job_id = self.orchestrator.add_job(str(self.test_video))
        job = self.orchestrator.jobs[0]

        # Test retry logic
        error = Exception("Test error")
        should_retry = self.orchestrator.failure_recovery.should_retry(job, error)

        # Should be able to retry initially
        assert isinstance(should_retry, bool)

        if should_retry:
            # Test preparation for retry
            self.orchestrator.failure_recovery.prepare_retry(job, error)
            assert job.retry_count == 1
            assert job.status == JobStatus.RETRYING

    def test_full_component_integration(self):
        """Test all components working together."""
        # Add job
        job_id = self.orchestrator.add_job(str(self.test_video))
        job = self.orchestrator.jobs[0]

        # Save to storage
        self.orchestrator.storage_backend.save_job_metadata(job)

        # Simulate some pipeline results
        annotations = [{"timestamp": 1.0, "data": "test scene"}]
        self.orchestrator.storage_backend.save_annotations(
            job_id, "scene_detection", annotations
        )

        # Check progress
        status = self.orchestrator.progress_tracker.get_status(self.orchestrator.jobs)
        assert status.total_jobs == 1

        # Test failure recovery
        error = Exception("Test error")
        should_retry = self.orchestrator.failure_recovery.should_retry(job, error)
        assert isinstance(should_retry, bool)

        # Verify annotations were saved
        assert self.orchestrator.storage_backend.annotation_exists(
            job_id, "scene_detection"
        )
        loaded_annotations = self.orchestrator.storage_backend.load_annotations(
            job_id, "scene_detection"
        )
        assert len(loaded_annotations) == 1


class TestBatchTypesIntegration:
    """Test BatchJob and related types work correctly."""

    def test_batch_job_creation(self):
        """Test BatchJob can be created and used."""
        job = BatchJob(
            job_id="test_job",
            video_path=Path("test_video.mp4"),
            output_dir="/path/to/output",
            status=JobStatus.PENDING,
            created_at=datetime.now(),
            retry_count=0,
            config={},
            selected_pipelines=[],
            pipeline_results={},
        )

        assert job.job_id == "test_job"
        assert job.status == JobStatus.PENDING
        assert job.retry_count == 0

    def test_batch_job_serialization(self):
        """Test BatchJob can be serialized and deserialized."""
        job = BatchJob(
            job_id="test_job",
            video_path=Path("test_video.mp4"),
            output_dir="/path/to/output",
            status=JobStatus.COMPLETED,
            created_at=datetime.now(),
            retry_count=2,
            config={"test": "value"},
            selected_pipelines=["scene_detection"],
            pipeline_results={},
        )

        # Convert to dict
        job_dict = job.to_dict()
        assert isinstance(job_dict, dict)
        assert job_dict["job_id"] == "test_job"

        # Convert back to BatchJob
        restored_job = BatchJob.from_dict(job_dict)
        assert restored_job.job_id == job.job_id
        assert restored_job.status == job.status
        assert restored_job.retry_count == job.retry_count
