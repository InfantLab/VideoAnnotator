"""
Test file to validate the actual batch processing APIs.
Run manually with: python -m pytest tests/test_batch_validation.py -v
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from src.batch.types import BatchJob, JobStatus, BatchStatus, PipelineResult
from src.batch.batch_orchestrator import BatchOrchestrator
from src.batch.progress_tracker import ProgressTracker
from src.batch.recovery import FailureRecovery, RetryStrategy
from src.storage.file_backend import FileStorageBackend


class TestBatchTypesValidation:
    """Validate BatchJob and related types work correctly."""

    def test_batch_job_default_creation(self):
        """Test BatchJob creation with defaults."""
        job = BatchJob()
        
        # Should have auto-generated ID
        assert job.job_id is not None
        assert len(job.job_id) > 0
        
        # Should have default values
        assert job.status == JobStatus.PENDING
        assert job.retry_count == 0
        assert job.config == {}
        assert job.pipeline_results == {}
        assert isinstance(job.created_at, datetime)

    def test_batch_job_full_creation(self):
        """Test BatchJob creation with all parameters."""
        job = BatchJob(
            job_id="test_job_001",
            video_path=Path("/videos/test.mp4"),
            output_dir=Path("/output/test"),
            config={"scene_detection": {"threshold": 0.5}},
            status=JobStatus.RUNNING,
            selected_pipelines=["scene_detection", "face_analysis"],
            retry_count=1
        )
        
        assert job.job_id == "test_job_001"
        assert job.video_path == Path("/videos/test.mp4")
        assert job.status == JobStatus.RUNNING
        assert job.retry_count == 1
        assert job.config["scene_detection"]["threshold"] == 0.5

    def test_batch_job_properties(self):
        """Test BatchJob computed properties."""
        job = BatchJob(video_path=Path("/videos/sample_video.mp4"))
        
        # video_id should be filename without extension
        assert job.video_id == "sample_video"
        
        # is_complete should be False for PENDING
        assert not job.is_complete
        
        # Test with completed status
        job.status = JobStatus.COMPLETED
        assert job.is_complete

    def test_batch_job_serialization(self):
        """Test BatchJob to_dict and from_dict."""
        original_job = BatchJob(
            job_id="serial_test",
            video_path=Path("/test/video.mp4"),
            output_dir=Path("/test/output"),
            status=JobStatus.COMPLETED,
            config={"test": "value"},
            retry_count=2
        )
        
        # Convert to dict
        job_dict = original_job.to_dict()
        assert isinstance(job_dict, dict)
        assert job_dict["job_id"] == "serial_test"
        assert job_dict["status"] == "completed"
        
        # Convert back to BatchJob
        restored_job = BatchJob.from_dict(job_dict)
        assert restored_job.job_id == original_job.job_id
        assert restored_job.status == original_job.status
        assert restored_job.retry_count == original_job.retry_count

    def test_pipeline_result(self):
        """Test PipelineResult creation and properties."""
        result = PipelineResult(
            pipeline_name="scene_detection",
            status=JobStatus.COMPLETED,
            start_time=datetime(2024, 1, 1, 10, 0, 0),
            end_time=datetime(2024, 1, 1, 10, 0, 30),
            annotation_count=5
        )
        
        assert result.pipeline_name == "scene_detection"
        assert result.status == JobStatus.COMPLETED
        assert result.annotation_count == 5
        assert result.duration == 30.0  # 30 seconds


class TestBatchOrchestratorValidation:
    """Validate BatchOrchestrator functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_video = self.temp_dir / "test_video.mp4"
        self.test_video.write_bytes(b"fake video content")

    def test_orchestrator_creation(self):
        """Test BatchOrchestrator creation."""
        orchestrator = BatchOrchestrator()
        
        # Should have required components
        assert orchestrator.storage_backend is not None
        assert orchestrator.progress_tracker is not None
        assert orchestrator.failure_recovery is not None
        assert orchestrator.jobs == []
        assert not orchestrator.is_running

    def test_orchestrator_add_job_basic(self):
        """Test adding a job with minimal parameters."""
        orchestrator = BatchOrchestrator()
        
        job_id = orchestrator.add_job(str(self.test_video))
        
        assert job_id is not None
        assert len(orchestrator.jobs) == 1
        
        job = orchestrator.jobs[0]
        assert job.job_id == job_id
        assert job.video_path == self.test_video
        assert job.status == JobStatus.PENDING

    def test_orchestrator_add_job_with_config(self):
        """Test adding a job with configuration."""
        orchestrator = BatchOrchestrator()
        
        config = {"scene_detection": {"threshold": 0.7}}
        output_dir = self.temp_dir / "custom_output"
        
        job_id = orchestrator.add_job(
            str(self.test_video),
            output_dir=str(output_dir),
            config=config,
            selected_pipelines=["scene_detection", "person_tracking"]
        )
        
        job = orchestrator.jobs[0]
        assert job.config == config
        assert job.output_dir == output_dir
        assert job.selected_pipelines == ["scene_detection", "person_tracking"]

    def test_orchestrator_nonexistent_file(self):
        """Test that adding non-existent file raises error."""
        orchestrator = BatchOrchestrator()
        
        with pytest.raises(FileNotFoundError):
            orchestrator.add_job("nonexistent_video.mp4")

    def test_orchestrator_add_multiple_jobs(self):
        """Test adding multiple jobs."""
        orchestrator = BatchOrchestrator()
        
        # Create second video
        video2 = self.temp_dir / "video2.mp4"
        video2.write_bytes(b"fake content 2")
        
        job_id1 = orchestrator.add_job(str(self.test_video))
        job_id2 = orchestrator.add_job(str(video2))
        
        assert len(orchestrator.jobs) == 2
        assert job_id1 != job_id2


class TestProgressTrackerValidation:
    """Validate ProgressTracker functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.tracker = ProgressTracker()

    def create_test_job(self, job_id="test", status=JobStatus.PENDING):
        """Helper to create test job."""
        return BatchJob(
            job_id=job_id,
            video_path=Path(f"/videos/{job_id}.mp4"),
            output_dir=Path(f"/output/{job_id}"),
            status=status
        )

    def test_progress_tracker_empty_jobs(self):
        """Test get_status with empty job list."""
        status = self.tracker.get_status([])
        
        assert isinstance(status, BatchStatus)
        assert status.total_jobs == 0
        assert status.pending_jobs == 0
        assert status.running_jobs == 0
        assert status.completed_jobs == 0
        assert status.failed_jobs == 0

    def test_progress_tracker_mixed_statuses(self):
        """Test get_status with jobs in different states."""
        jobs = [
            self.create_test_job("job1", JobStatus.COMPLETED),
            self.create_test_job("job2", JobStatus.RUNNING),
            self.create_test_job("job3", JobStatus.PENDING),
            self.create_test_job("job4", JobStatus.FAILED),
            self.create_test_job("job5", JobStatus.COMPLETED)
        ]
        
        status = self.tracker.get_status(jobs)
        
        assert status.total_jobs == 5
        assert status.completed_jobs == 2
        assert status.running_jobs == 1
        assert status.pending_jobs == 1
        assert status.failed_jobs == 1


class TestFailureRecoveryValidation:
    """Validate FailureRecovery functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.recovery = FailureRecovery(max_retries=3)

    def create_test_job(self, retry_count=0):
        """Helper to create test job."""
        return BatchJob(
            job_id="test_job",
            video_path=Path("/videos/test.mp4"),
            output_dir=Path("/output/test"),
            retry_count=retry_count
        )

    def test_failure_recovery_creation(self):
        """Test FailureRecovery creation."""
        recovery = FailureRecovery()
        
        assert recovery.max_retries == 3
        assert recovery.base_delay == 1.0
        assert recovery.strategy == RetryStrategy.EXPONENTIAL_BACKOFF

    def test_should_retry_fresh_job(self):
        """Test should_retry with new job."""
        job = self.create_test_job(retry_count=0)
        error = Exception("Test error")
        
        should_retry = self.recovery.should_retry(job, error)
        
        assert isinstance(should_retry, bool)
        # Should be True for fresh job with retryable error
        assert should_retry is True

    def test_should_retry_max_retries(self):
        """Test should_retry when max retries exceeded."""
        job = self.create_test_job(retry_count=3)  # At max retries
        error = Exception("Test error")
        
        should_retry = self.recovery.should_retry(job, error)
        
        assert should_retry is False

    def test_calculate_retry_delay(self):
        """Test retry delay calculation."""
        job = self.create_test_job(retry_count=1)
        
        delay = self.recovery.calculate_retry_delay(job)
        
        assert isinstance(delay, (int, float))
        assert delay >= 0

    def test_prepare_retry(self):
        """Test preparing job for retry."""
        job = self.create_test_job(retry_count=0)
        error = Exception("Test error")
        
        updated_job = self.recovery.prepare_retry(job, error)
        
        assert updated_job.retry_count == 1
        assert updated_job.status == JobStatus.RETRYING
        assert updated_job.error_message == "Test error"


class TestFileStorageBackendValidation:
    """Validate FileStorageBackend functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.storage = FileStorageBackend(self.temp_dir)

    def create_test_job(self, job_id="test_job"):
        """Helper to create test job."""
        return BatchJob(
            job_id=job_id,
            video_path=Path("/videos/test.mp4"),
            output_dir=Path("/output/test"),
            status=JobStatus.PENDING
        )

    def test_storage_creation(self):
        """Test FileStorageBackend creation."""
        storage = FileStorageBackend(Path("test_storage"))
        
        assert storage.base_dir == Path("test_storage")
        assert storage.jobs_dir == Path("test_storage") / "jobs"

    def test_save_and_load_job_metadata(self):
        """Test saving and loading job metadata."""
        job = self.create_test_job("storage_test")
        
        # Save metadata
        self.storage.save_job_metadata(job)
        
        # Load metadata back
        loaded_job = self.storage.load_job_metadata("storage_test")
        
        assert loaded_job.job_id == job.job_id
        assert loaded_job.video_path == job.video_path
        assert loaded_job.status == job.status

    def test_load_nonexistent_metadata(self):
        """Test loading non-existent job metadata."""
        with pytest.raises(FileNotFoundError):
            self.storage.load_job_metadata("nonexistent_job")

    def test_save_and_load_annotations(self):
        """Test saving and loading annotations."""
        annotations = [
            {"timestamp": 1.0, "data": "scene1"},
            {"timestamp": 2.0, "data": "scene2"}
        ]
        
        # Save annotations
        result_path = self.storage.save_annotations("test_job", "scene_detection", annotations)
        assert isinstance(result_path, str)
        
        # Load annotations back
        loaded_annotations = self.storage.load_annotations("test_job", "scene_detection")
        assert len(loaded_annotations) == 2
        assert loaded_annotations[0]["data"] == "scene1"

    def test_annotation_exists(self):
        """Test checking annotation existence."""
        # Should not exist initially
        assert not self.storage.annotation_exists("test_job", "face_analysis")
        
        # Save annotations
        annotations = [{"timestamp": 1.0, "face": "person1"}]
        self.storage.save_annotations("test_job", "face_analysis", annotations)
        
        # Should exist now
        assert self.storage.annotation_exists("test_job", "face_analysis")

    def test_list_jobs(self):
        """Test listing jobs."""
        # Should be empty initially
        job_ids = self.storage.list_jobs()
        assert job_ids == []
        
        # Save some jobs
        job1 = self.create_test_job("job1")
        job2 = self.create_test_job("job2")
        
        self.storage.save_job_metadata(job1)
        self.storage.save_job_metadata(job2)
        
        # Should list both jobs
        job_ids = self.storage.list_jobs()
        assert len(job_ids) == 2
        assert "job1" in job_ids
        assert "job2" in job_ids


class TestIntegrationValidation:
    """Validate integration between components."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.storage = FileStorageBackend(self.temp_dir / "storage")
        self.orchestrator = BatchOrchestrator(storage_backend=self.storage)
        
        # Create test video
        self.test_video = self.temp_dir / "integration_test.mp4"
        self.test_video.write_bytes(b"fake video for integration test")

    def test_full_integration(self):
        """Test all components working together."""
        # Add job through orchestrator
        job_id = self.orchestrator.add_job(str(self.test_video))
        
        # Verify job was added
        assert len(self.orchestrator.jobs) == 1
        job = self.orchestrator.jobs[0]
        
        # Save through storage backend
        self.orchestrator.storage_backend.save_job_metadata(job)
        
        # Load back through storage
        loaded_job = self.orchestrator.storage_backend.load_job_metadata(job_id)
        assert loaded_job.job_id == job_id
        
        # Test progress tracking
        status = self.orchestrator.progress_tracker.get_status(self.orchestrator.jobs)
        assert status.total_jobs == 1
        assert status.pending_jobs == 1
        
        # Test failure recovery
        error = Exception("Integration test error")
        should_retry = self.orchestrator.failure_recovery.should_retry(job, error)
        assert isinstance(should_retry, bool)
        
        print(f"✅ Full integration test passed for job: {job_id}")
