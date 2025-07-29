"""
Unit tests for BatchOrchestrator - testing actual implementation.

These tests focus on the real API and methods that exist in the BatchOrchestrator class.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from src.batch.batch_orchestrator import BatchOrchestrator
from src.batch.types import BatchJob, JobStatus
from src.storage.file_backend import FileStorageBackend


class TestBatchOrchestratorReal:
    """Test the actual BatchOrchestrator implementation."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.storage = FileStorageBackend(self.temp_dir / "storage")
        self.orchestrator = BatchOrchestrator(storage_backend=self.storage)
        
        # Create test video files
        self.test_video1 = self.temp_dir / "test1.mp4"
        self.test_video2 = self.temp_dir / "test2.mp4"
        self.test_video1.write_bytes(b"fake video content 1")
        self.test_video2.write_bytes(b"fake video content 2")

    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        orchestrator = BatchOrchestrator()
        
        # Check that required components exist
        assert orchestrator.storage_backend is not None
        assert orchestrator.progress_tracker is not None
        assert orchestrator.failure_recovery is not None
        assert orchestrator.jobs == []
        assert not orchestrator.is_running
        assert orchestrator.batch_id is None

    def test_orchestrator_custom_initialization(self):
        """Test orchestrator with custom storage backend."""
        storage = FileStorageBackend(Path("custom_storage"))
        orchestrator = BatchOrchestrator(
            storage_backend=storage,
            max_retries=5,
            checkpoint_interval=20
        )
        
        assert orchestrator.storage_backend == storage
        assert orchestrator.failure_recovery.max_retries == 5
        assert orchestrator.checkpoint_interval == 20

    def test_add_job_basic(self):
        """Test adding a basic job."""
        job_id = self.orchestrator.add_job(str(self.test_video1))
        
        assert job_id is not None
        assert len(self.orchestrator.jobs) == 1
        
        job = self.orchestrator.jobs[0]
        assert job.job_id == job_id
        assert job.video_path == self.test_video1
        assert job.status == JobStatus.PENDING

    def test_add_job_with_config(self):
        """Test adding job with custom configuration."""
        config = {"scene_detection": {"threshold": 0.5}}
        output_dir = self.temp_dir / "custom_output"
        
        job_id = self.orchestrator.add_job(
            str(self.test_video1),
            output_dir=str(output_dir),
            config=config,
            selected_pipelines=["scene_detection", "person_tracking"]
        )
        
        job = self.orchestrator.jobs[0]
        assert job.config == config
        assert job.output_dir == output_dir
        assert job.selected_pipelines == ["scene_detection", "person_tracking"]

    def test_add_job_nonexistent_file(self):
        """Test adding job with non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            self.orchestrator.add_job("nonexistent_video.mp4")

    def test_add_multiple_jobs(self):
        """Test adding multiple jobs."""
        job_id1 = self.orchestrator.add_job(str(self.test_video1))
        job_id2 = self.orchestrator.add_job(str(self.test_video2))
        
        assert len(self.orchestrator.jobs) == 2
        assert job_id1 != job_id2
        
        job_ids = {job.job_id for job in self.orchestrator.jobs}
        assert job_ids == {job_id1, job_id2}

    def test_add_jobs_from_directory(self):
        """Test adding jobs from a directory of videos."""
        # Create a directory with video files
        video_dir = self.temp_dir / "videos"
        video_dir.mkdir()
        
        video1 = video_dir / "video1.mp4"
        video2 = video_dir / "video2.avi"
        video3 = video_dir / "not_video.txt"  # Should be ignored
        
        video1.write_bytes(b"video content 1")
        video2.write_bytes(b"video content 2")
        video3.write_bytes(b"text content")
        
        job_ids = self.orchestrator.add_jobs_from_directory(str(video_dir))
        
        assert len(job_ids) == 2  # Only video files should be added
        assert len(self.orchestrator.jobs) == 2

    def test_pipeline_classes_imported(self):
        """Test that pipeline classes are imported correctly."""
        assert hasattr(self.orchestrator, 'pipeline_classes')
        assert isinstance(self.orchestrator.pipeline_classes, dict)
        assert len(self.orchestrator.pipeline_classes) > 0
        
        # Check for expected pipeline names
        pipeline_names = list(self.orchestrator.pipeline_classes.keys())
        expected_pipelines = ['scene_detection', 'person_tracking', 'face_analysis', 'audio_processing']
        
        # At least some of these should exist
        found_pipelines = [p for p in expected_pipelines if any(p in name for name in pipeline_names)]
        assert len(found_pipelines) > 0

    def test_job_creation_properties(self):
        """Test that jobs are created with correct properties."""
        job_id = self.orchestrator.add_job(str(self.test_video1))
        job = self.orchestrator.jobs[0]
        
        # Test basic properties
        assert job.job_id == job_id
        assert job.video_path == self.test_video1
        assert job.status == JobStatus.PENDING
        assert job.retry_count == 0
        assert job.created_at is not None
        assert job.started_at is None
        assert job.completed_at is None
        assert job.error_message is None
        assert job.pipeline_results == {}

    def test_job_video_id_generation(self):
        """Test that video_id is generated correctly from path."""
        job_id = self.orchestrator.add_job(str(self.test_video1))
        job = self.orchestrator.jobs[0]
        
        expected_video_id = self.test_video1.stem  # "test1"
        assert job.video_id == expected_video_id

    def test_progress_tracker_integration(self):
        """Test integration with progress tracker."""
        # Add some jobs
        self.orchestrator.add_job(str(self.test_video1))
        self.orchestrator.add_job(str(self.test_video2))
        
        # Get status through progress tracker
        status = self.orchestrator.progress_tracker.get_status(self.orchestrator.jobs)
        
        assert status.total_jobs == 2
        assert status.pending_jobs == 2
        assert status.running_jobs == 0
        assert status.completed_jobs == 0
        assert status.failed_jobs == 0

    def test_failure_recovery_integration(self):
        """Test integration with failure recovery."""
        job_id = self.orchestrator.add_job(str(self.test_video1))
        job = self.orchestrator.jobs[0]
        
        # Test retry logic
        error = Exception("Test error")
        should_retry = self.orchestrator.failure_recovery.should_retry(job, error)
        
        # Should be able to retry initially
        assert isinstance(should_retry, bool)
        
        # Test retry delay calculation
        delay = self.orchestrator.failure_recovery.calculate_retry_delay(job)
        assert isinstance(delay, (int, float))
        assert delay >= 0

    def test_storage_backend_integration(self):
        """Test integration with storage backend."""
        job_id = self.orchestrator.add_job(str(self.test_video1))
        job = self.orchestrator.jobs[0]
        
        # Test that storage backend can save job metadata
        self.orchestrator.storage_backend.save_job_metadata(job)
        
        # Test that we can load it back
        loaded_job = self.orchestrator.storage_backend.load_job_metadata(job_id)
        assert loaded_job.job_id == job.job_id
        assert loaded_job.video_path == job.video_path

    def test_state_management(self):
        """Test orchestrator state management."""
        # Initially not running
        assert not self.orchestrator.is_running
        assert not self.orchestrator.should_stop
        
        # Test state changes
        self.orchestrator.is_running = True
        assert self.orchestrator.is_running
        
        self.orchestrator.should_stop = True
        assert self.orchestrator.should_stop


class TestBatchOrchestratorEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.orchestrator = BatchOrchestrator()

    def test_add_job_empty_path(self):
        """Test adding job with empty path."""
        with pytest.raises(FileNotFoundError):
            self.orchestrator.add_job(video_path="")

    def test_add_job_relative_path(self):
        """Test adding job with relative path to non-existent file."""
        with pytest.raises(FileNotFoundError):
            self.orchestrator.add_job("relative/path/video.mp4")

    def test_add_jobs_from_nonexistent_directory(self):
        """Test adding jobs from non-existent directory."""
        with pytest.raises(FileNotFoundError):
            self.orchestrator.add_jobs_from_directory("nonexistent_dir")

    def test_add_jobs_from_empty_directory(self):
        """Test adding jobs from empty directory."""
        empty_dir = self.temp_dir / "empty"
        empty_dir.mkdir()
        
        job_ids = self.orchestrator.add_jobs_from_directory(str(empty_dir))
        assert job_ids == []


class TestBatchOrchestratorRealistic:
    """Test realistic scenarios and workflows."""

    def setup_method(self):
        """Set up realistic test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.storage = FileStorageBackend(self.temp_dir / "batch_storage")
        self.orchestrator = BatchOrchestrator(
            storage_backend=self.storage,
            max_retries=3,
            checkpoint_interval=5
        )
        
        # Create realistic test videos
        self.videos = []
        for i in range(3):
            video_path = self.temp_dir / f"sample_video_{i}.mp4"
            video_path.write_bytes(f"Mock video content {i}".encode())
            self.videos.append(video_path)

    def test_realistic_batch_setup(self):
        """Test setting up a realistic batch processing scenario."""
        # Add jobs with different configurations
        configs = [
            {"fast_mode": True, "scene_detection": {"threshold": 0.3}},
            {"high_quality": True, "face_analysis": {"model": "large"}},
            {"balanced": True}
        ]
        
        job_ids = []
        for i, video in enumerate(self.videos):
            job_id = self.orchestrator.add_job(
                str(video),
                config=configs[i],
                selected_pipelines=["scene_detection", "person_tracking"]
            )
            job_ids.append(job_id)
        
        # Verify all jobs were added correctly
        assert len(self.orchestrator.jobs) == 3
        assert len(job_ids) == 3
        assert all(isinstance(job_id, str) for job_id in job_ids)
        
        # Verify each job has correct configuration
        for i, job in enumerate(self.orchestrator.jobs):
            assert job.config == configs[i]
            assert job.selected_pipelines == ["scene_detection", "person_tracking"]

    def test_realistic_status_tracking(self):
        """Test realistic status tracking scenario."""
        # Add several jobs
        job_ids = [self.orchestrator.add_job(str(video)) for video in self.videos]
        
        # Simulate different job states
        jobs = {job.job_id: job for job in self.orchestrator.jobs}
        jobs[job_ids[0]].status = JobStatus.COMPLETED
        jobs[job_ids[1]].status = JobStatus.RUNNING
        jobs[job_ids[2]].status = JobStatus.FAILED
        
        # Check status
        status = self.orchestrator.progress_tracker.get_status(self.orchestrator.jobs)
        
        assert status.total_jobs == 3
        assert status.completed_jobs == 1
        assert status.running_jobs == 1
        assert status.failed_jobs == 1
        assert status.pending_jobs == 0
        
        # Test success rate calculation
        expected_success_rate = (1 / 2) * 100  # 1 completed out of 2 finished jobs
        assert status.success_rate == expected_success_rate

    def test_realistic_failure_scenario(self):
        """Test realistic failure and retry scenario."""
        job_id = self.orchestrator.add_job(str(self.videos[0]))
        job = next(job for job in self.orchestrator.jobs if job.job_id == job_id)
        
        # Simulate failures and retries
        errors = [
            Exception("Network timeout"),
            Exception("Out of memory"), 
            Exception("Corrupted video file")
        ]
        
        for i, error in enumerate(errors):
            should_retry = self.orchestrator.failure_recovery.should_retry(job, error)
            
            if should_retry:
                job.retry_count += 1
                job.error_message = str(error)
                delay = self.orchestrator.failure_recovery.calculate_retry_delay(job)
                assert delay > 0  # Should have some delay
            else:
                break
        
        # Should eventually stop retrying
        assert job.retry_count <= self.orchestrator.failure_recovery.max_retries
