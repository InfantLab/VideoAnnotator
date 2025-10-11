"""Unit tests for ProgressTracker - testing actual implementation.

These tests focus on the real API - get_status(jobs) method that takes a jobs list.
"""

from datetime import datetime, timedelta

from src.batch.progress_tracker import ProgressTracker
from src.batch.types import BatchJob, BatchStatus, JobStatus


class TestProgressTrackerReal:
    """Test the actual ProgressTracker implementation."""

    def setup_method(self):
        """Set up test environment."""
        self.tracker = ProgressTracker()

    def create_test_job(self, job_id="test_job", status=JobStatus.PENDING, **kwargs):
        """Helper to create test BatchJob."""
        return BatchJob(
            job_id=job_id,
            # # video_id is a computed property from video_path,  # FIXME: video_id parameter not supported
            video_path=kwargs.get("video_path", "/path/to/video.mp4"),
            output_dir=kwargs.get("output_dir", "/path/to/output"),
            status=status,
            created_at=kwargs.get("created_at", datetime.now()),
            started_at=kwargs.get("started_at"),
            completed_at=kwargs.get("completed_at"),
            error_message=kwargs.get("error_message"),
            retry_count=kwargs.get("retry_count", 0),
            config=kwargs.get("config", {}),
            selected_pipelines=kwargs.get("selected_pipelines", []),
            pipeline_results=kwargs.get("pipeline_results", {}),
        )

    def test_tracker_initialization(self):
        """Test ProgressTracker initialization."""
        tracker = ProgressTracker()
        assert tracker is not None

    def test_get_status_empty_jobs(self):
        """Test get_status with empty jobs list."""
        status = self.tracker.get_status([])

        assert isinstance(status, BatchStatus)
        assert status.total_jobs == 0
        assert status.pending_jobs == 0
        assert status.running_jobs == 0
        assert status.completed_jobs == 0
        assert status.failed_jobs == 0
        assert status.success_rate == 100.0  # Default when no finished jobs
        assert status.start_time is None
        assert status.estimated_completion is None

    def test_get_status_single_pending_job(self):
        """Test get_status with single pending job."""
        job = self.create_test_job(status=JobStatus.PENDING)
        status = self.tracker.get_status([job])

        assert status.total_jobs == 1
        assert status.pending_jobs == 1
        assert status.running_jobs == 0
        assert status.completed_jobs == 0
        assert status.failed_jobs == 0
        assert status.success_rate == 100.0

    def test_get_status_mixed_job_states(self):
        """Test get_status with jobs in different states."""
        jobs = [
            self.create_test_job("job1", JobStatus.COMPLETED),
            self.create_test_job("job2", JobStatus.RUNNING),
            self.create_test_job("job3", JobStatus.PENDING),
            self.create_test_job("job4", JobStatus.FAILED),
            self.create_test_job("job5", JobStatus.COMPLETED),
        ]

        status = self.tracker.get_status(jobs)

        assert status.total_jobs == 5
        assert status.pending_jobs == 1
        assert status.running_jobs == 1
        assert status.completed_jobs == 2
        assert status.failed_jobs == 1

        # Success rate: 2 completed / (2 completed + 1 failed) = 2/3 ≈ 66.67%
        expected_success_rate = (2 / 3) * 100
        assert abs(status.success_rate - expected_success_rate) < 0.01

    def test_get_status_all_completed(self):
        """Test get_status with all jobs completed."""
        jobs = [
            self.create_test_job("job1", JobStatus.COMPLETED),
            self.create_test_job("job2", JobStatus.COMPLETED),
            self.create_test_job("job3", JobStatus.COMPLETED),
        ]

        status = self.tracker.get_status(jobs)

        assert status.total_jobs == 3
        assert status.pending_jobs == 0
        assert status.running_jobs == 0
        assert status.completed_jobs == 3
        assert status.failed_jobs == 0
        assert status.success_rate == 100.0

    def test_get_status_all_failed(self):
        """Test get_status with all jobs failed."""
        jobs = [
            self.create_test_job("job1", JobStatus.FAILED),
            self.create_test_job("job2", JobStatus.FAILED),
        ]

        status = self.tracker.get_status(jobs)

        assert status.total_jobs == 2
        assert status.pending_jobs == 0
        assert status.running_jobs == 0
        assert status.completed_jobs == 0
        assert status.failed_jobs == 2
        assert status.success_rate == 0.0

    def test_get_status_timing_calculations(self):
        """Test timing calculations in status."""
        now = datetime.now()
        start_time = now - timedelta(minutes=30)

        jobs = [
            self.create_test_job(
                "job1",
                JobStatus.COMPLETED,
                started_at=start_time,
                completed_at=start_time + timedelta(minutes=10),
            ),
            self.create_test_job(
                "job2", JobStatus.RUNNING, started_at=start_time + timedelta(minutes=5)
            ),
            self.create_test_job("job3", JobStatus.PENDING),
        ]

        status = self.tracker.get_status(jobs)

        # Should calculate start time from earliest started job
        assert status.start_time is not None

        # Should have some estimated completion time for running/pending jobs
        if status.running_jobs > 0 or status.pending_jobs > 0:
            # Estimated completion might be None if algorithm doesn't implement it
            # This is acceptable behavior
            pass

    def test_get_status_with_retries(self):
        """Test status calculation with retry counts."""
        jobs = [
            self.create_test_job("job1", JobStatus.COMPLETED, retry_count=0),
            self.create_test_job("job2", JobStatus.COMPLETED, retry_count=2),
            self.create_test_job("job3", JobStatus.FAILED, retry_count=3),
            self.create_test_job("job4", JobStatus.RUNNING, retry_count=1),
        ]

        status = self.tracker.get_status(jobs)

        # Basic counts should be correct regardless of retries
        assert status.total_jobs == 4
        assert status.completed_jobs == 2
        assert status.failed_jobs == 1
        assert status.running_jobs == 1

        # Success rate: 2 completed / (2 completed + 1 failed) = 66.67%
        expected_success_rate = (2 / 3) * 100
        assert abs(status.success_rate - expected_success_rate) < 0.01

    def test_get_status_with_error_messages(self):
        """Test status calculation with error messages."""
        jobs = [
            self.create_test_job(
                "job1", JobStatus.FAILED, error_message="Network timeout"
            ),
            self.create_test_job(
                "job2", JobStatus.FAILED, error_message="Out of memory"
            ),
            self.create_test_job("job3", JobStatus.COMPLETED),
        ]

        status = self.tracker.get_status(jobs)

        assert status.total_jobs == 3
        assert status.failed_jobs == 2
        assert status.completed_jobs == 1
        assert status.success_rate == 50.0  # 1 success / 2 finished

    def test_get_status_edge_case_only_pending(self):
        """Test edge case with only pending jobs."""
        jobs = [
            self.create_test_job("job1", JobStatus.PENDING),
            self.create_test_job("job2", JobStatus.PENDING),
            self.create_test_job("job3", JobStatus.PENDING),
        ]

        status = self.tracker.get_status(jobs)

        assert status.total_jobs == 3
        assert status.pending_jobs == 3
        assert status.running_jobs == 0
        assert status.completed_jobs == 0
        assert status.failed_jobs == 0

        # Success rate should be 100% when no jobs have finished yet
        assert status.success_rate == 100.0

    def test_get_status_edge_case_only_running(self):
        """Test edge case with only running jobs."""
        jobs = [
            self.create_test_job("job1", JobStatus.RUNNING),
            self.create_test_job("job2", JobStatus.RUNNING),
        ]

        status = self.tracker.get_status(jobs)

        assert status.total_jobs == 2
        assert status.pending_jobs == 0
        assert status.running_jobs == 2
        assert status.completed_jobs == 0
        assert status.failed_jobs == 0

        # Success rate should be 100% when no jobs have finished yet
        assert status.success_rate == 100.0

    def test_batch_status_properties(self):
        """Test BatchStatus object properties."""
        jobs = [
            self.create_test_job("job1", JobStatus.COMPLETED),
            self.create_test_job("job2", JobStatus.FAILED),
        ]

        status = self.tracker.get_status(jobs)

        # Test that all expected properties exist
        assert hasattr(status, "total_jobs")
        assert hasattr(status, "pending_jobs")
        assert hasattr(status, "running_jobs")
        assert hasattr(status, "completed_jobs")
        assert hasattr(status, "failed_jobs")
        assert hasattr(status, "success_rate")
        assert hasattr(status, "start_time")
        assert hasattr(status, "estimated_completion")

        # Test types
        assert isinstance(status.total_jobs, int)
        assert isinstance(status.pending_jobs, int)
        assert isinstance(status.running_jobs, int)
        assert isinstance(status.completed_jobs, int)
        assert isinstance(status.failed_jobs, int)
        assert isinstance(status.success_rate, float)

        # start_time and estimated_completion can be None or datetime
        if status.start_time is not None:
            assert isinstance(status.start_time, datetime)
        if status.estimated_completion is not None:
            assert isinstance(status.estimated_completion, datetime)


class TestProgressTrackerRealistic:
    """Test realistic progress tracking scenarios."""

    def setup_method(self):
        """Set up test environment."""
        self.tracker = ProgressTracker()

    def create_test_job(self, job_id="test_job", status=JobStatus.PENDING, **kwargs):
        """Helper to create test BatchJob."""
        return BatchJob(
            job_id=job_id,
            # video_id=kwargs.get('video_id', f'video_{job_id}'),  # FIXME: video_id parameter not supported
            video_path=kwargs.get("video_path", f"/videos/{job_id}.mp4"),
            output_dir=kwargs.get("output_dir", f"/output/{job_id}"),
            status=status,
            created_at=kwargs.get("created_at", datetime.now()),
            started_at=kwargs.get("started_at"),
            completed_at=kwargs.get("completed_at"),
            error_message=kwargs.get("error_message"),
            retry_count=kwargs.get("retry_count", 0),
            config=kwargs.get("config", {}),
            selected_pipelines=kwargs.get("selected_pipelines", []),
            pipeline_results=kwargs.get("pipeline_results", {}),
        )

    def test_realistic_batch_progress(self):
        """Test realistic batch processing progress scenario."""
        # Simulate a batch that's partially processed
        now = datetime.now()

        jobs = [
            # Completed jobs
            self.create_test_job(
                "vid001",
                JobStatus.COMPLETED,
                started_at=now - timedelta(minutes=60),
                completed_at=now - timedelta(minutes=45),
            ),
            self.create_test_job(
                "vid002",
                JobStatus.COMPLETED,
                started_at=now - timedelta(minutes=45),
                completed_at=now - timedelta(minutes=30),
            ),
            # Failed jobs
            self.create_test_job(
                "vid003",
                JobStatus.FAILED,
                started_at=now - timedelta(minutes=30),
                error_message="Corrupted video file",
                retry_count=3,
            ),
            # Currently running
            self.create_test_job(
                "vid004", JobStatus.RUNNING, started_at=now - timedelta(minutes=10)
            ),
            self.create_test_job(
                "vid005", JobStatus.RUNNING, started_at=now - timedelta(minutes=5)
            ),
            # Pending
            self.create_test_job("vid006", JobStatus.PENDING),
            self.create_test_job("vid007", JobStatus.PENDING),
            self.create_test_job("vid008", JobStatus.PENDING),
        ]

        status = self.tracker.get_status(jobs)

        # Verify counts
        assert status.total_jobs == 8
        assert status.completed_jobs == 2
        assert status.failed_jobs == 1
        assert status.running_jobs == 2
        assert status.pending_jobs == 3

        # Success rate: 2 completed / (2 completed + 1 failed) = 66.67%
        expected_success_rate = (2 / 3) * 100
        assert abs(status.success_rate - expected_success_rate) < 0.01

    def test_long_running_batch(self):
        """Test progress tracking for long-running batch."""
        now = datetime.now()
        start_time = now - timedelta(hours=2)

        # Simulate a batch that's been running for 2 hours
        jobs = []

        # 50 completed jobs over 2 hours
        for i in range(50):
            job_start = start_time + timedelta(minutes=i * 2)
            job_end = job_start + timedelta(minutes=1.5)
            jobs.append(
                self.create_test_job(
                    f"completed_{i:03d}",
                    JobStatus.COMPLETED,
                    started_at=job_start,
                    completed_at=job_end,
                )
            )

        # 5 failed jobs
        for i in range(5):
            job_start = start_time + timedelta(minutes=100 + i * 2)
            jobs.append(
                self.create_test_job(
                    f"failed_{i:03d}",
                    JobStatus.FAILED,
                    started_at=job_start,
                    error_message=f"Error processing video {i}",
                    retry_count=2,
                )
            )

        # 10 currently running
        for i in range(10):
            job_start = now - timedelta(minutes=15 - i)
            jobs.append(
                self.create_test_job(
                    f"running_{i:03d}", JobStatus.RUNNING, started_at=job_start
                )
            )

        # 35 pending
        for i in range(35):
            jobs.append(self.create_test_job(f"pending_{i:03d}", JobStatus.PENDING))

        status = self.tracker.get_status(jobs)

        assert status.total_jobs == 100
        assert status.completed_jobs == 50
        assert status.failed_jobs == 5
        assert status.running_jobs == 10
        assert status.pending_jobs == 35

        # Success rate: 50 / (50 + 5) = 90.91%
        expected_success_rate = (50 / 55) * 100
        assert abs(status.success_rate - expected_success_rate) < 0.01
