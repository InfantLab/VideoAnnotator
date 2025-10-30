"""Unit tests for batch progress tracking functionality.

Tests the ProgressTracker class that monitors batch job progress,
calculates metrics, and provides real-time status updates.
"""

from pathlib import Path

import pytest

from videoannotator.batch.progress_tracker import ProgressTracker
from videoannotator.batch.types import BatchJob, BatchStatus, JobStatus


class TestProgressTracker:
    """Test ProgressTracker functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tracker = ProgressTracker()

    def test_tracker_initialization(self):
        """Test progress tracker initialization."""
        assert self.tracker.start_time is None
        assert len(self.tracker.completion_history) == 0
        assert self.tracker.current_jobs == {}
        assert self.tracker.total_processing_time == 0.0
        assert self.tracker.jobs_completed == 0
        assert self.tracker.jobs_failed == 0

    def test_start_job_tracking(self):
        """Test starting job tracking."""
        job1 = BatchJob(video_path=Path("video1.mp4"))
        job2 = BatchJob(video_path=Path("video2.mp4"))

        # Start tracking jobs
        self.tracker.start_job(job1.job_id)
        assert len(self.tracker.current_jobs) == 1
        assert job1.job_id in self.tracker.current_jobs

        self.tracker.start_job(job2.job_id)
        assert len(self.tracker.current_jobs) == 2

    def test_duplicate_job_tracking(self):
        """Test starting tracking for duplicate job ID."""
        job = BatchJob(video_path=Path("video1.mp4"))

        # Start tracking same job ID twice
        self.tracker.start_job(job.job_id)
        self.tracker.start_job(job.job_id)  # Should update start time

        assert len(self.tracker.current_jobs) == 1
        assert job.job_id in self.tracker.current_jobs

    def test_job_completion_tracking(self):
        """Test job completion tracking and timing."""
        job = BatchJob(video_path=Path("video1.mp4"))

        # Start and complete job
        self.tracker.start_job(job.job_id)
        assert job.job_id in self.tracker.current_jobs

        # Mark job as completed and track completion
        job.status = JobStatus.COMPLETED
        self.tracker.complete_job(job)

        # Job should be removed from current jobs and added to history
        assert job.job_id not in self.tracker.current_jobs
        assert len(self.tracker.completion_history) == 1
        assert self.tracker.jobs_completed == 1

    def test_nonexistent_job_operations(self):
        """Test operations on non-existent jobs."""
        # These should not raise errors
        job = BatchJob(video_path=Path("nonexistent.mp4"))
        job.status = JobStatus.FAILED

        # Complete job that was never started - should handle gracefully
        self.tracker.complete_job(job)

    def test_get_status_empty(self):
        """Test getting status with no jobs."""
        # ProgressTracker.get_status() requires a jobs list parameter
        empty_jobs = []
        status = self.tracker.get_status(empty_jobs)

        assert isinstance(status, BatchStatus)
        assert status.total_jobs == 0
        assert status.pending_jobs == 0
        assert status.running_jobs == 0
        assert status.completed_jobs == 0
        assert status.failed_jobs == 0
        assert status.cancelled_jobs == 0
        assert status.progress_percentage == 0.0
        assert status.success_rate == 0.0  # Empty batch has 0% success rate
        assert status.current_jobs == []

    def test_get_status_with_jobs(self):
        """Test getting status with various job states."""
        # Create jobs in different states
        jobs = [
            BatchJob(video_path=Path("video1.mp4"), status=JobStatus.PENDING),
            BatchJob(video_path=Path("video2.mp4"), status=JobStatus.PENDING),
            BatchJob(video_path=Path("video3.mp4"), status=JobStatus.RUNNING),
            BatchJob(video_path=Path("video4.mp4"), status=JobStatus.COMPLETED),
            BatchJob(video_path=Path("video5.mp4"), status=JobStatus.COMPLETED),
            BatchJob(video_path=Path("video6.mp4"), status=JobStatus.FAILED),
            BatchJob(video_path=Path("video7.mp4"), status=JobStatus.CANCELLED),
        ]

        # Start tracking one of the running jobs
        self.tracker.start_job(jobs[2].job_id)  # The RUNNING job

        # ProgressTracker.get_status() needs the jobs list as parameter
        status = self.tracker.get_status(jobs)

        assert status.total_jobs == 7
        assert status.pending_jobs == 2
        assert status.running_jobs == 1
        assert status.completed_jobs == 2
        assert status.failed_jobs == 1
        assert status.cancelled_jobs == 1
        assert status.progress_percentage == pytest.approx(57.14, abs=0.1)
        assert status.success_rate == pytest.approx(
            50.0, abs=0.1
        )  # 2 completed out of 4 finished jobs (2+1+1)
        assert len(status.current_jobs) == 1  # Only the job we started tracking
        assert jobs[2].job_id in status.current_jobs


@pytest.mark.integration
class TestProgressTrackerIntegration:
    """Integration tests for progress tracker."""

    def test_batch_start_and_timing(self):
        """Test batch start and basic timing functionality."""
        tracker = ProgressTracker()

        # Start batch
        tracker.start_batch()
        assert tracker.start_time is not None
        assert len(tracker.completion_history) == 0
        assert tracker.current_jobs == {}

        # Start some jobs
        job_ids = ["job1", "job2", "job3"]
        for job_id in job_ids:
            tracker.start_job(job_id)

        assert len(tracker.current_jobs) == 3

        # Create some jobs to test status calculation
        jobs = [
            BatchJob(video_path=Path("video1.mp4"), status=JobStatus.PENDING),
            BatchJob(video_path=Path("video2.mp4"), status=JobStatus.RUNNING),
            BatchJob(video_path=Path("video3.mp4"), status=JobStatus.COMPLETED),
        ]

        status = tracker.get_status(jobs)
        assert status.total_jobs == 3
        assert status.pending_jobs == 1
        assert status.running_jobs == 1
        assert status.completed_jobs == 1

    def test_job_completion_workflow(self):
        """Test complete job completion workflow."""
        tracker = ProgressTracker()
        tracker.start_batch()

        # Create and complete jobs
        completed_jobs = []
        for i in range(3):
            job = BatchJob(video_path=Path(f"video{i}.mp4"))
            job.status = JobStatus.COMPLETED

            # Start and complete job tracking
            tracker.start_job(job.job_id)
            tracker.complete_job(job)

            completed_jobs.append(job)

        # Check metrics
        assert tracker.jobs_completed == 3
        assert len(tracker.completion_history) == 3
        assert len(tracker.current_jobs) == 0  # All completed

        # Test status with completed jobs
        status = tracker.get_status(completed_jobs)
        assert status.total_jobs == 3
        assert status.completed_jobs == 3
        assert status.progress_percentage == 100.0
