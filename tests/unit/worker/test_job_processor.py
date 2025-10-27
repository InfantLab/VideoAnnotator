"""Tests for JobProcessor worker (v1.3.0).

Tests cover:
- Basic job processing
- Retry logic with exponential backoff
- Cancellation support
- Storage paths integration
- Error handling
- Concurrent job processing
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from videoannotator.batch.types import BatchJob, JobStatus
from videoannotator.storage.sqlite_backend import SQLiteStorageBackend
from videoannotator.worker.job_processor import JobProcessor


class TestJobProcessor:
    """Tests for JobProcessor worker."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.db_path = self.temp_dir / "test_jobs.db"
        self.storage = SQLiteStorageBackend(self.db_path)
        self.processor = JobProcessor(
            storage_backend=self.storage,
            poll_interval=0.1,  # Fast polling for tests
            max_concurrent_jobs=2,
            max_retries=3,
        )

    def teardown_method(self):
        """Clean up test environment."""
        import shutil

        self.processor.stop()
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_processor_initialization(self):
        """Test JobProcessor initializes correctly."""
        assert self.processor.storage is not None
        assert self.processor.poll_interval == 0.1
        assert self.processor.max_concurrent_jobs == 2
        assert self.processor.max_retries == 3
        assert self.processor.running is False
        assert len(self.processor.processing_jobs) == 0

    def test_request_cancellation(self):
        """Test cancellation request is tracked."""
        job_id = "test-job-123"
        self.processor.request_cancellation(job_id)

        assert job_id in self.processor.cancellation_requests
        assert self.processor._is_cancellation_requested(job_id)

    def test_should_retry_job(self):
        """Test retry logic decision."""
        job = BatchJob(
            video_path=Path("/test/video.mp4"),
            status=JobStatus.FAILED,
            retry_count=0,
        )

        # Should retry when under limit
        assert self.processor._should_retry_job(job)

        # Should not retry when at limit
        job.retry_count = 3
        assert not self.processor._should_retry_job(job)

        # Should not retry when over limit
        job.retry_count = 5
        assert not self.processor._should_retry_job(job)

    def test_calculate_retry_delay(self):
        """Test exponential backoff calculation."""
        processor = JobProcessor(retry_delay_base=2.0)

        assert processor._calculate_retry_delay(0) == 1.0  # 2^0
        assert processor._calculate_retry_delay(1) == 2.0  # 2^1
        assert processor._calculate_retry_delay(2) == 4.0  # 2^2
        assert processor._calculate_retry_delay(3) == 8.0  # 2^3

    @pytest.mark.asyncio
    async def test_process_job_with_cancellation_before_start(self):
        """Test job cancelled before processing starts."""
        job = BatchJob(
            video_path=Path("/test/video.mp4"),
            status=JobStatus.PENDING,
            config={},
        )
        self.storage.save_job_metadata(job)

        # Request cancellation before processing
        self.processor.request_cancellation(job.job_id)

        # Process job
        await self.processor._process_job_async(job)

        # Load and verify job was cancelled
        updated_job = self.storage.load_job_metadata(job.job_id)
        assert updated_job.status == JobStatus.CANCELLED
        assert updated_job.error_message is not None
        assert "cancelled" in updated_job.error_message.lower()
        assert job.job_id not in self.processor.cancellation_requests

    @pytest.mark.asyncio
    async def test_storage_directory_creation(self):
        """Test storage directory is created for job."""
        from storage.config import get_job_storage_path

        job = BatchJob(
            video_path=Path("/test/video.mp4"),
            status=JobStatus.PENDING,
            config={},
        )
        job.storage_path = get_job_storage_path(job.job_id)
        self.storage.save_job_metadata(job)

        # Mock the processing to succeed quickly
        with patch.object(
            self.processor, "_run_single_job_processing", return_value=True
        ):
            await self.processor._process_job_async(job)

        # Verify storage directory was created
        assert job.storage_path.exists()
        assert job.storage_path.is_dir()

    @pytest.mark.asyncio
    async def test_successful_job_processing(self):
        """Test successful job processing flow."""
        job = BatchJob(
            video_path=Path("/test/video.mp4"),
            status=JobStatus.PENDING,
            config={},
        )
        from storage.config import get_job_storage_path

        job.storage_path = get_job_storage_path(job.job_id)
        self.storage.save_job_metadata(job)

        # Mock successful processing
        with patch.object(
            self.processor, "_run_single_job_processing", return_value=True
        ):
            await self.processor._process_job_async(job)

        # Verify job completed
        updated_job = self.storage.load_job_metadata(job.job_id)
        assert updated_job.status == JobStatus.COMPLETED
        assert updated_job.completed_at is not None
        assert updated_job.retry_count == 0

    @pytest.mark.asyncio
    async def test_failed_job_with_retry(self):
        """Test failed job gets retried."""
        job = BatchJob(
            video_path=Path("/test/video.mp4"),
            status=JobStatus.PENDING,
            config={},
            retry_count=0,
        )
        from storage.config import get_job_storage_path

        job.storage_path = get_job_storage_path(job.job_id)
        self.storage.save_job_metadata(job)

        # Mock failed processing
        with patch.object(
            self.processor, "_run_single_job_processing", return_value=False
        ):
            await self.processor._process_job_async(job)

        # Verify job is pending for retry
        updated_job = self.storage.load_job_metadata(job.job_id)
        assert updated_job.status == JobStatus.PENDING
        assert updated_job.retry_count == 1

    @pytest.mark.asyncio
    async def test_failed_job_max_retries_exceeded(self):
        """Test job fails after max retries."""
        job = BatchJob(
            video_path=Path("/test/video.mp4"),
            status=JobStatus.PENDING,
            config={},
            retry_count=3,  # Already at max
        )
        from storage.config import get_job_storage_path

        job.storage_path = get_job_storage_path(job.job_id)
        self.storage.save_job_metadata(job)

        # Mock failed processing
        with patch.object(
            self.processor, "_run_single_job_processing", return_value=False
        ):
            await self.processor._process_job_async(job)

        # Verify job failed permanently
        updated_job = self.storage.load_job_metadata(job.job_id)
        assert updated_job.status == JobStatus.FAILED
        assert updated_job.completed_at is not None
        assert updated_job.error_message is not None
        assert "max retries" in updated_job.error_message.lower()

    @pytest.mark.asyncio
    async def test_exception_during_processing_with_retry(self):
        """Test exception triggers retry logic."""
        job = BatchJob(
            video_path=Path("/test/video.mp4"),
            status=JobStatus.PENDING,
            config={},
            retry_count=0,
        )
        from storage.config import get_job_storage_path

        job.storage_path = get_job_storage_path(job.job_id)
        self.storage.save_job_metadata(job)

        # Mock exception during processing
        with patch.object(
            self.processor,
            "_run_single_job_processing",
            side_effect=RuntimeError("Test error"),
        ):
            await self.processor._process_job_async(job)

        # Verify job is pending for retry
        updated_job = self.storage.load_job_metadata(job.job_id)
        assert updated_job.status == JobStatus.PENDING
        assert updated_job.retry_count == 1
        assert updated_job.error_message is not None
        assert "Test error" in updated_job.error_message

    @pytest.mark.asyncio
    async def test_process_cycle_handles_pending_jobs(self):
        """Test process cycle picks up pending jobs."""
        # Create pending jobs
        job1 = BatchJob(video_path=Path("/test/video1.mp4"), status=JobStatus.PENDING)
        job2 = BatchJob(video_path=Path("/test/video2.mp4"), status=JobStatus.PENDING)

        self.storage.save_job_metadata(job1)
        self.storage.save_job_metadata(job2)

        # Mock processing
        with patch.object(
            self.processor, "_run_single_job_processing", return_value=True
        ):
            await self.processor._process_cycle()

        # Both jobs should be picked up (max_concurrent_jobs=2)
        assert len(self.processor.processing_jobs) == 2

    @pytest.mark.asyncio
    async def test_process_cycle_respects_max_concurrent(self):
        """Test process cycle respects max concurrent jobs limit."""
        # Create more jobs than max_concurrent
        jobs = [
            BatchJob(video_path=Path(f"/test/video{i}.mp4"), status=JobStatus.PENDING)
            for i in range(5)
        ]

        for job in jobs:
            self.storage.save_job_metadata(job)

        # Mock processing to keep jobs running
        async def mock_process_long(job):
            await asyncio.sleep(1)  # Keep job running

        with patch.object(self.processor, "_process_job_async", new=mock_process_long):
            await self.processor._process_cycle()

        # Only max_concurrent_jobs should be started
        assert len(self.processor.processing_jobs) <= self.processor.max_concurrent_jobs

    @pytest.mark.asyncio
    async def test_process_cycle_handles_retrying_jobs(self):
        """Test process cycle picks up jobs in RETRYING status."""
        job = BatchJob(
            video_path=Path("/test/video.mp4"),
            status=JobStatus.RETRYING,
            retry_count=1,
        )
        self.storage.save_job_metadata(job)

        # Mock processing
        with patch.object(
            self.processor, "_run_single_job_processing", return_value=True
        ):
            await self.processor._process_cycle()

        # Job should be picked up
        assert job.job_id in self.processor.processing_jobs

    def test_signal_handler_stops_processor(self):
        """Test signal handler sets running to False."""
        self.processor.running = True
        self.processor._signal_handler(2, None)  # SIGINT
        assert self.processor.running is False

    @pytest.mark.asyncio
    async def test_cleanup_after_job_completion(self):
        """Test job is removed from processing set after completion."""
        job = BatchJob(
            video_path=Path("/test/video.mp4"),
            status=JobStatus.PENDING,
            config={},
        )
        from storage.config import get_job_storage_path

        job.storage_path = get_job_storage_path(job.job_id)
        self.storage.save_job_metadata(job)

        # Add to processing set
        self.processor.processing_jobs.add(job.job_id)

        # Mock successful processing
        with patch.object(
            self.processor, "_run_single_job_processing", return_value=True
        ):
            await self.processor._process_job_async(job)

        # Job should be removed from processing set
        assert job.job_id not in self.processor.processing_jobs

    @pytest.mark.asyncio
    async def test_cancellation_request_cleanup(self):
        """Test cancellation request is removed after handling."""
        job = BatchJob(
            video_path=Path("/test/video.mp4"),
            status=JobStatus.PENDING,
            config={},
        )
        self.storage.save_job_metadata(job)

        # Request cancellation
        self.processor.request_cancellation(job.job_id)
        assert job.job_id in self.processor.cancellation_requests

        # Process job (will be cancelled)
        await self.processor._process_job_async(job)

        # Cancellation request should be cleaned up
        assert job.job_id not in self.processor.cancellation_requests
