"""Unit tests for batch failure recovery functionality.

Tests the FailureRecovery class that handles retry logic,
error classification, and recovery strategies."""

from pathlib import Path

from src.batch.recovery import FailureRecovery, RetryStrategy
from src.batch.types import BatchJob, JobStatus


class TestFailureRecovery:
    """Test FailureRecovery functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.recovery = FailureRecovery()

    def test_recovery_initialization(self):
        """Test recovery system initialization."""
        assert self.recovery.max_retries == 3
        assert self.recovery.base_delay == 1.0
        assert self.recovery.max_delay == 300.0
        assert self.recovery.strategy == RetryStrategy.EXPONENTIAL_BACKOFF

    def test_custom_initialization(self):
        """Test recovery with custom parameters."""
        recovery = FailureRecovery(
            max_retries=5,
            base_delay=2.0,
            max_delay=600.0,
            strategy=RetryStrategy.FIXED_DELAY,
        )

        assert recovery.max_retries == 5
        assert recovery.base_delay == 2.0
        assert recovery.max_delay == 600.0
        assert recovery.strategy == RetryStrategy.FIXED_DELAY

    def test_should_retry_new_job(self):
        """Test retry decision for new job."""
        job = BatchJob(video_path=Path("video1.mp4"))
        _error = Exception("Test error")

        # New job should be retryable
        assert self.recovery.should_retry(job, _error)

    def test_should_retry_within_limit(self):
        """Test retry decision within retry limit."""
        job = BatchJob(video_path=Path("video1.mp4"))

        # Test retries within limit
        for i in range(self.recovery.max_retries):
            error = Exception(f"Error {i + 1}")
            assert self.recovery.should_retry(job, error)
            # Simulate retry by incrementing count
            job.retry_count += 1

        # Should not retry after max retries
        final_error = Exception("Final error")
        assert not self.recovery.should_retry(job, final_error)

    def test_calculate_retry_delay_exponential(self):
        """Test exponential backoff delay calculation."""
        job = BatchJob(video_path=Path("video1.mp4"))

        # Test exponential increase
        delays = []
        for i in range(4):
            job.retry_count = i
            delay = self.recovery.calculate_retry_delay(job)
            delays.append(delay)
            assert delay <= self.recovery.max_delay

        # Each delay should be larger than the previous (exponential)
        assert delays[1] > delays[0]
        assert delays[2] > delays[1]
        assert delays[3] > delays[2]

    def test_calculate_retry_delay_fixed(self):
        """Test fixed delay calculation."""
        recovery = FailureRecovery(strategy=RetryStrategy.FIXED_DELAY)
        job = BatchJob(video_path=Path("video1.mp4"))

        # All delays should be the same
        delays = []
        for i in range(3):
            job.retry_count = i
            delay = recovery.calculate_retry_delay(job)
            delays.append(delay)

        assert all(d == recovery.base_delay for d in delays)

    def test_calculate_retry_delay_max_limit(self):
        """Test delay calculation respects max limit."""
        job = BatchJob(video_path=Path("video1.mp4"))
        job.retry_count = 10  # High retry count

        delay = self.recovery.calculate_retry_delay(job)
        assert delay <= self.recovery.max_delay

    def test_prepare_retry(self):
        """Test preparing job for retry."""
        job = BatchJob(video_path=Path("video1.mp4"))
        job.status = JobStatus.FAILED
        job.error_message = "Original error"
        original_retry_count = job.retry_count

        error = Exception("Retry error")
        updated_job = self.recovery.prepare_retry(job, error)

        # Should increment retry count and preserve error
        assert updated_job.retry_count == original_retry_count + 1
        assert updated_job.status == JobStatus.RETRYING
        assert updated_job.error_message == "Retry error"

    def test_handle_partial_failure(self):
        """Test handling partial pipeline failures."""
        job = BatchJob(video_path=Path("video1.mp4"))
        error = Exception("Pipeline error")

        # Should return True for retryable pipeline errors
        result = self.recovery.handle_partial_failure(job, "audio_processing", error)
        assert isinstance(result, bool)

    def test_error_classification(self):
        """Test error classification for retry decisions."""
        job = BatchJob(video_path=Path("video1.mp4"))

        # Test different error types
        network_error = ConnectionError("Network timeout")
        file_error = FileNotFoundError("File not found")
        memory_error = MemoryError("Out of memory")

        # Most errors should be retryable initially
        assert self.recovery.should_retry(job, network_error)
        assert self.recovery.should_retry(job, file_error)
        assert self.recovery.should_retry(job, memory_error)


class TestFailureRecoveryIntegration:
    """Integration tests for failure recovery."""

    def test_complete_retry_workflow(self):
        """Test complete retry workflow."""
        recovery = FailureRecovery(max_retries=2)
        job = BatchJob(video_path=Path("video1.mp4"))

        # First failure
        error1 = Exception("First error")
        assert recovery.should_retry(job, error1)

        job = recovery.prepare_retry(job, error1)
        delay1 = recovery.calculate_retry_delay(job)
        assert delay1 >= recovery.base_delay

        # Second failure
        error2 = Exception("Second error")
        assert recovery.should_retry(job, error2)

        job = recovery.prepare_retry(job, error2)
        delay2 = recovery.calculate_retry_delay(job)
        assert delay2 >= delay1  # Should be larger for exponential backoff

        # Third failure - should not retry
        error3 = Exception("Third error")
        assert not recovery.should_retry(job, error3)

    def test_different_retry_strategies(self):
        """Test different retry strategies."""
        job = BatchJob(video_path=Path("video1.mp4"))
        _error = Exception("Test error")

        # Test exponential backoff
        exp_recovery = FailureRecovery(strategy=RetryStrategy.EXPONENTIAL_BACKOFF)
        exp_delays = []
        for i in range(3):
            job.retry_count = i
            exp_delays.append(exp_recovery.calculate_retry_delay(job))

        # Test fixed delay
        fixed_recovery = FailureRecovery(strategy=RetryStrategy.FIXED_DELAY)
        fixed_delays = []
        for i in range(3):
            job.retry_count = i
            fixed_delays.append(fixed_recovery.calculate_retry_delay(job))

        # Exponential should increase, fixed should stay same
        assert exp_delays[0] < exp_delays[1] < exp_delays[2]
        assert fixed_delays[0] == fixed_delays[1] == fixed_delays[2]

    def test_retry_with_different_errors(self):
        """Test retry behavior with different error types."""
        recovery = FailureRecovery()
        job = BatchJob(video_path=Path("video1.mp4"))

        # Test various error types
        errors = [
            ConnectionError("Network issue"),
            TimeoutError("Operation timeout"),
            ValueError("Invalid value"),
            RuntimeError("Runtime issue"),
            Exception("Generic error"),
        ]

        for error in errors:
            # All should be retryable for new job
            assert recovery.should_retry(job, error)

    def test_job_state_preservation(self):
        """Test that job state is properly preserved during retry."""
        recovery = FailureRecovery()
        job = BatchJob(
            video_path=Path("video1.mp4"),
            output_dir=Path("output"),
            config={"test": "config"},
        )

        original_id = job.job_id
        original_config = job.config.copy()

        error = Exception("Test error")
        updated_job = recovery.prepare_retry(job, error)

        # Core properties should be preserved
        assert updated_job.job_id == original_id
        assert updated_job.video_path == job.video_path
        assert updated_job.output_dir == job.output_dir
        assert updated_job.config == original_config

        # Status should be set for retry
        assert updated_job.status == JobStatus.RETRYING
        assert updated_job.error_message == "Test error"
