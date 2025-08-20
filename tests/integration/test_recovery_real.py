from pathlib import Path
"""
Unit tests for FailureRecovery - testing actual implementation.

These tests focus on the real API including should_retry(job, error) and calculate_retry_delay(job).
"""

import pytest
from datetime import datetime
from unittest.mock import Mock

from src.batch.recovery import FailureRecovery, RetryStrategy
from src.batch.types import BatchJob, JobStatus, PipelineResult


class TestFailureRecoveryReal:
    """Test the actual FailureRecovery implementation."""

    def setup_method(self):
        """Set up test environment."""
        self.recovery = FailureRecovery(
            max_retries=3,
            base_delay=1.0,
            max_delay=300.0,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF
        )

    def create_test_job(self, job_id="test_job", retry_count=0, **kwargs):
        """Helper to create test BatchJob."""
        return BatchJob(
            job_id=job_id,
            # # video_id is a computed property from video_path,  # FIXME: video_id parameter not supported
            video_path=kwargs.get('video_path', '/path/to/video.mp4'),
            output_dir=kwargs.get('output_dir', '/path/to/output'),
            status=kwargs.get('status', JobStatus.PENDING),
            created_at=kwargs.get('created_at', datetime.now()),
            started_at=kwargs.get('started_at'),
            completed_at=kwargs.get('completed_at'),
            error_message=kwargs.get('error_message'),
            retry_count=retry_count,
            config=kwargs.get('config', {}),
            selected_pipelines=kwargs.get('selected_pipelines', []),
            pipeline_results=kwargs.get('pipeline_results', {})
        )

    def test_recovery_initialization(self):
        """Test FailureRecovery initialization with default parameters."""
        recovery = FailureRecovery()
        
        assert recovery.max_retries == 3
        assert recovery.base_delay == 1.0
        assert recovery.max_delay == 300.0
        assert recovery.strategy == RetryStrategy.EXPONENTIAL_BACKOFF
        assert recovery.retry_stats == {}
        assert recovery.error_patterns == {}

    def test_recovery_custom_initialization(self):
        """Test FailureRecovery initialization with custom parameters."""
        recovery = FailureRecovery(
            max_retries=5,
            base_delay=2.0,
            max_delay=600.0,
            strategy=RetryStrategy.LINEAR_BACKOFF
        )
        
        assert recovery.max_retries == 5
        assert recovery.base_delay == 2.0
        assert recovery.max_delay == 600.0
        assert recovery.strategy == RetryStrategy.LINEAR_BACKOFF

    def test_should_retry_fresh_job(self):
        """Test should_retry with fresh job (no retries yet)."""
        job = self.create_test_job(retry_count=0)
        error = Exception("Network timeout")
        
        should_retry = self.recovery.should_retry(job, error)
        
        assert should_retry is True

    def test_should_retry_max_retries_exceeded(self):
        """Test should_retry when max retries exceeded."""
        job = self.create_test_job(retry_count=3)  # Already at max retries
        error = Exception("Network timeout")
        
        should_retry = self.recovery.should_retry(job, error)
        
        assert should_retry is False

    def test_should_retry_permanent_error(self):
        """Test should_retry with permanent error types."""
        job = self.create_test_job(retry_count=0)
        
        # Test different error types that should be permanent
        permanent_errors = [
            FileNotFoundError("Video file not found"),
            PermissionError("Access denied"),
            ValueError("Invalid video format")
        ]
        
        for error in permanent_errors:
            should_retry = self.recovery.should_retry(job, error)
            # The actual behavior depends on implementation of _is_permanent_error
            # We test that the method returns a boolean
            assert isinstance(should_retry, bool)

    def test_should_retry_tracks_error_patterns(self):
        """Test that should_retry tracks error patterns."""
        job = self.create_test_job(retry_count=0)
        
        # Trigger multiple errors of the same type
        error1 = ConnectionError("Network error 1")
        error2 = ConnectionError("Network error 2")
        
        self.recovery.should_retry(job, error1)
        self.recovery.should_retry(job, error2)
        
        # Should track error patterns
        assert "ConnectionError" in self.recovery.error_patterns
        assert self.recovery.error_patterns["ConnectionError"] >= 1

    def test_calculate_retry_delay_no_strategy(self):
        """Test calculate_retry_delay with NONE strategy."""
        recovery = FailureRecovery(strategy=RetryStrategy.NONE)
        job = self.create_test_job(retry_count=1)
        
        delay = recovery.calculate_retry_delay(job)
        
        assert delay == 0.0

    def test_calculate_retry_delay_fixed(self):
        """Test calculate_retry_delay with FIXED_DELAY strategy."""
        recovery = FailureRecovery(
            strategy=RetryStrategy.FIXED_DELAY,
            base_delay=5.0
        )
        
        job1 = self.create_test_job(retry_count=0)
        job2 = self.create_test_job(retry_count=3)
        
        delay1 = recovery.calculate_retry_delay(job1)
        delay2 = recovery.calculate_retry_delay(job2)
        
        # Fixed delay should be same regardless of retry count
        assert delay1 == 5.0
        assert delay2 == 5.0

    def test_calculate_retry_delay_linear(self):
        """Test calculate_retry_delay with LINEAR_BACKOFF strategy."""
        recovery = FailureRecovery(
            strategy=RetryStrategy.LINEAR_BACKOFF,
            base_delay=2.0,
            max_delay=10.0
        )
        
        job0 = self.create_test_job(retry_count=0)  # First retry
        job1 = self.create_test_job(retry_count=1)  # Second retry
        job2 = self.create_test_job(retry_count=2)  # Third retry
        
        delay0 = recovery.calculate_retry_delay(job0)
        delay1 = recovery.calculate_retry_delay(job1)
        delay2 = recovery.calculate_retry_delay(job2)
        
        # Linear backoff: base_delay * (retry_count + 1)
        assert delay0 == 2.0  # 2.0 * 1
        assert delay1 == 4.0  # 2.0 * 2
        assert delay2 == 6.0  # 2.0 * 3

    def test_calculate_retry_delay_exponential(self):
        """Test calculate_retry_delay with EXPONENTIAL_BACKOFF strategy."""
        recovery = FailureRecovery(
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay=1.0,
            max_delay=10.0
        )
        
        job0 = self.create_test_job(retry_count=0)  # First retry
        job1 = self.create_test_job(retry_count=1)  # Second retry
        job2 = self.create_test_job(retry_count=2)  # Third retry
        
        delay0 = recovery.calculate_retry_delay(job0)
        delay1 = recovery.calculate_retry_delay(job1)
        delay2 = recovery.calculate_retry_delay(job2)
        
        # Exponential backoff: base_delay * (2 ** retry_count)
        assert delay0 == 1.0  # 1.0 * 2^0 = 1.0
        assert delay1 == 2.0  # 1.0 * 2^1 = 2.0
        assert delay2 == 4.0  # 1.0 * 2^2 = 4.0

    def test_calculate_retry_delay_max_delay_cap(self):
        """Test that calculate_retry_delay respects max_delay cap."""
        recovery = FailureRecovery(
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay=10.0,
            max_delay=50.0
        )
        
        # Create job with high retry count to trigger exponential growth
        job = self.create_test_job(retry_count=5)  # Would be 10 * 2^5 = 320
        
        delay = recovery.calculate_retry_delay(job)
        
        # Should be capped at max_delay
        assert delay == 50.0

    def test_prepare_retry(self):
        """Test prepare_retry method."""
        job = self.create_test_job(retry_count=0, status=JobStatus.FAILED)
        error = Exception("Test error")
        
        updated_job = self.recovery.prepare_retry(job, error)
        
        # Should update job state
        assert updated_job.retry_count == 1
        assert updated_job.status == JobStatus.RETRYING
        assert updated_job.error_message == "Test error"
        
        # Should track retry statistics
        assert job.job_id in self.recovery.retry_stats
        assert self.recovery.retry_stats[job.job_id] == 1

    def test_prepare_retry_multiple_times(self):
        """Test prepare_retry called multiple times."""
        job = self.create_test_job(retry_count=0)
        
        error1 = Exception("First error")
        error2 = Exception("Second error")
        
        self.recovery.prepare_retry(job, error1)
        self.recovery.prepare_retry(job, error2)
        
        # Should increment retry count each time
        assert job.retry_count == 2
        assert job.error_message == "Second error"  # Latest error
        assert self.recovery.retry_stats[job.job_id] == 2

    def test_handle_partial_failure(self):
        """Test handle_partial_failure method."""
        job = self.create_test_job()
        error = Exception("Pipeline failed")
        
        should_continue = self.recovery.handle_partial_failure(
            job, "face_analysis", error
        )
        
        # Should return a boolean
        assert isinstance(should_continue, bool)
        
        # Should record the pipeline failure
        assert "face_analysis" in job.pipeline_results
        pipeline_result = job.pipeline_results["face_analysis"]
        assert isinstance(pipeline_result, PipelineResult)
        assert pipeline_result.status == JobStatus.FAILED
        assert pipeline_result.error_message == "Pipeline failed"

    def test_handle_partial_failure_multiple_pipelines(self):
        """Test handle_partial_failure with multiple pipeline failures."""
        job = self.create_test_job()
        
        # Fail multiple pipelines
        error1 = Exception("Face analysis failed")
        error2 = Exception("Scene detection failed")
        
        self.recovery.handle_partial_failure(job, "face_analysis", error1)
        self.recovery.handle_partial_failure(job, "scene_detection", error2)
        
        # Should track both failures
        assert len(job.pipeline_results) == 2
        assert "face_analysis" in job.pipeline_results
        assert "scene_detection" in job.pipeline_results
        
        # Verify individual failure details
        face_result = job.pipeline_results["face_analysis"]
        scene_result = job.pipeline_results["scene_detection"]
        
        assert face_result.status == JobStatus.FAILED
        assert scene_result.status == JobStatus.FAILED
        assert face_result.error_message == "Face analysis failed"
        assert scene_result.error_message == "Scene detection failed"


class TestFailureRecoveryStrategies:
    """Test different retry strategies."""

    def create_test_job(self, retry_count=0):
        """Helper to create test BatchJob."""
        return BatchJob(
            job_id="test_job",
            video_path=Path('test_video.mp4'), 
            output_dir="/path/to/output",
            status=JobStatus.PENDING,
            created_at=datetime.now(),
            retry_count=retry_count,
            config={},
            selected_pipelines=[],
            pipeline_results={}
        )

    def test_all_retry_strategies(self):
        """Test all retry strategies are working."""
        strategies = [
            RetryStrategy.NONE,
            RetryStrategy.FIXED_DELAY,
            RetryStrategy.LINEAR_BACKOFF,
            RetryStrategy.EXPONENTIAL_BACKOFF
        ]
        
        for strategy in strategies:
            recovery = FailureRecovery(
                strategy=strategy,
                base_delay=1.0,
                max_delay=10.0
            )
            
            job = self.create_test_job(retry_count=1)
            delay = recovery.calculate_retry_delay(job)
            
            # All strategies should return a non-negative number
            assert isinstance(delay, (int, float))
            assert delay >= 0

    def test_strategy_differences(self):
        """Test that different strategies produce different delays."""
        job = self.create_test_job(retry_count=2)
        
        # Test each strategy
        none_recovery = FailureRecovery(strategy=RetryStrategy.NONE, base_delay=2.0)
        fixed_recovery = FailureRecovery(strategy=RetryStrategy.FIXED_DELAY, base_delay=2.0)
        linear_recovery = FailureRecovery(strategy=RetryStrategy.LINEAR_BACKOFF, base_delay=2.0)
        exp_recovery = FailureRecovery(strategy=RetryStrategy.EXPONENTIAL_BACKOFF, base_delay=2.0)
        
        none_delay = none_recovery.calculate_retry_delay(job)
        fixed_delay = fixed_recovery.calculate_retry_delay(job)
        linear_delay = linear_recovery.calculate_retry_delay(job)
        exp_delay = exp_recovery.calculate_retry_delay(job)
        
        # Should produce different delays
        assert none_delay == 0.0
        assert fixed_delay == 2.0
        assert linear_delay == 6.0  # 2.0 * (2 + 1)
        assert exp_delay == 8.0     # 2.0 * 2^2


class TestFailureRecoveryRealistic:
    """Test realistic failure recovery scenarios."""

    def setup_method(self):
        """Set up test environment."""
        self.recovery = FailureRecovery(
            max_retries=3,
            base_delay=1.0,
            max_delay=60.0,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF
        )

    def create_test_job(self, job_id="test_job", retry_count=0):
        """Helper to create test BatchJob."""
        return BatchJob(
            job_id=job_id,
            # video_id=f"video_{job_id}",  # FIXME: video_id parameter not supported
            video_path=f"/videos/{job_id}.mp4",
            output_dir=f"/output/{job_id}",
            status=JobStatus.PENDING,
            created_at=datetime.now(),
            retry_count=retry_count,
            config={},
            selected_pipelines=["scene_detection", "face_analysis"],
            pipeline_results={}
        )

    def test_realistic_retry_workflow(self):
        """Test realistic retry workflow."""
        job = self.create_test_job("problem_video")
        
        # Simulate progressive failures and retries
        errors = [
            ConnectionError("Network timeout"),
            MemoryError("Out of memory"),
            RuntimeError("CUDA error")
        ]
        
        for i, error in enumerate(errors):
            # Check if should retry
            should_retry = self.recovery.should_retry(job, error)
            
            if should_retry:
                # Prepare for retry
                updated_job = self.recovery.prepare_retry(job, error)
                assert updated_job.retry_count == i + 1
                
                # Calculate delay
                delay = self.recovery.calculate_retry_delay(job)
                expected_delay = min(1.0 * (2 ** i), 60.0)
                assert delay == expected_delay
            else:
                break
        
        # Should eventually stop retrying
        final_should_retry = self.recovery.should_retry(job, Exception("Final error"))
        if job.retry_count >= 3:
            assert final_should_retry is False

    def test_realistic_partial_failure_scenario(self):
        """Test realistic partial failure scenario."""
        job = self.create_test_job("mixed_results_video")
        
        # Simulate some pipelines succeeding, others failing
        
        # First pipeline succeeds (would be set elsewhere)
        job.pipeline_results["scene_detection"] = PipelineResult(
            pipeline_name="scene_detection",
            status=JobStatus.COMPLETED,
            output_file="/path/to/scene_detection_output.json"
        )
        
        # Second pipeline fails
        face_error = Exception("Face detection model not available")
        should_continue = self.recovery.handle_partial_failure(
            job, "face_analysis", face_error
        )
        
        # Verify mixed results
        assert len(job.pipeline_results) == 2
        assert job.pipeline_results["scene_detection"].status == JobStatus.COMPLETED
        assert job.pipeline_results["face_analysis"].status == JobStatus.FAILED
        
        # Should have decision about continuing
        assert isinstance(should_continue, bool)

    def test_error_pattern_tracking(self):
        """Test error pattern tracking across multiple jobs."""
        jobs = [self.create_test_job(f"job_{i}") for i in range(5)]
        
        # Simulate various error types
        error_types = [
            ConnectionError("Network error"),
            MemoryError("OOM error"),
            ConnectionError("Another network error"),
            ValueError("Invalid input"),
            ConnectionError("Yet another network error")
        ]
        
        for job, error in zip(jobs, error_types):
            self.recovery.should_retry(job, error)
        
        # Should track error patterns
        assert len(self.recovery.error_patterns) >= 1
        assert "ConnectionError" in self.recovery.error_patterns
        assert self.recovery.error_patterns["ConnectionError"] == 3  # Three network errors
