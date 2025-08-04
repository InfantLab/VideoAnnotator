"""
Failure recovery system for VideoAnnotator batch processing.

Provides robust error handling, retry mechanisms, and graceful degradation.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from datetime import datetime, timedelta

from .types import BatchJob, JobStatus, PipelineResult


class RetryStrategy(Enum):
    """Retry strategy options."""
    NONE = "none"
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"


class FailureRecovery:
    """Handles failure recovery and retry logic for batch processing."""
    
    def __init__(self, 
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 300.0,
                 strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF):
        """
        Initialize failure recovery system.
        
        Args:
            max_retries: Maximum number of retry attempts per job
            base_delay: Base delay in seconds for retry strategies
            max_delay: Maximum delay in seconds between retries
            strategy: Retry strategy to use
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)
        
        # Track retry statistics
        self.retry_stats: Dict[str, int] = {}
        self.error_patterns: Dict[str, int] = {}
    
    def should_retry(self, job: BatchJob, error: Exception) -> bool:
        """
        Determine if a job should be retried.
        
        Args:
            job: The failed job
            error: The exception that caused the failure
            
        Returns:
            True if job should be retried, False otherwise
        """
        # Check retry count
        if job.retry_count >= self.max_retries:
            self.logger.info(f"Job {job.job_id} exceeded max retries ({self.max_retries})")
            return False
        
        # Check for non-retryable errors
        if self._is_permanent_error(error):
            self.logger.info(f"Job {job.job_id} failed with permanent error: {type(error).__name__}")
            return False
        
        # Track error patterns
        error_type = type(error).__name__
        self.error_patterns[error_type] = self.error_patterns.get(error_type, 0) + 1
        
        return True
    
    def calculate_retry_delay(self, job: BatchJob) -> float:
        """
        Calculate delay before retry based on strategy.
        
        Args:
            job: The job being retried
            
        Returns:
            Delay in seconds
        """
        if self.strategy == RetryStrategy.NONE:
            return 0.0
        
        elif self.strategy == RetryStrategy.FIXED_DELAY:
            return min(self.base_delay, self.max_delay)
        
        elif self.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.base_delay * (job.retry_count + 1)
            return min(delay, self.max_delay)
        
        elif self.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.base_delay * (2 ** job.retry_count)
            return min(delay, self.max_delay)
        
        return self.base_delay
    
    def prepare_retry(self, job: BatchJob, error: Exception) -> BatchJob:
        """
        Prepare a job for retry.
        
        Args:
            job: The failed job
            error: The exception that caused the failure
            
        Returns:
            Updated job ready for retry
        """
        job.retry_count += 1
        job.status = JobStatus.PENDING  # Reset to pending for retry
        job.error_message = None  # Clear error message for clean retry
        
        # Clear partial results if needed
        if self._should_clear_partial_results(error):
            job.pipeline_results.clear()
        
        # Update retry statistics
        self.retry_stats[job.job_id] = job.retry_count
        
        self.logger.info(
            f"Preparing job {job.job_id} for retry {job.retry_count}/{self.max_retries} "
            f"after error: {error}"
        )
        
        return job
    
    def handle_partial_failure(self, job: BatchJob, failed_pipeline: str, error: Exception) -> bool:
        """
        Handle partial failure of a single pipeline within a job.
        
        Args:
            job: The job with partial failure
            failed_pipeline: Name of the failed pipeline
            error: The exception that caused the failure
            
        Returns:
            True if job should continue with other pipelines, False if job should fail completely
        """
        # Record pipeline failure
        job.pipeline_results[failed_pipeline] = PipelineResult(
            pipeline_name=failed_pipeline,
            status=JobStatus.FAILED,
            error_message=str(error)
        )
        
        # Check if error affects other pipelines
        if self._is_catastrophic_error(error):
            self.logger.error(
                f"Catastrophic error in {failed_pipeline} for job {job.job_id}, "
                f"failing entire job: {error}"
            )
            return False
        
        # Continue with graceful degradation
        self.logger.warning(
            f"Pipeline {failed_pipeline} failed for job {job.job_id}, "
            f"continuing with other pipelines: {error}"
        )
        return True
    
    def create_checkpoint(self, jobs: List[BatchJob], checkpoint_file: str) -> None:
        """
        Create a checkpoint of current job states.
        
        Args:
            jobs: List of current jobs
            checkpoint_file: Path to save checkpoint
        """
        import json
        
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'total_jobs': len(jobs),
            'jobs': [job.to_dict() for job in jobs],
            'retry_stats': self.retry_stats,
            'error_patterns': self.error_patterns,
        }
        
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
            self.logger.info(f"Created checkpoint with {len(jobs)} jobs")
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_file: str) -> Optional[List[BatchJob]]:
        """
        Load jobs from a checkpoint file.
        
        Args:
            checkpoint_file: Path to checkpoint file
            
        Returns:
            List of jobs if successful, None if failed
        """
        import json
        from pathlib import Path
        from ..batch.types import BatchJob
        
        if not Path(checkpoint_file).exists():
            self.logger.warning(f"Checkpoint file not found: {checkpoint_file}")
            return None
        
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            
            jobs = [BatchJob.from_dict(job_data) for job_data in data['jobs']]
            self.retry_stats = data.get('retry_stats', {})
            self.error_patterns = data.get('error_patterns', {})
            
            self.logger.info(f"Loaded {len(jobs)} jobs from checkpoint")
            return jobs
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def get_recovery_report(self) -> Dict[str, Any]:
        """Get detailed recovery and error statistics."""
        return {
            'retry_statistics': {
                'total_retries': sum(self.retry_stats.values()),
                'jobs_with_retries': len(self.retry_stats),
                'max_retries_allowed': self.max_retries,
                'retry_strategy': self.strategy.value,
                'retry_details': self.retry_stats.copy(),
            },
            'error_patterns': self.error_patterns.copy(),
            'recovery_config': {
                'max_retries': self.max_retries,
                'base_delay': self.base_delay,
                'max_delay': self.max_delay,
                'strategy': self.strategy.value,
            },
        }
    
    def _is_permanent_error(self, error: Exception) -> bool:
        """Check if an error is permanent and shouldn't be retried."""
        # Be more conservative - only truly permanent errors should not be retried
        error_str = str(error).lower()
        
        # Only these specific patterns are considered truly permanent
        permanent_patterns = [
            'invalid video format',
            'codec not supported',
            'invalid configuration',
            'malformed input',
        ]
        
        # Permission errors might be temporary (file lock, etc.)
        # File not found might be temporary (network mount, etc.)
        # Most other errors could be transient
        
        return any(pattern in error_str for pattern in permanent_patterns)
    
    def _is_catastrophic_error(self, error: Exception) -> bool:
        """Check if an error is catastrophic and affects the entire job."""
        catastrophic_errors = [
            MemoryError,
            SystemError,
            OSError,
        ]
        
        error_str = str(error).lower()
        catastrophic_messages = [
            'out of memory',
            'cuda out of memory',
            'disk full',
            'system error',
        ]
        
        return (
            type(error) in catastrophic_errors or
            any(msg in error_str for msg in catastrophic_messages)
        )
    
    def _should_clear_partial_results(self, error: Exception) -> bool:
        """Determine if partial results should be cleared before retry."""
        # Clear results for memory-related errors
        error_str = str(error).lower()
        clear_conditions = [
            'out of memory',
            'cuda out of memory',
            'corrupted',
            'invalid state',
        ]
        
        return any(condition in error_str for condition in clear_conditions)
