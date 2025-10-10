"""VideoAnnotator Batch Processing.

This module provides robust batch processing capabilities for VideoAnnotator,
including job queue management, failure recovery, and progress tracking."""

# Import only types by default to prevent heavy import chain during testing
from .types import BatchJob, BatchReport, BatchStatus, JobStatus

# Heavy imports commented out to prevent pytest collection hangs
# Uncomment when needed for actual batch processing
# from .batch_orchestrator import BatchOrchestrator
# from .progress_tracker import ProgressTracker
# from .recovery import FailureRecovery

__all__ = [
    "BatchJob",
    "BatchReport",
    "BatchStatus",
    "JobStatus",
    # "BatchOrchestrator",
    # "ProgressTracker",
    # "FailureRecovery",
]
