"""Storage cleanup utilities for automatic job retention management.

This module provides safe deletion of old job data with multiple safety checks
and audit logging. Cleanup is disabled by default and requires explicit configuration.

v1.3.0: Added automatic storage cleanup with retention policy.
"""

import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from videoannotator.config_env import STORAGE_RETENTION_DAYS, STORAGE_BASE_DIR
from videoannotator.database.models import Job, JobStatus
from videoannotator.storage.file_backend import FileStorageBackend
from videoannotator.utils.logging_config import get_logger

logger = get_logger(__name__)

# Terminal states that are safe to delete (string values)
TERMINAL_STATES = [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]


class CleanupResult:
    """Result of a cleanup operation."""

    def __init__(self) -> None:
        """Initialize cleanup result."""
        self.jobs_found = 0
        self.jobs_deleted = 0
        self.jobs_skipped = 0
        self.bytes_freed = 0
        self.errors: list[str] = []
        self.deleted_jobs: list[str] = []
        self.skipped_jobs: list[dict[str, Any]] = []

    def add_deleted(self, job_id: str, size: int) -> None:
        """Record a deleted job.

        Args:
            job_id: Job identifier
            size: Bytes freed
        """
        self.jobs_deleted += 1
        self.bytes_freed += size
        self.deleted_jobs.append(job_id)

    def add_skipped(self, job_id: str, reason: str) -> None:
        """Record a skipped job.

        Args:
            job_id: Job identifier
            reason: Why it was skipped
        """
        self.jobs_skipped += 1
        self.skipped_jobs.append({"job_id": job_id, "reason": reason})

    def add_error(self, job_id: str, error: str) -> None:
        """Record an error.

        Args:
            job_id: Job identifier
            error: Error message
        """
        self.errors.append(f"{job_id}: {error}")

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "jobs_found": self.jobs_found,
            "jobs_deleted": self.jobs_deleted,
            "jobs_skipped": self.jobs_skipped,
            "bytes_freed": self.bytes_freed,
            "bytes_freed_mb": round(self.bytes_freed / 1024 / 1024, 2),
            "deleted_jobs": self.deleted_jobs,
            "skipped_jobs": self.skipped_jobs,
            "errors": self.errors,
        }


def is_cleanup_enabled() -> bool:
    """Check if cleanup is enabled via configuration.

    Returns:
        True if STORAGE_RETENTION_DAYS is set and > 0
    """
    return STORAGE_RETENTION_DAYS is not None and STORAGE_RETENTION_DAYS > 0


def find_old_jobs(retention_days: int | None = None) -> list[Job]:
    """Find jobs eligible for cleanup based on retention policy.

    Args:
        retention_days: Days to retain (overrides config if provided)

    Returns:
        List of jobs that can be deleted

    Raises:
        ValueError: If cleanup is disabled and no override provided
    """
    days = retention_days if retention_days is not None else STORAGE_RETENTION_DAYS

    if days is None or days <= 0:
        raise ValueError(
            "Cleanup is disabled (STORAGE_RETENTION_DAYS not set or <= 0)"
        )

    cutoff_date = datetime.now() - timedelta(days=days)

    # Query jobs in terminal states older than retention period
    from videoannotator.database.session import get_session

    with get_session() as session:
        jobs = (
            session.query(Job)
            .filter(Job.status.in_(TERMINAL_STATES))
            .filter(Job.completed_at.isnot(None))
            .filter(Job.completed_at < cutoff_date)
            .all()
        )

        # Detach from session to avoid lazy loading issues
        for job in jobs:
            session.expunge(job)

        return jobs


def verify_job_safe_to_delete(job: Job) -> tuple[bool, str]:
    """Verify a job is safe to delete with multiple safety checks.

    Args:
        job: Job to verify

    Returns:
        Tuple of (is_safe, reason_if_not_safe)
    """
    # Check 1: Must be in terminal state (status is stored as string)
    if job.status not in TERMINAL_STATES:
        return False, f"Job not in terminal state (status={job.status})"

    # Check 2: Must have completion timestamp
    if job.completed_at is None:
        return False, "Job has no completion timestamp"

    # Check 3: Completion must be in the past (naive comparison for test compatibility)
    completed_at = job.completed_at.replace(tzinfo=None) if hasattr(job.completed_at, 'replace') else job.completed_at
    if completed_at > datetime.now():
        return False, "Job completion timestamp is in the future"

    return True, ""


def get_job_storage_size(job_id: str, storage: FileStorageBackend) -> int:
    """Calculate total storage size for a job.

    Args:
        job_id: Job identifier
        storage: Storage backend

    Returns:
        Total bytes used by job
    """
    total_size = 0

    try:
        job_dir = storage._get_job_dir(job_id)
        if job_dir.exists():
            for item in job_dir.rglob("*"):
                if item.is_file():
                    total_size += item.stat().st_size
    except Exception as e:
        logger.warning(f"Error calculating size for job {job_id}: {e}")

    return total_size


def cleanup_job_storage(
    job: Job, storage: FileStorageBackend, dry_run: bool = True
) -> tuple[int, str | None]:
    """Delete job storage with audit logging.

    Args:
        job: Job to delete
        storage: Storage backend
        dry_run: If True, only simulate deletion

    Returns:
        Tuple of (bytes_freed, error_message)
    """
    # Job.id is a UUID, convert to string
    job_id = str(job.id)

    # Safety check
    is_safe, reason = verify_job_safe_to_delete(job)
    if not is_safe:
        return 0, f"Safety check failed: {reason}"

    # Get storage size before deletion
    size = get_job_storage_size(job_id, storage)

    if dry_run:
        logger.info(f"[DRY-RUN] Would delete job {job_id} ({size} bytes)")
        return size, None

    # Actual deletion
    try:
        job_dir = storage._get_job_dir(job_id)

        if not job_dir.exists():
            logger.warning(f"Job directory does not exist: {job_dir}")
            return 0, "Directory not found"

        # Delete the entire job directory
        shutil.rmtree(job_dir)

        # Audit log
        logger.info(
            f"[CLEANUP] Deleted job {job_id}: "
            f"status={job.status}, "
            f"completed={job.completed_at.isoformat() if hasattr(job.completed_at, 'isoformat') else job.completed_at}, "
            f"freed={size} bytes"
        )

        return size, None

    except Exception as e:
        error_msg = f"Failed to delete job {job_id}: {e}"
        logger.error(error_msg)
        return 0, error_msg


def cleanup_old_jobs(
    retention_days: int | None = None,
    dry_run: bool = True,
    storage: FileStorageBackend | None = None,
) -> CleanupResult:
    """Clean up old job data based on retention policy.

    Args:
        retention_days: Days to retain (overrides config)
        dry_run: If True, only simulate deletion
        storage: Storage backend (creates default if None)

    Returns:
        CleanupResult with statistics

    Raises:
        ValueError: If cleanup is disabled and no override provided
    """
    result = CleanupResult()

    # Initialize storage backend
    if storage is None:
        storage = FileStorageBackend(base_dir=STORAGE_BASE_DIR)

    # Find eligible jobs
    try:
        jobs = find_old_jobs(retention_days)
        result.jobs_found = len(jobs)

        logger.info(
            f"Found {len(jobs)} jobs eligible for cleanup "
            f"(retention_days={retention_days or STORAGE_RETENTION_DAYS}, "
            f"dry_run={dry_run})"
        )

    except ValueError as e:
        logger.error(f"Cleanup disabled: {e}")
        result.add_error("config", str(e))
        return result

    # Process each job
    for job in jobs:
        size, error = cleanup_job_storage(job, storage, dry_run)
        job_id_str = str(job.id)

        if error:
            result.add_error(job_id_str, error)
            result.add_skipped(job_id_str, error)
        else:
            if dry_run:
                result.add_skipped(
                    job_id_str, f"Dry-run mode (would free {size} bytes)"
                )
            else:
                result.add_deleted(job_id_str, size)

    # Summary logging
    if dry_run:
        logger.info(
            f"[DRY-RUN] Would delete {result.jobs_deleted + result.jobs_skipped} jobs, "
            f"freeing {result.bytes_freed / 1024 / 1024:.2f} MB"
        )
    else:
        logger.info(
            f"[CLEANUP] Deleted {result.jobs_deleted} jobs, "
            f"freed {result.bytes_freed / 1024 / 1024:.2f} MB, "
            f"skipped {result.jobs_skipped}, "
            f"errors {len(result.errors)}"
        )

    return result
