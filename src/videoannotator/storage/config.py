"""Storage configuration for VideoAnnotator persistent job storage.

This module provides configuration for persistent job storage paths,
supporting both local filesystem and future cloud storage backends.

Environment Variables:
    STORAGE_ROOT: Root directory for persistent job storage
                  Default: ./storage/jobs
"""

import os
from pathlib import Path


def get_storage_root() -> Path:
    """Get the root directory for persistent job storage.

    Returns:
        Path: Absolute path to the storage root directory

    Environment Variables:
        STORAGE_ROOT: Override the default storage root path

    Examples:
        >>> root = get_storage_root()
        >>> print(root)
        /app/storage/jobs

        >>> os.environ['STORAGE_ROOT'] = '/mnt/shared/storage'
        >>> root = get_storage_root()
        >>> print(root)
        /mnt/shared/storage
    """
    storage_root_str = os.getenv("STORAGE_ROOT", "./storage/jobs")
    storage_root = Path(storage_root_str).expanduser().resolve()
    return storage_root


def get_job_storage_path(job_id: str) -> Path:
    """Get the storage directory path for a specific job.

    Creates a job-specific subdirectory under the storage root for
    organizing job artifacts, logs, and results.

    Args:
        job_id: Unique identifier for the job

    Returns:
        Path: Absolute path to the job's storage directory

    Examples:
        >>> path = get_job_storage_path("abc-123")
        >>> print(path)
        /app/storage/jobs/abc-123

        >>> # Path is consistent across calls
        >>> path1 = get_job_storage_path("test-job")
        >>> path2 = get_job_storage_path("test-job")
        >>> assert path1 == path2

    Notes:
        - Directory is NOT created automatically by this function
        - Path is deterministic based on job_id
        - Safe for concurrent access (different jobs = different paths)
    """
    storage_root = get_storage_root()
    job_path = storage_root / job_id
    return job_path


def ensure_job_storage_path(job_id: str) -> Path:
    """Ensure the storage directory for a job exists.

    Creates the job storage directory and any parent directories if they
    don't already exist.

    Args:
        job_id: Unique identifier for the job

    Returns:
        Path: Absolute path to the created job storage directory

    Raises:
        OSError: If directory creation fails due to permissions or disk space

    Examples:
        >>> path = ensure_job_storage_path("new-job")
        >>> assert path.exists()
        >>> assert path.is_dir()

    Notes:
        - Creates parent directories if needed (like mkdir -p)
        - Idempotent: safe to call multiple times
        - Sets directory permissions based on umask
    """
    job_path = get_job_storage_path(job_id)
    job_path.mkdir(parents=True, exist_ok=True)
    return job_path


# Configuration constants
STORAGE_ROOT = get_storage_root()

__all__ = [
    "STORAGE_ROOT",
    "ensure_job_storage_path",
    "get_job_storage_path",
    "get_storage_root",
]
