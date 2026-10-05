"""Storage configuration for VideoAnnotator persistent job storage.

This module provides configuration for persistent job storage paths,
supporting both local filesystem and future cloud storage backends.

Environment Variables:
    STORAGE_ROOT: Root directory for persistent job storage
                  Default: `jobs` in the per-user data directory, next to the
                  database (it was ./storage/jobs, under whatever directory the
                  server started in, before v1.6.0)
"""

import os
from pathlib import Path

import yaml


def get_storage_root() -> Path:
    """Get the root directory for persistent job storage.

    Priority:
    1. STORAGE_ROOT environment variable
    2. 'storage.root_path' in configs/default.yaml
    3. Default: `jobs` in the per-user data directory

    Returns:
        Path: Absolute path to the storage root directory
    """
    # 1. Check environment variable
    env_root = os.getenv("STORAGE_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    # 2. Check default config file
    try:
        # Assuming running from repo root or src/..
        # Try to find configs/default.yaml
        repo_root = Path(__file__).parent.parent.parent.parent
        config_path = repo_root / "configs" / "default.yaml"

        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
                if config and "storage" in config and "root_path" in config["storage"]:
                    return Path(config["storage"]["root_path"]).expanduser().resolve()
    except Exception:
        # Fallback if config parsing fails
        pass

    # 3. Default: beside the database, so jobs don't depend on where the server
    # was started (spec 016 did the same for weights, logs and the database).
    from videoannotator.models_dir import user_data_dir

    return (user_data_dir() / "jobs").resolve()


def legacy_storage_root() -> Path | None:
    """`./storage/jobs`, the pre-v1.6.0 default, if it holds jobs and isn't the
    storage root. Those jobs still open: each records its own folder."""
    old = Path("storage/jobs").resolve()
    try:
        if old == get_storage_root() or not old.is_dir():
            return None
        return old if any(child.is_dir() for child in old.iterdir()) else None
    except OSError:
        return None


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
    "legacy_storage_root",
    "get_storage_root",
]
