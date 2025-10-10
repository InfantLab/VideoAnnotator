"""
Database configuration and dependency injection for VideoAnnotator API.

This module handles database backend selection and provides dependency injection
for FastAPI endpoints. It supports both SQLite (default) and PostgreSQL backends
based on environment configuration.
"""

import logging
import os
from functools import lru_cache
from pathlib import Path

from storage.base import StorageBackend
from storage.sqlite_backend import SQLiteStorageBackend

logger = logging.getLogger(__name__)


@lru_cache
def get_storage_backend() -> StorageBackend:
    """
    Get storage backend based on configuration.

    This function uses environment variables to determine which backend to use:
    - DATABASE_URL: If set and starts with "postgresql://", use PostgreSQL
    - VIDEOANNOTATOR_DB_PATH: Custom SQLite database path
    - Default: SQLite database in current directory (./videoannotator.db)

    Returns:
        Configured storage backend instance
    """
    database_url = os.environ.get("DATABASE_URL")

    if database_url:
        if database_url.startswith("postgresql://") or database_url.startswith(
            "postgres://"
        ):
            logger.info("[DATABASE] Using PostgreSQL backend")
            try:
                # Future: Import PostgreSQL backend when implemented
                # from ..storage.postgresql_backend import PostgreSQLStorageBackend
                # return PostgreSQLStorageBackend(database_url)
                logger.error("[ERROR] PostgreSQL backend not yet implemented")
                logger.info("[FALLBACK] Using SQLite backend instead")
            except ImportError as e:
                logger.error(f"[ERROR] PostgreSQL backend not available: {e}")
                logger.info("[FALLBACK] Using SQLite backend instead")

        elif database_url.startswith("sqlite://"):
            # Extract path from sqlite:///path/to/db.db format
            db_path = database_url.replace("sqlite:///", "")
            logger.info(f"[DATABASE] Using SQLite backend: {db_path}")
            return SQLiteStorageBackend(Path(db_path))

    # Check for custom SQLite path
    db_path_env = os.environ.get("VIDEOANNOTATOR_DB_PATH")
    if db_path_env:
        logger.info(f"[DATABASE] Using custom SQLite path: {db_path_env}")
        return SQLiteStorageBackend(Path(db_path_env))

    # Default: SQLite in current directory
    default_path = Path.cwd() / "videoannotator.db"
    logger.info(f"[DATABASE] Using default SQLite database: {default_path}")
    return SQLiteStorageBackend(default_path)


def get_database_info() -> dict:
    """
    Get information about the current database configuration.

    Returns:
        Dictionary with database configuration details
    """
    storage = get_storage_backend()
    stats = storage.get_stats()

    return {
        "backend_type": stats.get("backend", "unknown"),
        "connection_info": {
            "database_path": stats.get("database_path"),
            "database_url": stats.get("database_url"),
            "database_size_mb": stats.get("database_size_mb", 0),
        },
        "statistics": {
            "total_jobs": stats.get("total_jobs", 0),
            "pending_jobs": stats.get("pending_jobs", 0),
            "running_jobs": stats.get("running_jobs", 0),
            "completed_jobs": stats.get("completed_jobs", 0),
            "failed_jobs": stats.get("failed_jobs", 0),
            "total_annotations": stats.get("total_annotations", 0),
        },
        "schema_version": stats.get("schema_version", "unknown"),
    }


def reset_storage_backend():
    """
    Clear the cached storage backend.

    This forces get_storage_backend() to create a new instance on next call.
    Useful for testing or when configuration changes.
    """
    get_storage_backend.cache_clear()


# Database health check
def check_database_health() -> tuple[bool, str]:
    """
    Check if the database is healthy and accessible.

    Returns:
        Tuple of (is_healthy, status_message)
    """
    try:
        storage = get_storage_backend()

        # Try to get statistics (basic database operation)
        stats = storage.get_stats()

        if "error" in stats:
            return False, f"Database error: {stats['error']}"

        return (
            True,
            f"Database healthy - {stats['total_jobs']} jobs in {stats['backend']} backend",
        )

    except Exception as e:
        logger.error(f"[ERROR] Database health check failed: {e}")
        return False, f"Database health check failed: {e!s}"


# Context manager for manual database operations
class DatabaseSession:
    """
    Context manager for direct database operations.

    Provides access to the underlying database session for complex queries
    that aren't covered by the StorageBackend interface.
    """

    def __init__(self):
        self.storage = get_storage_backend()
        self.session = None

    def __enter__(self):
        if hasattr(self.storage, "SessionLocal"):
            self.session = self.storage.SessionLocal()
            return self.session
        else:
            raise NotImplementedError(
                "Direct session access not supported for this backend"
            )

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            if exc_type:
                self.session.rollback()
            else:
                self.session.commit()
            self.session.close()


# Environment configuration helpers
def set_database_path(path: Path):
    """Set database path via environment variable."""
    os.environ["VIDEOANNOTATOR_DB_PATH"] = str(path)
    reset_storage_backend()


def set_database_url(url: str):
    """Set database URL via environment variable."""
    os.environ["DATABASE_URL"] = url
    reset_storage_backend()


def get_current_database_path() -> Path:
    """
    Get the current database file path (SQLite only).

    Returns:
        Path to current database file

    Raises:
        ValueError: If not using SQLite backend
    """
    storage = get_storage_backend()
    if hasattr(storage, "database_path"):
        return storage.database_path
    else:
        raise ValueError("Current backend does not use a database file")


# Development and testing utilities
def create_test_database() -> SQLiteStorageBackend:
    """
    Create a temporary in-memory SQLite database for testing.

    Returns:
        SQLite storage backend using in-memory database
    """
    import tempfile

    # Create temporary database file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    temp_file.close()

    return SQLiteStorageBackend(Path(temp_file.name))


def backup_database(backup_path: Path) -> bool:
    """
    Backup the current database to specified location.

    Args:
        backup_path: Where to save the backup

    Returns:
        True if backup successful, False otherwise
    """
    try:
        current_path = get_current_database_path()

        if current_path.exists():
            import shutil

            shutil.copy2(current_path, backup_path)
            logger.info(f"[BACKUP] Database backed up to: {backup_path}")
            return True
        else:
            logger.warning(f"[BACKUP] Database file not found: {current_path}")
            return False

    except Exception as e:
        logger.error(f"[ERROR] Database backup failed: {e}")
        return False
