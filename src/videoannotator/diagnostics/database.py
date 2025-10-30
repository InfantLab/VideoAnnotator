"""Database diagnostics for VideoAnnotator.

Checks database connectivity and schema version.

v1.3.0: Phase 11 - T073
"""

import os
from typing import Any

from videoannotator.api.database import check_database_health
from videoannotator.utils.logging_config import get_logger

logger = get_logger("diagnostics")


def diagnose_database() -> dict[str, Any]:
    """Run comprehensive database diagnostics.

    Returns:
        Dictionary with database diagnostic information:
        {
            "status": "ok" | "warning" | "error",
            "connected": true,
            "database_path": "/app/storage/videoannotator.db",
            "schema_version": "1.3.0",
            "job_count": 42,
            "errors": [],
            "warnings": []
        }
    """
    result: dict[str, Any] = {
        "status": "ok",
        "connected": False,
        "database_path": None,
        "schema_version": None,
        "job_count": None,
        "errors": [],
        "warnings": [],
    }

    try:
        # Get database path from DATABASE_URL
        database_url = os.getenv("DATABASE_URL", "sqlite:///./videoannotator.db")
        # Extract path from sqlite URL
        if database_url.startswith("sqlite:///"):
            db_path = database_url[10:]  # Remove "sqlite:///"
        else:
            db_path = database_url
        result["database_path"] = db_path

        # Check database connectivity
        is_healthy, message = check_database_health()
        result["connected"] = is_healthy

        if not is_healthy:
            result["status"] = "error"
            result["errors"].append(f"Database connection failed: {message}")
            return result

        # Get schema version
        schema_version = _get_schema_version()
        result["schema_version"] = schema_version

        # Get job count
        job_count = _get_job_count()
        result["job_count"] = job_count

    except Exception as e:
        result["status"] = "error"
        result["errors"].append(f"Database diagnostic failed: {e!s}")
        logger.error(f"Database diagnostic error: {e}", exc_info=True)

    return result


def _get_schema_version() -> str | None:
    """Get the database schema version.

    Returns:
        Schema version string or None if not available
    """
    try:
        from sqlalchemy import text

        from videoannotator.database.database import SessionLocal

        with SessionLocal() as session:
            # Try to query for schema version from alembic_version table
            try:
                result = session.execute(
                    text("SELECT version_num FROM alembic_version")
                )
                row = result.fetchone()
                if row:
                    return row[0]
            except Exception:
                # Table might not exist in older versions
                pass

            # Fallback: check if tables exist to infer version
            result = session.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            )
            tables = {row[0] for row in result.fetchall()}

            if "jobs" in tables and "api_keys" in tables:
                return "1.3.0 (inferred)"
            elif "jobs" in tables:
                return "1.2.0 (inferred)"
            else:
                return "unknown"

    except Exception as e:
        logger.warning(f"Failed to get schema version: {e}")
        return None


def _get_job_count() -> int | None:
    """Get the total number of jobs in the database.

    Returns:
        Job count or None if not available
    """
    try:
        from videoannotator.database.database import SessionLocal
        from videoannotator.storage.models import Job

        with SessionLocal() as session:
            count = session.query(Job).count()
            return count
    except Exception as e:
        logger.warning(f"Failed to get job count: {e}")
        return None
