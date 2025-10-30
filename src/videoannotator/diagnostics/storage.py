"""Storage diagnostics for VideoAnnotator.

Checks disk space and write permissions for storage paths.

v1.3.0: Phase 11 - T072
"""

import tempfile
from pathlib import Path
from typing import Any

import psutil

from videoannotator.storage.config import get_storage_root
from videoannotator.utils.logging_config import get_logger

logger = get_logger("diagnostics")


def diagnose_storage() -> dict[str, Any]:
    """Run comprehensive storage diagnostics.

    Returns:
        Dictionary with storage diagnostic information:
        {
            "status": "ok" | "warning" | "error",
            "storage_root": "/app/storage/jobs",
            "disk_usage": {
                "total_gb": 500.0,
                "used_gb": 250.0,
                "free_gb": 250.0,
                "percent_used": 50.0
            },
            "writable": true,
            "errors": [],
            "warnings": []
        }
    """
    result: dict[str, Any] = {
        "status": "ok",
        "storage_root": None,
        "disk_usage": {},
        "writable": False,
        "errors": [],
        "warnings": [],
    }

    try:
        # Get storage root
        storage_root = get_storage_root()
        result["storage_root"] = str(storage_root)

        # Check if directory exists, create if needed
        if not storage_root.exists():
            try:
                storage_root.mkdir(parents=True, exist_ok=True)
                result["warnings"].append(f"Storage directory created: {storage_root}")
            except Exception as e:
                result["status"] = "error"
                result["errors"].append(f"Failed to create storage directory: {e!s}")
                return result

        # Check disk usage
        disk_info = _check_disk_usage(storage_root)
        result["disk_usage"] = disk_info

        # Check for low disk space
        if disk_info["percent_used"] > 90:
            result["status"] = "error"
            result["errors"].append(
                f"Critical: Disk usage at {disk_info['percent_used']:.1f}% "
                f"({disk_info['free_gb']:.1f} GB free)"
            )
        elif disk_info["percent_used"] > 80:
            result["warnings"].append(
                f"Low disk space: {disk_info['percent_used']:.1f}% used "
                f"({disk_info['free_gb']:.1f} GB free)"
            )
            if result["status"] == "ok":
                result["status"] = "warning"

        # Check write permissions
        writable = _check_write_permission(storage_root)
        result["writable"] = writable

        if not writable:
            result["status"] = "error"
            result["errors"].append(f"Storage directory not writable: {storage_root}")

    except Exception as e:
        result["status"] = "error"
        result["errors"].append(f"Storage diagnostic failed: {e!s}")
        logger.error(f"Storage diagnostic error: {e}", exc_info=True)

    return result


def _check_disk_usage(path: Path) -> dict[str, Any]:
    """Check disk usage for the given path.

    Args:
        path: Path to check disk usage for

    Returns:
        Dictionary with disk usage information
    """
    disk = psutil.disk_usage(str(path))

    return {
        "total_gb": round(disk.total / 1024**3, 2),
        "used_gb": round(disk.used / 1024**3, 2),
        "free_gb": round(disk.free / 1024**3, 2),
        "percent_used": round(disk.percent, 1),
    }


def _check_write_permission(path: Path) -> bool:
    """Check if the path is writable.

    Args:
        path: Path to check write permission for

    Returns:
        True if writable, False otherwise
    """
    try:
        # Try to create a temporary file
        with tempfile.NamedTemporaryFile(
            dir=path, delete=True, prefix=".videoannotator_test_"
        ):
            pass
        return True
    except (PermissionError, OSError) as e:
        logger.warning(f"Write permission check failed for {path}: {e}")
        return False
