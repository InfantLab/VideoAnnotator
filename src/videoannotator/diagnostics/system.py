"""System diagnostics for VideoAnnotator.

Checks Python version, FFmpeg availability, and OS information.

v1.3.0: Phase 11 - T070
"""

import platform
import shutil
import subprocess
import sys
from typing import Any

from videoannotator.utils.logging_config import get_logger

logger = get_logger("diagnostics")


def diagnose_system() -> dict[str, Any]:
    """Run comprehensive system diagnostics.

    Returns:
        Dictionary with system diagnostic information:
        {
            "status": "ok" | "warning" | "error",
            "python": {
                "version": "3.12.3",
                "executable": "/usr/bin/python3",
                "platform": "linux"
            },
            "ffmpeg": {
                "installed": true,
                "version": "4.4.2",
                "path": "/usr/bin/ffmpeg"
            },
            "os": {
                "system": "Linux",
                "release": "6.8.0",
                "machine": "x86_64"
            },
            "errors": [],
            "warnings": []
        }
    """
    result: dict[str, Any] = {
        "status": "ok",
        "python": {},
        "ffmpeg": {},
        "os": {},
        "errors": [],
        "warnings": [],
    }

    try:
        # Check Python version
        python_info = _check_python()
        result["python"] = python_info

        # Check FFmpeg
        ffmpeg_info = _check_ffmpeg()
        result["ffmpeg"] = ffmpeg_info

        if not ffmpeg_info["installed"]:
            result["errors"].append("FFmpeg is not installed")
            result["status"] = "error"

        # Check OS information
        os_info = _check_os()
        result["os"] = os_info

        # Check Python version compatibility
        if sys.version_info < (3, 10):
            result["warnings"].append(
                f"Python {sys.version_info.major}.{sys.version_info.minor} "
                "is below recommended version 3.10+"
            )
            if result["status"] == "ok":
                result["status"] = "warning"

    except Exception as e:
        result["status"] = "error"
        result["errors"].append(f"System diagnostic failed: {e!s}")
        logger.error(f"System diagnostic error: {e}", exc_info=True)

    return result


def _check_python() -> dict[str, Any]:
    """Check Python installation and version.

    Returns:
        Dictionary with Python information
    """
    return {
        "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "version_info": {
            "major": sys.version_info.major,
            "minor": sys.version_info.minor,
            "micro": sys.version_info.micro,
        },
        "executable": sys.executable,
        "platform": sys.platform,
    }


def _check_ffmpeg() -> dict[str, Any]:
    """Check FFmpeg installation and version.

    Returns:
        Dictionary with FFmpeg information
    """
    ffmpeg_path = shutil.which("ffmpeg")

    if not ffmpeg_path:
        return {
            "installed": False,
            "version": None,
            "path": None,
        }

    try:
        # Get FFmpeg version
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

        version_line = result.stdout.split("\n")[0] if result.stdout else ""
        # Parse version from "ffmpeg version X.Y.Z" format
        version = None
        if "version" in version_line:
            parts = version_line.split()
            if len(parts) >= 3:
                version = parts[2]

        return {
            "installed": True,
            "version": version,
            "path": ffmpeg_path,
        }
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        logger.warning(f"Failed to get FFmpeg version: {e}")
        return {
            "installed": True,  # Binary exists but version check failed
            "version": "unknown",
            "path": ffmpeg_path,
        }


def _check_os() -> dict[str, Any]:
    """Check operating system information.

    Returns:
        Dictionary with OS information
    """
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
    }
