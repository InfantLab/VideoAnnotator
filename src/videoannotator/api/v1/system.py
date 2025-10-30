"""System management endpoints for VideoAnnotator API."""

import platform
import time
from datetime import datetime
from typing import Any

import psutil
from fastapi import APIRouter

from ...registry.pipeline_registry import get_registry
from ...version import __version__ as videoannotator_version
from ..database import check_database_health, get_database_info
from ..errors import APIError
from ..middleware.auth import is_auth_required

PROCESS_START_TIME = time.time()

router = APIRouter()


def _get_disk_usage_percent() -> float:
    """Get disk usage percentage, handling Windows vs Unix."""
    try:
        if platform.system() == "Windows":
            disk = psutil.disk_usage("C:/")
        else:
            disk = psutil.disk_usage("/")
        return (disk.used / disk.total * 100) if disk.total > 0 else 0.0
    except Exception:
        return 0.0


def _get_database_status() -> dict[str, Any]:
    """Get database status for health check."""
    try:
        is_healthy, message = check_database_health()
        if is_healthy:
            return {"status": "healthy", "message": message}
        else:
            return {"status": "unhealthy", "message": message}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get(
    "/health",
    summary="Comprehensive system health and resource monitoring",
    description="""
Returns detailed system health information including CPU, memory, disk usage,
GPU availability, database status, pipeline registry, and service health.

This endpoint provides comprehensive diagnostics useful for:
- Monitoring system resource usage
- Verifying GPU availability for ML pipelines
- Checking pipeline registry status
- Validating database connectivity
- Assessing overall system capacity

**Example Request:**
```bash
curl -X GET "http://localhost:18011/api/v1/system/health" \\
  -H "X-API-Key: your-api-key-here"
```

**Success Response (200 OK):**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-08T10:30:45.123456",
  "api_version": "1.2.0",
  "videoannotator_version": "1.2.0",
  "system": {
    "platform": "Linux-6.5.0-1025-azure-x86_64-with-glibc2.35",
    "python_version": "3.10.12",
    "cpu_count": 8,
    "cpu_percent": 15.3,
    "memory_percent": 42.5,
    "memory": {
      "total": 16777216000,
      "available": 9663676416,
      "percent": 42.5,
      "used": 7113539584,
      "free": 9663676416
    },
    "disk": {
      "total": 107374182400,
      "used": 45097156608,
      "free": 62277025792,
      "percent": 42.0
    }
  },
  "database": {
    "status": "healthy",
    "message": "Database is accessible",
    "writable": true
  },
  "gpu": {
    "available": true,
    "device_count": 1,
    "current_device": 0,
    "device_name": "NVIDIA GeForce RTX 3090",
    "compute_capability": "8.6",
    "memory_allocated": 512000000,
    "memory_reserved": 1024000000
  },
  "pipelines": {
    "total": 12,
    "names": [
      "openface3_identity",
      "whisper_transcription",
      "diarization_pyannote"
    ]
  },
  "services": {
    "database": {
      "status": "healthy",
      "message": "Database is accessible"
    },
    "job_queue": "embedded",
    "pipelines": "ready"
  },
  "security": {
    "auth_required": true
  },
  "uptime_seconds": 3672
}
```

**Unhealthy Response (200 OK with status unhealthy):**
```json
{
  "status": "unhealthy",
  "timestamp": "2025-01-08T10:30:45.123456",
  "error": "Failed to connect to database",
  "api_version": "1.2.0",
  "videoannotator_version": "1.2.0"
}
```

**GPU Not Available Response:**
```json
{
  "gpu": {
    "available": false,
    "reason": "CUDA not available"
  }
}
```

**GPU Compatibility Warning (old GPU detected):**
```json
{
  "gpu": {
    "available": true,
    "device_name": "NVIDIA GeForce GTX 1060 6GB",
    "compute_capability": "6.1",
    "compatibility_warning": "GPU compute capability 6.1 is below PyTorch 2.5.0 minimum requirement (7.0). GPU acceleration may not work. Consider downgrading PyTorch to 1.13.x for older GPUs or upgrading hardware.",
    "pytorch_version": "2.5.0",
    "min_compute_capability": 7.0
  }
}
```

**Note**: This endpoint is more resource-intensive than the basic `/health` endpoint
due to system metrics collection (CPU sampling, disk I/O). For lightweight health
checks, use the root `/health` endpoint instead.

The `uptime_seconds` field shows how long the API server has been running,
useful for monitoring restarts and stability.

The `gpu.compute_capability` field indicates the CUDA compute capability of the GPU,
which determines compatibility with PyTorch versions. Modern PyTorch (2.5+) requires
compute capability 7.0 or higher (Volta architecture). If your GPU is older, a
`compatibility_warning` will be included with recommended actions.
""",
)
async def detailed_health_check():
    """Detailed system health check with comprehensive diagnostics."""
    try:
        # Get system information
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        # Get disk usage - handle Windows vs Unix paths
        try:
            if platform.system() == "Windows":
                disk = psutil.disk_usage("C:/")
            else:
                disk = psutil.disk_usage("/")
        except Exception:
            # Fallback if disk check fails
            disk = type("MockDisk", (), {"total": 0, "used": 0, "free": 0})()

        # Get GPU information if available
        gpu_info = None
        try:
            import torch

            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                gpu_info = {
                    "available": True,
                    "device_count": device_count,
                    "current_device": torch.cuda.current_device(),
                    "device_name": torch.cuda.get_device_name(0)
                    if device_count > 0
                    else None,
                    "memory_allocated": torch.cuda.memory_allocated(0)
                    if device_count > 0
                    else 0,
                    "memory_reserved": torch.cuda.memory_reserved(0)
                    if device_count > 0
                    else 0,
                }

                # Add compute capability for compatibility checking
                if device_count > 0:
                    try:
                        props = torch.cuda.get_device_properties(0)
                        compute_capability = f"{props.major}.{props.minor}"
                        gpu_info["compute_capability"] = compute_capability

                        # Check PyTorch version compatibility
                        # PyTorch 2.5+ requires compute capability >= 7.0 (Volta architecture)
                        # PyTorch 2.0-2.4 requires >= 3.5 (Kepler architecture)
                        pytorch_version = torch.__version__.split("+")[
                            0
                        ]  # Strip +cuXXX suffix
                        major_minor = tuple(map(int, pytorch_version.split(".")[:2]))

                        min_compute_capability = 7.0 if major_minor >= (2, 5) else 3.5
                        current_compute_capability = float(compute_capability)

                        if current_compute_capability < min_compute_capability:
                            gpu_info["compatibility_warning"] = (
                                f"GPU compute capability {compute_capability} is below PyTorch {pytorch_version} "
                                f"minimum requirement ({min_compute_capability}). GPU acceleration may not work. "
                                f"Consider downgrading PyTorch to 1.13.x for older GPUs or upgrading hardware."
                            )
                            gpu_info["pytorch_version"] = pytorch_version
                            gpu_info["min_compute_capability"] = min_compute_capability
                    except Exception:
                        pass  # Silently skip if we can't get compute capability

            else:
                gpu_info = {"available": False, "reason": "CUDA not available"}
        except ImportError:
            gpu_info = {"available": False, "reason": "PyTorch not installed"}

        # Registry enrichment (non-fatal)
        reg = get_registry()
        try:
            reg.load()
            reg_pipelines = reg.list()
        except Exception:
            reg_pipelines = []

        uptime_seconds = int(time.time() - PROCESS_START_TIME)

        db_status = _get_database_status()
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": cpu_percent,
            # Backward compat alias expected by older tests
            "memory_percent": memory.percent,
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "used": memory.used,
                "free": memory.free,
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": (disk.used / disk.total) * 100 if disk.total > 0 else 0,
            },
        }

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            # Use single source-of-truth for API/package version to avoid drift
            "api_version": videoannotator_version,
            "videoannotator_version": videoannotator_version,
            "system": system_info,
            # Top-level database key for backward compatibility (older tests expect this)
            "database": db_status,
            "gpu": gpu_info,
            "pipelines": {
                "total": len(reg_pipelines),
                "names": [p.name for p in reg_pipelines][:20],  # cap list
            },
            "services": {
                "database": db_status,
                "job_queue": "embedded",  # in-process execution model
                "pipelines": "ready",
            },
            "security": {
                "auth_required": is_auth_required(),
            },
            "uptime_seconds": uptime_seconds,
        }

    except Exception as e:
        # If the health check fails, still report the current packaged version
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "api_version": videoannotator_version,
            "videoannotator_version": videoannotator_version,
            "security": {
                "auth_required": is_auth_required(),
            },
        }


@router.get("/metrics")
async def get_system_metrics():
    """Get system performance metrics.

    Returns:
        System performance and usage metrics
    """
    try:
        # Get database statistics
        db_info = get_database_info()
        db_stats = db_info.get("statistics", {})

        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "jobs_total": db_stats.get("total_jobs", 0),
                "jobs_active": db_stats.get("running_jobs", 0)
                + db_stats.get("pending_jobs", 0),
                "jobs_completed": db_stats.get("completed_jobs", 0),
                "jobs_failed": db_stats.get("failed_jobs", 0),
                "jobs_pending": db_stats.get("pending_jobs", 0),
                "jobs_running": db_stats.get("running_jobs", 0),
                "total_annotations": db_stats.get("total_annotations", 0),
                "api_requests_total": 0,  # TODO: Implement request counting
                "response_time_avg": 0.0,  # TODO: Implement response time tracking
                "pipeline_processing_time_avg": 0.0,  # TODO: Track pipeline performance
                "uptime_seconds": 0,  # TODO: Track server uptime
            },
            "performance": {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": _get_disk_usage_percent(),
            },
        }

    except Exception as e:
        raise APIError(
            status_code=500,
            code="METRICS_FAILED",
            message=f"Failed to get metrics: {e!s}",
            hint="Check server logs for details",
        ) from e


@router.get("/config")
async def get_system_config():
    """Get current system configuration.

    Returns:
        System configuration information (non-sensitive)
    """
    try:
        # TODO: Load and return current system configuration
        # Exclude sensitive information like API keys, database passwords, etc.

        # TODO: Load actual config when dependencies are fixed
        # from ...config import load_config
        # config = load_config()
        config: dict[str, Any] = {}

        # Filter out sensitive information
        safe_config = {}
        for key, value in config.items():
            if key not in ["database_url", "redis_url", "api_keys", "secrets"]:
                safe_config[key] = value

        return {
            "configuration": safe_config,
            "environment": "development",  # TODO: Get from environment variable
            "features": {
                "authentication": False,  # TODO: Check if auth is enabled
                "rate_limiting": False,  # TODO: Check if rate limiting is enabled
                "webhooks": False,  # TODO: Check if webhooks are enabled
                "async_processing": False,  # TODO: Check if Celery is configured
            },
        }

    except Exception as e:
        raise APIError(
            status_code=500,
            code="CONFIG_FAILED",
            message=f"Failed to get config: {e!s}",
            hint="Check server logs for details",
        ) from e


@router.get("/database")
async def get_database_info_endpoint():
    """Get database information and statistics.

    Returns:
        Database configuration and statistics
    """
    try:
        return get_database_info()

    except Exception as e:
        raise APIError(
            status_code=500,
            code="DATABASE_INFO_FAILED",
            message=f"Failed to get database info: {e!s}",
            hint="Check database connection and server logs",
        ) from e
