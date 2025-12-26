"""Health check endpoints for VideoAnnotator API.

Provides both basic liveness checks and detailed system diagnostics.

v1.3.0: Phase 11 - T068
"""

from datetime import UTC, datetime
from typing import Any

import psutil
from fastapi import APIRouter, Query, Response, status

from videoannotator.utils.logging_config import get_logger

from ...registry.pipeline_registry import get_registry
from ...storage.config import get_storage_root
from ...version import __version__
from ..database import check_database_health

logger = get_logger("api")
router = APIRouter()


def _check_gpu_status() -> dict[str, Any]:
    """Check GPU availability and status.

    Returns:
        Dictionary with GPU information
    """
    try:
        import torch

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            devices = []

            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_reserved = torch.cuda.memory_reserved(i)
                memory_total = props.total_memory

                devices.append(
                    {
                        "name": props.name,
                        "memory_total_gb": round(memory_total / 1024**3, 2),
                        "memory_allocated_gb": round(memory_allocated / 1024**3, 2),
                        "memory_reserved_gb": round(memory_reserved / 1024**3, 2),
                        "memory_free_gb": round(
                            (memory_total - memory_reserved) / 1024**3, 2
                        ),
                    }
                )

            return {
                "available": True,
                "device_count": device_count,
                "devices": devices,
            }
        else:
            return {"available": False, "reason": "CUDA not available"}
    except ImportError:
        return {"available": False, "reason": "PyTorch not installed"}
    except Exception as e:
        return {"available": False, "reason": str(e)}


def _check_storage_status() -> dict[str, Any]:
    """Check storage availability and disk space.

    Returns:
        Dictionary with storage information
    """
    try:
        # Get disk usage for storage directory
        storage_root = get_storage_root()
        disk = psutil.disk_usage(str(storage_root))

        total_gb = disk.total / 1024**3
        used_gb = disk.used / 1024**3
        free_gb = disk.free / 1024**3
        percent_used = disk.percent

        result = {
            "status": "ok",
            "total_gb": round(total_gb, 2),
            "used_gb": round(used_gb, 2),
            "free_gb": round(free_gb, 2),
            "percent_used": round(percent_used, 1),
        }

        # Add warning if low on space
        if free_gb < total_gb * 0.05:  # Less than 5% free
            result["status"] = "warning"
            result["warning"] = "Less than 5% free space remaining"
        elif free_gb < total_gb * 0.10:  # Less than 10% free
            result["status"] = "warning"
            result["warning"] = "Less than 10% free space remaining"

        return result
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _check_database_status() -> dict[str, Any]:
    """Check database connectivity and status.

    Returns:
        Dictionary with database information
    """
    try:
        is_healthy, message = check_database_health()

        if is_healthy:
            # Try to get job count
            try:
                from ..database import get_storage_backend

                storage = get_storage_backend()
                all_jobs = storage.list_jobs()
                jobs_count = len(all_jobs)
            except Exception:
                jobs_count = None

            return {
                "status": "ok",
                "message": message,
                "jobs_count": jobs_count,
            }
        else:
            return {"status": "error", "error": message}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _check_registry_status() -> dict[str, Any]:
    """Check pipeline registry status.

    Returns:
        Dictionary with registry information
    """
    try:
        registry = get_registry()
        pipelines = registry.list()
        pipeline_names = [p.name for p in pipelines]

        return {
            "status": "ok",
            "pipelines_loaded": len(pipeline_names),
            "pipeline_names": pipeline_names,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.get(
    "/health",
    summary="Health check with optional detailed diagnostics",
    description="""
Check API server liveness and optionally retrieve detailed system status.

**Basic mode** (default): Returns simple 200 OK for load balancers.
Fast response, minimal overhead.

**Detailed mode** (?detailed=true): Returns comprehensive diagnostics including:
- Database connectivity and job count
- Storage disk space and warnings
- GPU availability and memory
- Pipeline registry status

Detailed mode returns 503 status if critical subsystems are unhealthy.

**Example Requests:**

Basic (fast):
```bash
curl -X GET "http://localhost:18011/api/v1/health"
```

Detailed (diagnostic):
```bash
curl -X GET "http://localhost:18011/api/v1/health?detailed=true" \\
  -H "X-API-Key: your-api-key-here"
```
""",
    responses={
        200: {
            "description": "Service is healthy",
            "content": {
                "application/json": {
                    "examples": {
                        "basic": {
                            "summary": "Basic health check (fast)",
                            "value": {
                                "status": "ok",
                                "version": "1.3.0",
                                "timestamp": "2025-10-22T12:00:00.000Z",
                            },
                        },
                        "detailed": {
                            "summary": "Detailed health check",
                            "value": {
                                "status": "ok",
                                "version": "1.3.0",
                                "timestamp": "2025-10-22T12:00:00.000Z",
                                "details": {
                                    "database": {
                                        "status": "ok",
                                        "message": "Database is accessible",
                                        "jobs_count": 42,
                                    },
                                    "storage": {
                                        "status": "ok",
                                        "free_gb": 128.5,
                                        "used_gb": 15.2,
                                        "total_gb": 143.7,
                                        "percent_used": 10.6,
                                    },
                                    "gpu": {
                                        "available": True,
                                        "device_count": 1,
                                        "devices": [
                                            {
                                                "name": "NVIDIA GeForce RTX 3080",
                                                "memory_total_gb": 10.0,
                                                "memory_free_gb": 8.5,
                                            }
                                        ],
                                    },
                                    "registry": {
                                        "status": "ok",
                                        "pipelines_loaded": 4,
                                        "pipeline_names": [
                                            "face",
                                            "person",
                                            "audio",
                                            "scene",
                                        ],
                                    },
                                },
                            },
                        },
                    }
                }
            },
        },
        503: {
            "description": "Service unhealthy (detailed mode only)",
            "content": {
                "application/json": {
                    "example": {
                        "status": "unhealthy",
                        "version": "1.3.0",
                        "timestamp": "2025-10-22T12:00:00.000Z",
                        "details": {
                            "database": {
                                "status": "error",
                                "error": "Connection timeout after 5s",
                            },
                            "storage": {"status": "ok"},
                            "gpu": {"available": True},
                            "registry": {"status": "ok"},
                        },
                    }
                }
            },
        },
    },
)
async def health_check(
    response: Response,
    detailed: bool = Query(
        False, description="Include detailed diagnostics (slower response)"
    ),
):
    """Health check endpoint with optional detailed diagnostics.

    Args:
        response: FastAPI response object for setting status code
        detailed: If True, return detailed system diagnostics

    Returns:
        Health status dictionary
    """
    timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    # Basic mode: fast liveness check
    if not detailed:
        logger.debug("[HEALTH] Basic health check requested")
        return {
            "status": "ok",
            "version": __version__,
            "timestamp": timestamp,
        }

    # Detailed mode: comprehensive diagnostics
    logger.info("[HEALTH] Detailed health check requested")

    # Check all subsystems
    database_status = _check_database_status()
    storage_status = _check_storage_status()
    gpu_status = _check_gpu_status()
    registry_status = _check_registry_status()

    details = {
        "database": database_status,
        "storage": storage_status,
        "gpu": gpu_status,
        "registry": registry_status,
    }

    # Determine overall health status
    is_healthy = all(
        [
            database_status.get("status") in ["ok", "warning"],
            storage_status.get("status") in ["ok", "warning"],
            registry_status.get("status") in ["ok", "warning"],
            # GPU is optional - don't fail if unavailable
        ]
    )

    if is_healthy:
        response_data = {
            "status": "ok",
            "version": __version__,
            "timestamp": timestamp,
            "details": details,
        }
        response.status_code = status.HTTP_200_OK
    else:
        response_data = {
            "status": "unhealthy",
            "version": __version__,
            "timestamp": timestamp,
            "details": details,
        }
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        logger.warning("[HEALTH] System unhealthy - returning 503")

    return response_data
