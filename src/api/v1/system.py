"""
System management endpoints for VideoAnnotator API
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import psutil
import platform
from datetime import datetime

from version import __version__ as videoannotator_version
from registry.pipeline_registry import get_registry
import time

PROCESS_START_TIME = time.time()
from api.database import get_storage_backend, check_database_health, get_database_info


router = APIRouter()


def _get_disk_usage_percent() -> float:
    """Get disk usage percentage, handling Windows vs Unix."""
    try:
        if platform.system() == 'Windows':
            disk = psutil.disk_usage('C:/')
        else:
            disk = psutil.disk_usage('/')
        return (disk.used / disk.total * 100) if disk.total > 0 else 0.0
    except Exception:
        return 0.0


def _get_database_status() -> Dict[str, Any]:
    """Get database status for health check."""
    try:
        is_healthy, message = check_database_health()
        if is_healthy:
            return {"status": "healthy", "message": message}
        else:
            return {"status": "unhealthy", "message": message}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/health")
async def detailed_health_check():
    """
    Detailed system health check.
    
    Returns:
        Comprehensive system health information
    """
    try:
        # Get system information
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Get disk usage - handle Windows vs Unix paths
        try:
            if platform.system() == 'Windows':
                disk = psutil.disk_usage('C:/')
            else:
                disk = psutil.disk_usage('/')
        except Exception:
            # Fallback if disk check fails
            disk = type('MockDisk', (), {'total': 0, 'used': 0, 'free': 0})()
        
        # Get GPU information if available
        gpu_info = None
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info = {
                    "available": True,
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "device_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None,
                    "memory_allocated": torch.cuda.memory_allocated(0) if torch.cuda.device_count() > 0 else 0,
                    "memory_reserved": torch.cuda.memory_reserved(0) if torch.cuda.device_count() > 0 else 0
                }
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
                "free": memory.free
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": (disk.used / disk.total) * 100 if disk.total > 0 else 0
            }
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
                "pipelines": "ready"
            },
            "uptime_seconds": uptime_seconds
        }
        
    except Exception as e:
        # If the health check fails, still report the current packaged version
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "api_version": videoannotator_version,
            "videoannotator_version": videoannotator_version
        }


@router.get("/metrics")
async def get_system_metrics():
    """
    Get system performance metrics.
    
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
                "jobs_active": db_stats.get("running_jobs", 0) + db_stats.get("pending_jobs", 0),
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
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get metrics: {str(e)}"
        )


@router.get("/config")
async def get_system_config():
    """
    Get current system configuration.
    
    Returns:
        System configuration information (non-sensitive)
    """
    try:
        # TODO: Load and return current system configuration
        # Exclude sensitive information like API keys, database passwords, etc.
        
        # TODO: Load actual config when dependencies are fixed
        # from ...config import load_config
        # config = load_config()
        config = {}
        
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
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get config: {str(e)}"
        )


@router.get("/database")
async def get_database_info_endpoint():
    """
    Get database information and statistics.
    
    Returns:
        Database configuration and statistics
    """
    try:
        return get_database_info()
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get database info: {str(e)}"
        )