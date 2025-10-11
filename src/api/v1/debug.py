"""Debug endpoints for VideoAnnotator API v1.2.0.

These endpoints provide debugging and diagnostic information for client-
server collaboration and troubleshooting.
"""

import json
import platform
from datetime import datetime, timedelta
from typing import Any

import psutil
from fastapi import APIRouter, Depends, HTTPException, Query

from api.database import get_database_info, get_storage_backend
from api.dependencies import get_current_user
from version import __version__ as videoannotator_version

# Store for request logging (in production, use Redis or database)
_request_log: list[dict[str, Any]] = []
_server_start_time = datetime.now()

router = APIRouter()


def log_request(
    method: str,
    path: str,
    status_code: int,
    response_time_ms: int,
    user_id: str | None = None,
) -> None:
    """Log API request for debugging purposes."""
    global _request_log

    request_entry = {
        "timestamp": datetime.now().isoformat(),
        "method": method,
        "path": path,
        "status_code": status_code,
        "response_time_ms": response_time_ms,
        "user_id": user_id,
        "request_id": f"req_{datetime.now().strftime('%H%M%S')}",
    }

    _request_log.append(request_entry)

    # Keep only last 100 requests to prevent memory issues
    if len(_request_log) > 100:
        _request_log.pop(0)


@router.get("/server-info")
async def get_server_debug_info():
    """Get comprehensive server information for debugging.

    Returns server configuration, status, and system information.
    """
    try:
        # Get database info
        db_info = get_database_info()

        # Get system information
        memory = psutil.virtual_memory()

        # Try to get GPU info
        gpu_info = {"available": False, "error": "Not detected"}
        try:
            import torch

            if torch.cuda.is_available():
                gpu_info = {
                    "available": True,
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "device_name": torch.cuda.get_device_name(0)
                    if torch.cuda.device_count() > 0
                    else None,
                    "memory_allocated_gb": round(
                        torch.cuda.memory_allocated(0) / 1024**3, 2
                    )
                    if torch.cuda.device_count() > 0
                    else 0,
                    "memory_reserved_gb": round(
                        torch.cuda.memory_reserved(0) / 1024**3, 2
                    )
                    if torch.cuda.device_count() > 0
                    else 0,
                }
            else:
                gpu_info = {"available": False, "error": "CUDA not available"}
        except ImportError:
            gpu_info = {"available": False, "error": "PyTorch not installed"}

        # Get pipeline status (basic check)
        pipeline_status = _get_pipeline_status()

        uptime = datetime.now() - _server_start_time

        return {
            "server": {
                "version": videoannotator_version,
                "videoannotator_version": videoannotator_version,
                "environment": "development",  # TODO: Get from config
                "start_time": _server_start_time.isoformat(),
                "uptime_seconds": int(uptime.total_seconds()),
                "debug_mode": True,  # TODO: Get from config
                "api_base_path": "/api/v1",
            },
            "database": {
                "backend": db_info.get("backend_type", "unknown"),
                "path": db_info.get("connection_info", {}).get(
                    "database_path", "unknown"
                ),
                "connection_status": "healthy",  # TODO: Add proper health check
                "total_jobs": db_info.get("statistics", {}).get("total_jobs", 0),
                "active_connections": 1,  # TODO: Track actual connections
                "size_mb": db_info.get("connection_info", {}).get(
                    "database_size_mb", 0
                ),
            },
            "pipelines": pipeline_status,
            "system": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "cpu_usage_percent": psutil.cpu_percent(),
                "memory_total_gb": round(memory.total / 1024**3, 2),
                "memory_used_gb": round(memory.used / 1024**3, 2),
                "memory_usage_percent": memory.percent,
                "gpu": gpu_info,
            },
            "request_stats": {
                "total_requests": len(_request_log),
                "requests_last_hour": len(
                    [
                        r
                        for r in _request_log
                        if datetime.fromisoformat(r["timestamp"])
                        > datetime.now() - timedelta(hours=1)
                    ]
                ),
                "average_response_time_ms": sum(
                    r["response_time_ms"] for r in _request_log[-10:]
                )
                / min(len(_request_log), 10)
                if _request_log
                else 0,
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get server info: {e!s}")


@router.get("/token-info")
async def get_token_debug_info(
    current_user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Get detailed information about the current API token.

    Returns token validation status, permissions, and rate limiting
    info.
    """
    try:
        # Mock response - replace with actual token validation logic
        return {
            "token": {
                "valid": True,
                "user_id": current_user.get("id", "unknown"),
                "username": current_user.get("username", "unknown"),
                "permissions": [
                    "job:submit",
                    "job:read",
                    "job:delete",
                    "pipeline:read",
                    "system:read",
                ],
                "created_at": "2025-01-23T10:00:00Z",  # TODO: Get from database
                "expires_at": None,  # TODO: Implement token expiry
                "rate_limit": {
                    "requests_per_minute": 100,
                    "requests_per_hour": 1000,
                    "remaining_this_minute": 87,  # TODO: Implement actual rate limiting
                    "remaining_this_hour": 945,
                    "reset_at": datetime.now().replace(second=0, microsecond=0)
                    + timedelta(minutes=1),
                },
            },
            "session": {
                "first_request": _request_log[0]["timestamp"] if _request_log else None,
                "last_request": _request_log[-1]["timestamp"] if _request_log else None,
                "total_requests": len(
                    [
                        r
                        for r in _request_log
                        if r.get("user_id") == current_user.get("id")
                    ]
                ),
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get token info: {e!s}")


@router.get("/pipelines")
async def get_pipeline_debug_info():
    """Get detailed pipeline configuration and status information.

    Returns comprehensive pipeline information including components,
    parameters, and current status.
    """
    try:
        # This will need to be updated when pipeline system is fully integrated
        pipelines = [
            {
                "name": "person_tracking",
                "display_name": "Person Tracking",
                "description": "YOLO11 + ByteTrack person detection and tracking",
                "status": "ready",
                "enabled": True,
                "components": [
                    {
                        "name": "yolo11_pose",
                        "enabled": True,
                        "model_loaded": True,
                        "model_path": "models/yolo/yolo11n-pose.pt",
                        "parameters": {
                            "confidence_threshold": {
                                "type": "float",
                                "default": 0.5,
                                "min": 0.0,
                                "max": 1.0,
                            },
                            "iou_threshold": {
                                "type": "float",
                                "default": 0.7,
                                "min": 0.0,
                                "max": 1.0,
                            },
                            "max_detections": {
                                "type": "int",
                                "default": 100,
                                "min": 1,
                                "max": 1000,
                            },
                        },
                    },
                    {
                        "name": "bytetrack",
                        "enabled": True,
                        "model_loaded": True,
                        "parameters": {
                            "track_buffer": {
                                "type": "int",
                                "default": 30,
                                "min": 1,
                                "max": 100,
                            },
                            "match_threshold": {
                                "type": "float",
                                "default": 0.8,
                                "min": 0.0,
                                "max": 1.0,
                            },
                        },
                    },
                ],
            },
            {
                "name": "face_analysis",
                "display_name": "Face Analysis",
                "description": "Comprehensive facial analysis pipeline",
                "status": "ready",
                "enabled": True,
                "components": [
                    {
                        "name": "openface3",
                        "display_name": "OpenFace 3.0",
                        "enabled": False,  # Usually not available
                        "model_loaded": False,
                        "error": "OpenFace 3.0 not installed",
                        "parameters": {
                            "confidence_threshold": {
                                "type": "float",
                                "default": 0.5,
                                "min": 0.0,
                                "max": 1.0,
                            },
                            "enable_gaze": {"type": "bool", "default": True},
                            "enable_head_pose": {"type": "bool", "default": True},
                        },
                    },
                    {
                        "name": "laion_face",
                        "display_name": "LAION Face Analysis",
                        "enabled": True,
                        "model_loaded": True,
                        "supported_analyses": [
                            "age",
                            "gender",
                            "emotion",
                            "attractiveness",
                            "race",
                        ],
                        "parameters": {
                            "enable_age_estimation": {"type": "bool", "default": False},
                            "enable_emotion_recognition": {
                                "type": "bool",
                                "default": True,
                            },
                            "enable_gender_detection": {
                                "type": "bool",
                                "default": False,
                            },
                            "emotion_categories": {
                                "type": "list",
                                "default": [
                                    "happy",
                                    "sad",
                                    "angry",
                                    "surprised",
                                    "neutral",
                                    "fear",
                                    "disgust",
                                ],
                                "available": [
                                    "happy",
                                    "sad",
                                    "angry",
                                    "surprised",
                                    "neutral",
                                    "fear",
                                    "disgust",
                                    "contempt",
                                ],
                            },
                        },
                    },
                ],
            },
            {
                "name": "audio_processing",
                "display_name": "Audio Processing",
                "description": "Speech recognition and speaker diarization",
                "status": "ready",
                "enabled": True,
                "components": [
                    {
                        "name": "whisper",
                        "display_name": "Whisper Speech Recognition",
                        "enabled": True,
                        "model_loaded": True,
                        "model_size": "base",
                        "parameters": {
                            "model_size": {
                                "type": "select",
                                "default": "base",
                                "options": ["tiny", "base", "small", "medium", "large"],
                            },
                            "language": {
                                "type": "select",
                                "default": "auto",
                                "options": ["auto", "en", "es", "fr", "de", "it"],
                            },
                            "temperature": {
                                "type": "float",
                                "default": 0.0,
                                "min": 0.0,
                                "max": 1.0,
                            },
                        },
                    },
                    {
                        "name": "pyannote_diarization",
                        "display_name": "PyAnnote Speaker Diarization",
                        "enabled": True,
                        "model_loaded": True,
                        "parameters": {
                            "min_speakers": {
                                "type": "int",
                                "default": 1,
                                "min": 1,
                                "max": 10,
                            },
                            "max_speakers": {
                                "type": "int",
                                "default": 5,
                                "min": 1,
                                "max": 10,
                            },
                        },
                    },
                    {
                        "name": "laion_voice",
                        "display_name": "LAION Voice Emotion",
                        "enabled": True,
                        "model_loaded": True,
                        "parameters": {
                            "enable_voice_emotion": {"type": "bool", "default": False},
                            "emotion_categories": {
                                "type": "list",
                                "default": [
                                    "neutral",
                                    "happy",
                                    "sad",
                                    "angry",
                                    "surprised",
                                ],
                                "available": [
                                    "neutral",
                                    "happy",
                                    "sad",
                                    "angry",
                                    "surprised",
                                    "fear",
                                    "disgust",
                                ],
                            },
                        },
                    },
                ],
            },
            {
                "name": "scene_detection",
                "display_name": "Scene Detection",
                "description": "PySceneDetect + CLIP scene analysis",
                "status": "ready",
                "enabled": True,
                "components": [
                    {
                        "name": "pyscenedetect",
                        "enabled": True,
                        "model_loaded": True,
                        "parameters": {
                            "threshold": {
                                "type": "float",
                                "default": 30.0,
                                "min": 1.0,
                                "max": 100.0,
                            },
                            "min_scene_len": {
                                "type": "float",
                                "default": 1.0,
                                "min": 0.1,
                                "max": 10.0,
                            },
                        },
                    },
                    {
                        "name": "clip_classification",
                        "enabled": True,
                        "model_loaded": True,
                        "parameters": {
                            "confidence_threshold": {
                                "type": "float",
                                "default": 0.3,
                                "min": 0.0,
                                "max": 1.0,
                            },
                            "scene_categories": {
                                "type": "list",
                                "default": [
                                    "indoor",
                                    "outdoor",
                                    "office",
                                    "home",
                                    "restaurant",
                                    "street",
                                ],
                                "available": [
                                    "indoor",
                                    "outdoor",
                                    "office",
                                    "home",
                                    "restaurant",
                                    "street",
                                    "park",
                                    "car",
                                    "classroom",
                                    "kitchen",
                                ],
                            },
                        },
                    },
                ],
            },
        ]

        return {"pipelines": pipelines}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get pipeline info: {e!s}"
        )


@router.get("/jobs/{job_id}")
async def get_job_debug_info(
    job_id: str, current_user: dict = Depends(get_current_user)
) -> dict[str, Any]:
    """Get detailed debugging information for a specific job.

    Returns comprehensive job status, logs, resource usage, and file
    information.
    """
    try:
        storage = get_storage_backend()
        job = storage.get_job(job_id)

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # Mock debug information - replace with actual implementation
        debug_info = {
            "job": {
                "id": job.job_id,
                "status": job.status.value,
                "created_at": job.created_at.isoformat(),
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat()
                if job.completed_at
                else None,
                "debug_info": {
                    "queue_wait_time_ms": 1500,  # TODO: Calculate actual wait time
                    "processing_start_time": job.started_at.isoformat()
                    if job.started_at
                    else None,
                    "current_pipeline": "face_analysis",  # TODO: Get from actual processing
                    "progress_percentage": 45.2,  # TODO: Get from actual processing
                    "estimated_completion": None,  # TODO: Implement estimation
                    "resource_usage": {
                        "gpu_memory_mb": 1200,  # TODO: Get actual usage
                        "cpu_usage": 85.3,
                        "processing_fps": 2.1,
                    },
                },
                "pipeline_logs": [
                    {
                        "timestamp": "10:00:15",
                        "level": "INFO",
                        "message": "Starting job processing",
                    },
                    {
                        "timestamp": "10:01:30",
                        "level": "DEBUG",
                        "message": "Processing video analysis",
                    },
                    {
                        "timestamp": "10:02:45",
                        "level": "INFO",
                        "message": "Pipeline processing in progress",
                    },
                ],
                "errors": job.error_message.split("\n") if job.error_message else [],
                "files": {
                    "input_video": str(job.video_path) if job.video_path else None,
                    "output_directory": str(job.output_dir) if job.output_dir else None,
                    "temp_files": [],  # TODO: Track temporary files
                },
                "configuration": job.config,
                "selected_pipelines": job.selected_pipelines or [],
                "retry_count": job.retry_count,
            }
        }

        return debug_info

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get job debug info: {e!s}"
        )


@router.get("/request-log")
async def get_request_log(
    limit: int = Query(
        50, description="Maximum number of requests to return", ge=1, le=100
    ),
    current_user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Get recent API request log for debugging.

    Returns recent API requests with timing and status information.
    """
    try:
        # Return most recent requests, limited by the limit parameter
        recent_requests = _request_log[-limit:] if _request_log else []

        return {
            "requests": recent_requests,
            "total_logged": len(_request_log),
            "showing": len(recent_requests),
            "log_retention": "Last 100 requests",
            "server_uptime": int((datetime.now() - _server_start_time).total_seconds()),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get request log: {e!s}")


@router.get("/mock-events")
async def mock_sse_events(
    token: str | None = Query(None, description="API token for authentication"),
    job_id: str | None = Query(None, description="Job ID to monitor"),
) -> Any:
    """Mock SSE endpoint for client testing until real SSE is implemented.

    Returns Server-Sent Events format for job monitoring testing.
    """
    import asyncio

    from fastapi.responses import StreamingResponse

    async def event_generator():
        """Generate mock SSE events for testing."""
        event_count = 0

        while event_count < 10:  # Send 10 mock events then stop
            event_count += 1

            # Simulate different event types
            if event_count == 1:
                event_data = {
                    "type": "job.update",
                    "data": {
                        "job_id": job_id or "test-job-123",
                        "status": "processing",
                        "progress": 15.5,
                        "timestamp": datetime.now().isoformat(),
                    },
                }
            elif event_count == 5:
                event_data = {
                    "type": "job.log",
                    "data": {
                        "job_id": job_id or "test-job-123",
                        "level": "INFO",
                        "message": "Processing frame 150/300",
                        "timestamp": datetime.now().isoformat(),
                    },
                }
            elif event_count == 10:
                event_data = {
                    "type": "job.complete",
                    "data": {
                        "job_id": job_id or "test-job-123",
                        "artifacts": ["annotations.json", "results.coco"],
                        "completion_time": datetime.now().isoformat(),
                    },
                }
            else:
                event_data = {
                    "type": "job.update",
                    "data": {
                        "job_id": job_id or "test-job-123",
                        "status": "processing",
                        "progress": min(15.5 + (event_count * 8), 90),
                        "timestamp": datetime.now().isoformat(),
                    },
                }

            # Format as SSE
            yield f"data: {json.dumps(event_data)}\n\n"

            # Wait between events
            await asyncio.sleep(2)

        # Send close event
        yield f"data: {json.dumps({'type': 'connection.close', 'data': {'message': 'Mock session complete'}})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
        },
    )


@router.get("/background-jobs")
async def get_background_jobs_status() -> dict[str, Any]:
    """Get status of background job processing system.

    Returns information about the background job manager including
    currently processing jobs and system status.
    """
    try:
        from ..background_tasks import get_background_manager

        manager = get_background_manager()
        status = manager.get_status()

        # Add additional system info
        import os

        import psutil

        return {
            "background_processing": status,
            "system_info": {
                "pid": os.getpid(),
                "memory_usage_mb": round(
                    psutil.Process().memory_info().rss / 1024 / 1024, 2
                ),
                "cpu_percent": psutil.Process().cpu_percent(),
            },
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {
            "error": f"Failed to get background job status: {e!s}",
            "background_processing": {"running": False, "error": str(e)},
            "timestamp": datetime.now().isoformat(),
        }


def _get_pipeline_status() -> dict[str, Any]:
    """Get current pipeline initialization status."""
    # TODO: Replace with actual pipeline status checking
    return {
        "initialized": [
            "person_tracking",
            "face_analysis",
            "scene_detection",
            "audio_processing",
        ],
        "available": [
            "person_tracking",
            "face_analysis",
            "scene_detection",
            "audio_processing",
        ],
        "loading_errors": [],
        "total_available": 4,
        "initialization_time": "2.3 seconds",
    }
