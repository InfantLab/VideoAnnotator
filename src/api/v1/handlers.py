"""Exception handlers for v1.3.0 error envelope system.

This module registers FastAPI exception handlers that convert VideoAnnotatorException
instances into ErrorEnvelope responses. These handlers work alongside the existing
APIError handlers from src/api/errors.py.

Usage:
    from api.v1.handlers import register_v1_exception_handlers
    register_v1_exception_handlers(app)
"""

from datetime import UTC, datetime

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.v1.errors import ErrorDetail, ErrorEnvelope
from api.v1.exceptions import VideoAnnotatorException
from utils.logging_config import get_logger

logger = get_logger("api")


async def videoannotator_exception_handler(
    request: Request, exc: VideoAnnotatorException
) -> JSONResponse:
    """Convert VideoAnnotatorException to ErrorEnvelope JSON response.

    Args:
        request: FastAPI request object (provides context for logging)
        exc: The VideoAnnotatorException to convert

    Returns:
        JSONResponse with ErrorEnvelope structure and appropriate HTTP status code

    Example Response:
        {
            "error": {
                "code": "JOB_NOT_FOUND",
                "message": "Job 'abc123' not found",
                "hint": "Check job ID or list jobs with GET /api/v1/jobs",
                "detail": {"job_id": "abc123"},
                "timestamp": "2025-10-12T14:30:00Z"
            }
        }
    """
    # Extract request context for enhanced logging
    request_context = {
        "method": request.method,
        "path": request.url.path,
        "client": request.client.host if request.client else None,
    }

    # Log the error with context
    logger.warning(
        f"API Error: {exc.code} - {exc.message}",
        extra={
            "error_code": exc.code,
            "status_code": exc.status_code,
            "request": request_context,
            "exception_type": type(exc).__name__,
        },
    )

    # Build ErrorDetail from exception
    error_detail = ErrorDetail(
        code=exc.code,
        message=exc.message,
        detail=exc.detail,
        hint=exc.hint,
        timestamp=datetime.now(UTC),
    )

    # Wrap in ErrorEnvelope
    error_envelope = ErrorEnvelope(error=error_detail)

    # Return JSON response with appropriate status code
    # Include top-level "detail" key for backward compatibility
    content = error_envelope.model_dump(mode="json", exclude_none=True)
    content["detail"] = exc.message  # Legacy field for older clients
    return JSONResponse(
        status_code=exc.status_code,
        content=content,
    )


def register_v1_exception_handlers(app: FastAPI) -> None:
    """Register v1.3.0 exception handlers on the FastAPI app.

    This function should be called during application startup to enable
    automatic conversion of VideoAnnotatorException instances to ErrorEnvelope
    responses.

    Args:
        app: The FastAPI application instance

    Example:
        app = FastAPI()
        register_v1_exception_handlers(app)

    Note:
        This works alongside existing APIError handlers from src/api/errors.py.
        The handlers are tried in registration order, so VideoAnnotatorException
        will be caught before falling through to generic APIError handling.
    """
    app.add_exception_handler(VideoAnnotatorException, videoannotator_exception_handler)
    logger.info(
        "Registered v1.3.0 exception handlers",
        extra={"handler": "VideoAnnotatorException -> ErrorEnvelope"},
    )


__all__ = ["register_v1_exception_handlers", "videoannotator_exception_handler"]
