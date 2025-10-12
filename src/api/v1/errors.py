"""Error models and exception handling for VideoAnnotator API.

This module provides standardized error response models and custom exceptions
for consistent error handling across all API endpoints.

All 4xx and 5xx responses use the ErrorEnvelope format with:
- Machine-readable error codes
- Human-readable messages
- Actionable hints for resolution
- Timestamps for error tracking
"""

from datetime import UTC, datetime
from typing import Any, ClassVar

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Detailed error information for API responses.

    Provides structured error data including error codes, messages, hints,
    and timestamps for comprehensive error reporting and debugging.

    Attributes:
        code: Machine-readable error code (uppercase snake_case)
              e.g., "JOB_NOT_FOUND", "INVALID_CONFIG"
        message: Human-readable error message describing what went wrong
        detail: Optional additional context (e.g., field-level validation errors)
        hint: Optional suggested action to fix the error
        field: Optional field name for validation errors (dot-notation path)
        timestamp: When the error occurred (ISO 8601 UTC)

    Examples:
        >>> error = ErrorDetail(
        ...     code="JOB_NOT_FOUND",
        ...     message="Job abc123 not found",
        ...     hint="Check job ID or use GET /api/v1/jobs to list jobs"
        ... )
        >>> error.model_dump_json()
    """

    code: str = Field(
        ...,
        description="Machine-readable error code (e.g., 'JOB_NOT_FOUND')",
        examples=["JOB_NOT_FOUND", "INVALID_CONFIG", "STORAGE_FULL"],
    )

    message: str = Field(
        ...,
        description="Human-readable error message",
        examples=[
            "Job abc123 not found",
            "Configuration validation failed",
            "Insufficient disk space",
        ],
    )

    detail: dict[str, Any] | None = Field(
        None,
        description="Additional context (field-level errors, system info, etc.)",
        examples=[
            {"errors": [{"field": "pipeline", "message": "Pipeline 'xyz' not found"}]},
            {"pid": 12345, "signal": "SIGKILL"},
        ],
    )

    hint: str | None = Field(
        None,
        description="Suggested action to fix the error",
        examples=[
            "Use GET /api/v1/jobs to list all jobs",
            "Available pipelines: face, person, audio, scene",
            "Reduce batch_size or wait for running jobs to finish",
        ],
    )

    field: str | None = Field(
        None,
        description="Field name for validation errors (dot-notation path)",
        examples=["pipeline", "config.detection.confidence_threshold"],
    )

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When error occurred (ISO 8601 UTC)",
    )

    class Config:
        """Pydantic model configuration."""

        json_schema_extra: ClassVar[dict[str, Any]] = {
            "example": {
                "code": "INVALID_CONFIG",
                "message": "Configuration validation failed",
                "detail": {
                    "errors": [
                        {
                            "field": "config.detection.confidence_threshold",
                            "message": "Value must be between 0.0 and 1.0",
                        }
                    ]
                },
                "hint": "Fix validation errors in config.detection section",
                "field": None,
                "timestamp": "2025-10-12T10:30:00Z",
            }
        }


class ErrorEnvelope(BaseModel):
    """Standard error response wrapper for all API errors.

    All HTTP error responses (4xx, 5xx) are wrapped in this envelope
    for consistent error handling across the API.

    Attributes:
        error: The error detail object

    Examples:
        >>> envelope = ErrorEnvelope(
        ...     error=ErrorDetail(
        ...         code="JOB_NOT_FOUND",
        ...         message="Job abc123 not found",
        ...         hint="Check job ID or use GET /api/v1/jobs"
        ...     )
        ... )
        >>> envelope.model_dump_json()
    """

    error: ErrorDetail = Field(..., description="Detailed error information")

    class Config:
        """Pydantic model configuration."""

        json_schema_extra: ClassVar[dict[str, Any]] = {
            "example": {
                "error": {
                    "code": "JOB_NOT_FOUND",
                    "message": "Job abc123 not found",
                    "hint": "Check job ID or use GET /api/v1/jobs to list all jobs",
                    "timestamp": "2025-10-12T10:30:00Z",
                }
            }
        }
