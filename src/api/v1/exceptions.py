"""Custom exceptions for VideoAnnotator API.

This module defines a hierarchy of exceptions that map to specific HTTP status codes
and error responses. All exceptions inherit from VideoAnnotatorException and are
automatically converted to ErrorEnvelope responses by FastAPI exception handlers.
"""


class VideoAnnotatorException(Exception):
    """Base exception for all VideoAnnotator errors.

    All custom exceptions should inherit from this base class to enable
    centralized error handling and consistent error responses.

    Attributes:
        message: Human-readable error message
        code: Machine-readable error code (uppercase snake_case)
        hint: Optional suggested action to fix the error
        detail: Optional additional context
        status_code: HTTP status code for this error type
    """

    message: str
    code: str
    hint: str | None
    detail: dict | None
    status_code: int

    def __init__(
        self,
        message: str,
        code: str,
        hint: str | None = None,
        detail: dict | None = None,
    ):
        """Initialize the exception.

        Args:
            message: Human-readable error message
            code: Machine-readable error code
            hint: Optional suggested action
            detail: Optional additional context
        """
        super().__init__(message)
        self.message = message
        self.code = code
        self.hint = hint
        self.detail = detail
        self.status_code = 500  # Default, overridden by subclasses


# 404 Not Found Exceptions


class JobNotFoundException(VideoAnnotatorException):
    """Exception raised when a job ID is not found in the database."""

    def __init__(self, job_id: str, hint: str | None = None):
        """Initialize JobNotFoundException.

        Args:
            job_id: The job ID that was not found
            hint: Optional hint for resolution
        """
        super().__init__(
            message=f"Job {job_id} not found",
            code="JOB_NOT_FOUND",
            hint=hint or "Check job ID or use GET /api/v1/jobs to list all jobs",
        )
        self.status_code = 404


class PipelineNotFoundException(VideoAnnotatorException):
    """Exception raised when a pipeline name is not found in the registry."""

    def __init__(
        self, pipeline_name: str, available_pipelines: list[str] | None = None
    ):
        """Initialize PipelineNotFoundException.

        Args:
            pipeline_name: The pipeline name that was not found
            available_pipelines: Optional list of available pipeline names
        """
        hint = None
        if available_pipelines:
            hint = f"Available pipelines: {', '.join(available_pipelines)}"

        super().__init__(
            message=f"Pipeline '{pipeline_name}' not found in registry",
            code="PIPELINE_NOT_FOUND",
            hint=hint,
        )
        self.status_code = 404


# 400 Bad Request Exceptions


class InvalidConfigException(VideoAnnotatorException):
    """Exception raised when configuration validation fails."""

    def __init__(
        self, message: str, errors: list | None = None, hint: str | None = None
    ):
        """Initialize InvalidConfigException.

        Args:
            message: Human-readable error message
            errors: Optional list of field-level validation errors
            hint: Optional hint for resolution
        """
        detail = {"errors": errors} if errors else None
        super().__init__(
            message=message,
            code="INVALID_CONFIG",
            hint=hint or "Fix validation errors and retry",
            detail=detail,
        )
        self.status_code = 400


class InvalidRequestException(VideoAnnotatorException):
    """Exception raised when request body is malformed or invalid."""

    def __init__(self, message: str, hint: str | None = None):
        """Initialize InvalidRequestException.

        Args:
            message: Human-readable error message
            hint: Optional hint for resolution
        """
        super().__init__(
            message=message,
            code="INVALID_REQUEST",
            hint=hint or "Check request body syntax and required fields",
        )
        self.status_code = 400


# 401 Unauthorized Exception


class UnauthorizedException(VideoAnnotatorException):
    """Exception raised when authentication credentials are missing or invalid."""

    def __init__(self, message: str = "Missing or invalid API key"):
        """Initialize UnauthorizedException.

        Args:
            message: Human-readable error message
        """
        super().__init__(
            message=message,
            code="UNAUTHORIZED",
            hint="Provide X-API-Key header or JWT bearer token",
        )
        self.status_code = 401


# 403 Forbidden Exception


class ForbiddenException(VideoAnnotatorException):
    """Exception raised when user has insufficient permissions for resource."""

    def __init__(self, resource: str):
        """Initialize ForbiddenException.

        Args:
            resource: The resource that requires additional permissions
        """
        super().__init__(
            message=f"Insufficient permissions to access {resource}",
            code="FORBIDDEN",
            hint="Contact admin for access to this resource",
        )
        self.status_code = 403


# 409 Conflict Exception


class JobAlreadyCompletedException(VideoAnnotatorException):
    """Exception raised when trying to cancel a job that is already completed."""

    def __init__(self, job_id: str, status: str):
        """Initialize JobAlreadyCompletedException.

        Args:
            job_id: The job ID
            status: The current terminal status (COMPLETED, FAILED, CANCELLED)
        """
        super().__init__(
            message=f"Cannot cancel job {job_id} in {status} state",
            code="JOB_ALREADY_COMPLETED",
            hint=f"Job already in terminal state: {status}",
        )
        self.status_code = 409


# 507 Insufficient Storage Exceptions


class StorageFullException(VideoAnnotatorException):
    """Exception raised when disk space is insufficient for operation."""

    def __init__(self, required_gb: float, available_gb: float):
        """Initialize StorageFullException.

        Args:
            required_gb: Required disk space in GB
            available_gb: Available disk space in GB
        """
        super().__init__(
            message=f"Insufficient disk space (need {required_gb:.1f}GB, have {available_gb:.1f}GB)",
            code="STORAGE_FULL",
            hint=f"Free up {required_gb - available_gb:.1f}GB or configure larger storage volume",
            detail={"required_gb": required_gb, "available_gb": available_gb},
        )
        self.status_code = 507


class GPUOutOfMemoryException(VideoAnnotatorException):
    """Exception raised when GPU VRAM is exhausted."""

    def __init__(
        self,
        requested_mb: int | None = None,
        available_mb: int | None = None,
        device: str | None = None,
    ):
        """Initialize GPUOutOfMemoryException.

        Args:
            requested_mb: Requested GPU memory in MB
            available_mb: Available GPU memory in MB
            device: GPU device name
        """
        if requested_mb and available_mb:
            message = f"GPU out of memory (requested {requested_mb}MB, available {available_mb}MB)"
        else:
            message = "GPU out of memory"

        detail = {}
        if requested_mb:
            detail["requested_mb"] = requested_mb
        if available_mb:
            detail["available_mb"] = available_mb
        if device:
            detail["device"] = device

        super().__init__(
            message=message,
            code="GPU_OUT_OF_MEMORY",
            hint="Reduce batch_size or wait for running jobs to finish",
            detail=detail if detail else None,
        )
        self.status_code = 507


# 500 Internal Server Error Exceptions


class CancellationFailedException(VideoAnnotatorException):
    """Exception raised when job cancellation fails."""

    def __init__(self, job_id: str, reason: str, pid: int | None = None):
        """Initialize CancellationFailedException.

        Args:
            job_id: The job ID
            reason: Why cancellation failed
            pid: Optional process ID
        """
        detail = {"reason": reason}
        if pid:
            detail["pid"] = pid

        super().__init__(
            message=f"Failed to cancel job {job_id}: {reason}",
            code="CANCELLATION_FAILED",
            hint="Worker process may require manual cleanup. Contact support.",
            detail=detail,
        )
        self.status_code = 500


class ValidationSystemException(VideoAnnotatorException):
    """Exception raised when validation system encounters an error."""

    def __init__(self, pipeline: str, reason: str):
        """Initialize ValidationSystemException.

        Args:
            pipeline: The pipeline name
            reason: Why validation failed
        """
        super().__init__(
            message=f"Failed to load validation schema for pipeline '{pipeline}'",
            code="VALIDATION_ERROR",
            hint="Check registry metadata integrity. Contact support.",
            detail={
                "pipeline": pipeline,
                "registry_path": f"/app/registry/metadata/{pipeline}.yaml",
            },
        )
        self.status_code = 500


class InternalServerException(VideoAnnotatorException):
    """Exception raised for unexpected internal server errors."""

    def __init__(
        self, message: str = "An unexpected error occurred", trace_id: str | None = None
    ):
        """Initialize InternalServerException.

        Args:
            message: Human-readable error message
            trace_id: Optional trace ID for debugging
        """
        detail = {"trace_id": trace_id} if trace_id else None
        hint = (
            "Contact support with timestamp"
            if not trace_id
            else f"Contact support with trace ID: {trace_id}"
        )

        super().__init__(
            message=message,
            code="INTERNAL_SERVER_ERROR",
            hint=hint,
            detail=detail,
        )
        self.status_code = 500
