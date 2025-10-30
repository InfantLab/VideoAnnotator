"""Unit tests for v1.3.0 error envelope system.

Tests error models (ErrorDetail, ErrorEnvelope), custom exceptions, and
exception handler integration with FastAPI.
"""

from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from videoannotator.api.v1.errors import ErrorDetail, ErrorEnvelope
from videoannotator.api.v1.exceptions import (
    CancellationFailedException,
    ForbiddenException,
    GPUOutOfMemoryException,
    InternalServerException,
    InvalidConfigException,
    InvalidRequestException,
    JobAlreadyCompletedException,
    JobNotFoundException,
    PipelineNotFoundException,
    StorageFullException,
    UnauthorizedException,
    ValidationSystemException,
    VideoAnnotatorException,
)
from videoannotator.api.v1.handlers import register_v1_exception_handlers


class TestErrorDetail:
    """Test ErrorDetail Pydantic model."""

    def test_minimal_error_detail(self):
        """Test ErrorDetail with only required fields."""
        error = ErrorDetail(
            code="TEST_ERROR",
            message="Test error message",
        )

        assert error.code == "TEST_ERROR"
        assert error.message == "Test error message"
        assert error.detail is None
        assert error.hint is None
        assert error.field is None
        assert isinstance(error.timestamp, datetime)

    def test_full_error_detail(self):
        """Test ErrorDetail with all fields populated."""
        timestamp = datetime.now(UTC)
        error = ErrorDetail(
            code="VALIDATION_ERROR",
            message="Invalid configuration",
            detail={"field": "pipeline_name", "value": "invalid"},
            hint="Use 'videoannotator pipelines --detailed' to list valid pipelines",
            field="config.pipeline_name",
            timestamp=timestamp,
        )

        assert error.code == "VALIDATION_ERROR"
        assert error.message == "Invalid configuration"
        assert error.detail == {"field": "pipeline_name", "value": "invalid"}
        assert (
            error.hint
            == "Use 'videoannotator pipelines --detailed' to list valid pipelines"
        )
        assert error.field == "config.pipeline_name"
        assert error.timestamp == timestamp

    def test_error_detail_json_serialization(self):
        """Test ErrorDetail serializes to JSON correctly."""
        error = ErrorDetail(
            code="TEST_ERROR",
            message="Test message",
            detail={"key": "value"},
            hint="Test hint",
        )

        json_data = error.model_dump(mode="json")

        assert json_data["code"] == "TEST_ERROR"
        assert json_data["message"] == "Test message"
        assert json_data["detail"] == {"key": "value"}
        assert json_data["hint"] == "Test hint"
        assert "timestamp" in json_data
        assert json_data["field"] is None

    def test_error_detail_exclude_none(self):
        """Test ErrorDetail excludes None values in JSON."""
        error = ErrorDetail(
            code="TEST_ERROR",
            message="Test message",
        )

        json_data = error.model_dump(mode="json", exclude_none=True)

        assert "detail" not in json_data
        assert "hint" not in json_data
        assert "field" not in json_data
        assert "code" in json_data
        assert "message" in json_data
        assert "timestamp" in json_data


class TestErrorEnvelope:
    """Test ErrorEnvelope Pydantic model."""

    def test_error_envelope_wraps_error_detail(self):
        """Test ErrorEnvelope wraps ErrorDetail correctly."""
        error_detail = ErrorDetail(
            code="WRAPPED_ERROR",
            message="This is wrapped",
        )
        envelope = ErrorEnvelope(error=error_detail)

        assert envelope.error == error_detail
        assert envelope.error.code == "WRAPPED_ERROR"
        assert envelope.error.message == "This is wrapped"

    def test_error_envelope_json_serialization(self):
        """Test ErrorEnvelope serializes to expected JSON structure."""
        error_detail = ErrorDetail(
            code="API_ERROR",
            message="Something went wrong",
            hint="Try again",
        )
        envelope = ErrorEnvelope(error=error_detail)

        json_data = envelope.model_dump(mode="json", exclude_none=True)

        assert "error" in json_data
        assert json_data["error"]["code"] == "API_ERROR"
        assert json_data["error"]["message"] == "Something went wrong"
        assert json_data["error"]["hint"] == "Try again"
        assert "timestamp" in json_data["error"]


class TestCustomExceptions:
    """Test custom exception classes."""

    def test_base_videoannotator_exception(self):
        """Test base VideoAnnotatorException."""
        exc = VideoAnnotatorException(
            message="Base error",
            code="BASE_ERROR",
            hint="Fix it",
            detail={"context": "test"},
        )

        assert exc.message == "Base error"
        assert exc.code == "BASE_ERROR"
        assert exc.hint == "Fix it"
        assert exc.detail == {"context": "test"}
        assert exc.status_code == 500  # Default
        assert str(exc) == "Base error"

    def test_job_not_found_exception(self):
        """Test JobNotFoundException (404)."""
        exc = JobNotFoundException(job_id="abc123")

        assert exc.status_code == 404
        assert exc.code == "JOB_NOT_FOUND"
        assert "abc123" in exc.message
        assert exc.detail is not None
        assert exc.detail["job_id"] == "abc123"
        assert exc.hint is not None

    def test_pipeline_not_found_exception(self):
        """Test PipelineNotFoundException (404)."""
        exc = PipelineNotFoundException(
            pipeline_name="invalid_pipeline",
            available_pipelines=["pipeline1", "pipeline2"],
        )

        assert exc.status_code == 404
        assert exc.code == "PIPELINE_NOT_FOUND"
        assert "invalid_pipeline" in exc.message
        assert exc.detail is not None
        assert exc.detail["pipeline_name"] == "invalid_pipeline"
        assert exc.detail["available_pipelines"] == ["pipeline1", "pipeline2"]

    def test_invalid_config_exception(self):
        """Test InvalidConfigException (400)."""
        errors = [{"field": "timeout", "error": "must be positive"}]
        exc = InvalidConfigException(
            message="Invalid config",
            errors=errors,
            hint="Check docs",
        )

        assert exc.status_code == 400
        assert exc.code == "INVALID_CONFIG"
        assert exc.detail is not None
        assert exc.detail["errors"] == errors

    def test_invalid_request_exception(self):
        """Test InvalidRequestException (400)."""
        exc = InvalidRequestException(
            message="Bad request",
            hint="Fix parameter",
        )

        assert exc.status_code == 400
        assert exc.code == "INVALID_REQUEST"

    def test_unauthorized_exception(self):
        """Test UnauthorizedException (401)."""
        exc = UnauthorizedException(message="Invalid token")

        assert exc.status_code == 401
        assert exc.code == "UNAUTHORIZED"
        assert "Invalid token" in exc.message

    def test_forbidden_exception(self):
        """Test ForbiddenException (403)."""
        exc = ForbiddenException(resource="admin_endpoint")

        assert exc.status_code == 403
        assert exc.code == "FORBIDDEN"
        assert "admin_endpoint" in exc.message

    def test_job_already_completed_exception(self):
        """Test JobAlreadyCompletedException (409)."""
        exc = JobAlreadyCompletedException(
            job_id="xyz789",
            status="completed",
        )

        assert exc.status_code == 409
        assert exc.code == "JOB_ALREADY_COMPLETED"
        assert "xyz789" in exc.message
        assert exc.detail is not None
        assert exc.detail["job_id"] == "xyz789"
        assert exc.detail["status"] == "completed"

    def test_storage_full_exception(self):
        """Test StorageFullException (507)."""
        exc = StorageFullException(
            required_gb=50.5,
            available_gb=10.2,
        )

        assert exc.status_code == 507
        assert exc.code == "STORAGE_FULL"
        assert "50.5" in exc.message
        assert "10.2" in exc.message
        assert exc.detail is not None
        assert exc.detail["required_gb"] == 50.5
        assert exc.detail["available_gb"] == 10.2

    def test_gpu_out_of_memory_exception(self):
        """Test GPUOutOfMemoryException (507)."""
        exc = GPUOutOfMemoryException(
            requested_mb=8192,
            available_mb=4096,
            device="cuda:0",
        )

        assert exc.status_code == 507
        assert exc.code == "GPU_OUT_OF_MEMORY"
        assert "8192" in exc.message
        assert "4096" in exc.message
        assert exc.detail is not None
        assert exc.detail["requested_mb"] == 8192
        assert exc.detail["available_mb"] == 4096
        assert exc.detail["device"] == "cuda:0"

    def test_cancellation_failed_exception(self):
        """Test CancellationFailedException (500)."""
        exc = CancellationFailedException(
            job_id="cancel123",
            reason="Process not responding",
            pid=12345,
        )

        assert exc.status_code == 500
        assert exc.code == "CANCELLATION_FAILED"
        assert "cancel123" in exc.message
        assert exc.detail is not None
        assert exc.detail["job_id"] == "cancel123"
        assert exc.detail["reason"] == "Process not responding"
        assert exc.detail["pid"] == 12345

    def test_validation_system_exception(self):
        """Test ValidationSystemException (500)."""
        exc = ValidationSystemException(
            pipeline="test_pipeline",
            reason="Validation failed",
        )

        assert exc.status_code == 500
        assert exc.code == "VALIDATION_SYSTEM_ERROR"
        assert "test_pipeline" in exc.message
        assert exc.detail is not None
        assert exc.detail["pipeline"] == "test_pipeline"
        assert exc.detail["reason"] == "Validation failed"

    def test_internal_server_exception(self):
        """Test InternalServerException (500)."""
        exc = InternalServerException(
            message="Unexpected error",
            trace_id="trace-abc-123",
        )

        assert exc.status_code == 500
        assert exc.code == "INTERNAL_SERVER_ERROR"
        assert "Unexpected error" in exc.message
        assert exc.detail is not None
        assert exc.detail["trace_id"] == "trace-abc-123"


class TestExceptionHandlers:
    """Test FastAPI exception handler integration."""

    @pytest.fixture
    def app(self) -> FastAPI:
        """Create FastAPI app with exception handlers registered."""
        app = FastAPI()
        register_v1_exception_handlers(app)

        # Add test endpoints that raise exceptions
        @app.get("/test/job-not-found")
        async def raise_job_not_found():
            raise JobNotFoundException(job_id="test123")

        @app.get("/test/pipeline-not-found")
        async def raise_pipeline_not_found():
            raise PipelineNotFoundException(
                pipeline_name="invalid",
                available_pipelines=["pipeline1", "pipeline2"],
            )

        @app.get("/test/invalid-config")
        async def raise_invalid_config():
            raise InvalidConfigException(
                message="Config error",
                errors=[{"field": "timeout", "error": "invalid"}],
                hint="Check config",
            )

        @app.get("/test/storage-full")
        async def raise_storage_full():
            raise StorageFullException(required_gb=100.0, available_gb=5.0)

        @app.get("/test/generic-error")
        async def raise_generic():
            raise VideoAnnotatorException(
                message="Generic error",
                code="GENERIC_ERROR",
                hint="Try something",
            )

        return app

    @pytest.fixture
    def client(self, app: FastAPI) -> TestClient:
        """Create test client."""
        return TestClient(app)

    def test_job_not_found_handler(self, client: TestClient):
        """Test JobNotFoundException is converted to ErrorEnvelope."""
        response = client.get("/test/job-not-found")

        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "JOB_NOT_FOUND"
        assert "test123" in data["error"]["message"]
        assert "timestamp" in data["error"]
        assert data["error"]["detail"]["job_id"] == "test123"

    def test_pipeline_not_found_handler(self, client: TestClient):
        """Test PipelineNotFoundException is converted to ErrorEnvelope."""
        response = client.get("/test/pipeline-not-found")

        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "PIPELINE_NOT_FOUND"
        assert "invalid" in data["error"]["message"]
        assert data["error"]["detail"]["pipeline_name"] == "invalid"
        assert data["error"]["detail"]["available_pipelines"] == [
            "pipeline1",
            "pipeline2",
        ]

    def test_invalid_config_handler(self, client: TestClient):
        """Test InvalidConfigException is converted to ErrorEnvelope."""
        response = client.get("/test/invalid-config")

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "INVALID_CONFIG"
        assert data["error"]["hint"] == "Check config"
        assert len(data["error"]["detail"]["errors"]) == 1

    def test_storage_full_handler(self, client: TestClient):
        """Test StorageFullException is converted to ErrorEnvelope."""
        response = client.get("/test/storage-full")

        assert response.status_code == 507
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "STORAGE_FULL"
        assert data["error"]["detail"]["required_gb"] == 100.0
        assert data["error"]["detail"]["available_gb"] == 5.0

    def test_generic_exception_handler(self, client: TestClient):
        """Test generic VideoAnnotatorException is converted to ErrorEnvelope."""
        response = client.get("/test/generic-error")

        assert response.status_code == 500  # Default status code
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "GENERIC_ERROR"
        assert data["error"]["message"] == "Generic error"
        assert data["error"]["hint"] == "Try something"
        assert "timestamp" in data["error"]

    def test_error_envelope_structure(self, client: TestClient):
        """Test all errors follow ErrorEnvelope structure."""
        response = client.get("/test/job-not-found")
        data = response.json()

        # Verify top-level structure
        assert "error" in data
        assert len(data) == 1  # Only 'error' key at top level

        # Verify error structure
        error = data["error"]
        assert "code" in error
        assert "message" in error
        assert "timestamp" in error
        # detail and hint may or may not be present
        assert isinstance(error["code"], str)
        assert isinstance(error["message"], str)
        assert isinstance(error["timestamp"], str)

    def test_timestamp_format(self, client: TestClient):
        """Test timestamp is ISO 8601 UTC format."""
        response = client.get("/test/generic-error")
        data = response.json()

        timestamp_str = data["error"]["timestamp"]
        # Should parse as ISO 8601
        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        assert timestamp.tzinfo is not None  # Has timezone info
