"""Request logging middleware for VideoAnnotator API Server.

Provides comprehensive request/response logging for debugging and monitoring."""

import time
import uuid
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from utils.logging_config import get_logger, log_api_request


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all API requests and responses."""

    def __init__(self, app, exclude_paths: set | None = None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or {
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
        }
        self.request_logger = get_logger("requests")
        self.api_logger = get_logger("api")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log details."""

        # Skip logging for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Generate unique request ID
        request_id = str(uuid.uuid4())[:8]

        # Extract user information from request
        user_id = None
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # Extract user info from token (simplified)
            token = auth_header.split(" ")[1]
            if token in ["dev-token", "test-token"]:
                user_id = "test-user"
            else:
                user_id = token[:8] if len(token) > 8 else token

        # Log request start
        start_time = time.time()

        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "Unknown")

        self.api_logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "request_id": request_id,
                "user_id": user_id,
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "client_ip": client_ip,
                "user_agent": user_agent,
                "headers": dict(request.headers),
            },
        )

        # Process request
        try:
            response = await call_next(request)

            # Calculate response time
            response_time = time.time() - start_time

            # Log successful response
            log_api_request(
                self.request_logger,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                response_time=response_time,
                request_id=request_id,
                user_id=user_id,
                client_ip=client_ip,
                query_params=dict(request.query_params)
                if request.query_params
                else None,
            )

            return response

        except Exception as e:
            # Calculate response time for failed requests
            response_time = time.time() - start_time

            # Log error
            self.api_logger.error(
                f"Request failed: {request.method} {request.url.path} - {e!s}",
                extra={
                    "request_id": request_id,
                    "user_id": user_id,
                    "method": request.method,
                    "path": request.url.path,
                    "response_time": response_time,
                    "error": str(e),
                    "client_ip": client_ip,
                },
            )

            # Re-raise the exception
            raise

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers first (for reverse proxies)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        forwarded = request.headers.get("x-forwarded")
        if forwarded:
            return forwarded.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fall back to direct client
        return request.client.host if request.client else "unknown"


class ErrorLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to capture and log unhandled errors."""

    def __init__(self, app):
        super().__init__(app)
        self.error_logger = get_logger("api")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Capture and log unhandled errors."""
        try:
            return await call_next(request)
        except Exception as e:
            # Log the unhandled error
            self.error_logger.error(
                f"Unhandled error in {request.method} {request.url.path}: {e!s}",
                exc_info=True,
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "query_params": dict(request.query_params),
                    "client_ip": request.client.host if request.client else "unknown",
                    "error_type": type(e).__name__,
                },
            )

            # Re-raise to be handled by FastAPI's exception handlers
            raise
