"""VideoAnnotator API Server.

FastAPI-based REST API for video annotation processing.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..utils.logging_config import get_logger
from ..version import __version__ as videoannotator_version
from .errors import register_error_handlers
from .middleware import ErrorLoggingMiddleware, RequestLoggingMiddleware
from .v1 import api_router

API_VERSION = videoannotator_version

try:
    # Load environment variables from .env early (best-effort)
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


# Apply SciPy compatibility patch for OpenFace 3.0 before any pipeline imports
def apply_scipy_compatibility_patch():
    """Apply SciPy compatibility patch for OpenFace 3.0 globally."""
    try:
        import scipy.integrate

        if not hasattr(scipy.integrate, "simps"):
            import logging

            logging.info(
                "Applying scipy.integrate.simps compatibility patch for OpenFace 3.0"
            )
            scipy.integrate.simps = scipy.integrate.simpson
            logging.info("Successfully patched scipy.integrate.simps")
    except ImportError:
        pass  # SciPy not available
    except Exception as e:
        import logging

        logging.warning(f"Failed to apply scipy compatibility patch: {e}")


# Apply patch early
apply_scipy_compatibility_patch()

# Logging configuration is handled by entrypoints; this module uses the shared logger.
logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Startup
    logger.info("VideoAnnotator API server starting up...", extra={"event": "startup"})

    # Initialize security (API keys, CORS, authentication)
    try:
        from .startup import initialize_security

        logger.info("Initializing security configuration...")
        initialize_security()
        logger.info("Security configuration initialized")
    except Exception as e:
        logger.error(f"Security initialization failed: {e}")
        # Continue startup but log error

    # Run database migrations (v1.3.0)
    try:
        from ..database.migrations import migrate_to_v1_3_0

        logger.info("Running database migrations...")
        migration_success = migrate_to_v1_3_0()
        if migration_success:
            logger.info("Database migrations completed successfully")
        else:
            logger.warning("Database migrations completed with warnings")
    except Exception as e:
        logger.error(f"Database migration failed: {e}")
        # Don't fail startup if migration fails, but log prominently

    # Log server configuration
    from ..config_env import CORS_ORIGINS

    logger.info(
        "Server configuration initialized",
        extra={
            "api_version": API_VERSION,
            "videoannotator_version": videoannotator_version,
            "logging": "enhanced",
            "middleware": ["CORS", "RequestLogging", "ErrorLogging"],
            "cors_origins": CORS_ORIGINS,
            "background_processing": "enabled",
        },
    )

    # Start background job processing
    from .background_tasks import start_background_processing

    await start_background_processing()
    logger.info(
        "Background job processing started", extra={"component": "background_tasks"}
    )

    # TODO: Initialize pipeline cache

    yield

    # Shutdown
    logger.info(
        "VideoAnnotator API server shutting down...", extra={"event": "shutdown"}
    )

    # Stop background job processing
    from .background_tasks import stop_background_processing

    await stop_background_processing()
    logger.info(
        "Background job processing stopped", extra={"component": "background_tasks"}
    )

    # TODO: Cleanup pipeline resources


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="VideoAnnotator API",
        description="Production-ready REST API for video annotation processing",
        version=API_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        redirect_slashes=True,  # Enable automatic trailing slash redirects (standard behavior)
    )

    # Add middleware in correct order (last added = first executed)

    # Error logging middleware (innermost)
    app.add_middleware(ErrorLoggingMiddleware)

    # Request logging middleware
    app.add_middleware(
        RequestLoggingMiddleware,
        exclude_paths={"/health", "/docs", "/redoc", "/openapi.json", "/favicon.ico"},
    )

    # CORS middleware (outermost) - permissive defaults for local development
    from ..config_env import CORS_ORIGINS

    # CORS_ORIGINS is already a string from config_env
    cors_origins = [origin.strip() for origin in CORS_ORIGINS.split(",")]

    # Handle wildcard CORS with credentials (common dev requirement)
    # If "*" is present, we use allow_origin_regex to match any http/https origin
    # This allows credentials to work, which strict allow_origins=["*"] does not permit
    allow_origin_regex = None
    if "*" in cors_origins:
        cors_origins = []
        allow_origin_regex = r"https?://.*"

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,  # Configured via CORS_ORIGINS env var
        allow_origin_regex=allow_origin_regex,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(api_router, prefix="/api/v1")

    # Register standard error handlers
    register_error_handlers(app)

    # Register v1.3.0 exception handlers (VideoAnnotatorException -> ErrorEnvelope)
    from .v1.handlers import register_v1_exception_handlers

    register_v1_exception_handlers(app)

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Return basic health status information."""
        logger.debug("Health check requested")
        # Lightweight system metrics (avoid heavy psutil calls here)
        try:
            import psutil  # local import to keep import cost low

            mem = psutil.virtual_memory()
            memory_percent = mem.percent
        except Exception:
            memory_percent = None
        # Database quick status
        try:
            from .database import check_database_health

            db_ok, db_msg = check_database_health()
            db_status = {
                "status": "healthy" if db_ok else "unhealthy",
                "message": db_msg,
            }
        except Exception:
            db_status = {"status": "unknown", "message": "database check failed"}
        return {
            "status": "healthy",
            "api_version": API_VERSION,
            "videoannotator_version": videoannotator_version,
            "message": "VideoAnnotator API is running",
            "logging": "enhanced",
            "memory_percent": memory_percent,  # backward-compatible alias expected by some tests
            "database": db_status,
        }

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app", host="0.0.0.0", port=18011, reload=True, log_level="info"
    )
