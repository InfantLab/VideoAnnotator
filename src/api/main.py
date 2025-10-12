"""VideoAnnotator API Server.

FastAPI-based REST API for video annotation processing.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.errors import register_error_handlers
from api.middleware import ErrorLoggingMiddleware, RequestLoggingMiddleware
from api.v1 import api_router
from utils.logging_config import get_logger
from version import __version__ as videoannotator_version

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

# Note: Logging is set up by api_server.py - no need to duplicate
logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Startup
    logger.info("VideoAnnotator API server starting up...", extra={"event": "startup"})

    # Run database migrations (v1.3.0)
    try:
        from database.migrations import migrate_to_v1_3_0

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
    logger.info(
        "Server configuration initialized",
        extra={
            "api_version": API_VERSION,
            "videoannotator_version": videoannotator_version,
            "logging": "enhanced",
            "middleware": ["CORS", "RequestLogging", "ErrorLogging"],
            "background_processing": "enabled",
        },
    )

    # Start background job processing
    from api.background_tasks import start_background_processing

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
    from api.background_tasks import stop_background_processing

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
    )

    # Add middleware in correct order (last added = first executed)

    # Error logging middleware (innermost)
    app.add_middleware(ErrorLoggingMiddleware)

    # Request logging middleware
    app.add_middleware(
        RequestLoggingMiddleware,
        exclude_paths={"/health", "/docs", "/redoc", "/openapi.json", "/favicon.ico"},
    )

    # CORS middleware (outermost)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(api_router, prefix="/api/v1")

    # Register standard error handlers
    register_error_handlers(app)

    # Register v1.3.0 exception handlers (VideoAnnotatorException -> ErrorEnvelope)
    from api.v1.handlers import register_v1_exception_handlers

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
            from api.database import check_database_health

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
