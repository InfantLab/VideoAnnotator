"""
VideoAnnotator API Server

FastAPI-based REST API for video annotation processing.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator

from ..version import __version__ as videoannotator_version
from .v1 import api_router


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Startup
    logger.info("VideoAnnotator API server starting up...")
    
    # TODO: Initialize database connections
    # TODO: Initialize pipeline cache
    # TODO: Initialize job queue connections
    
    yield
    
    # Shutdown
    logger.info("VideoAnnotator API server shutting down...")
    
    # TODO: Cleanup database connections
    # TODO: Cleanup pipeline resources
    # TODO: Cleanup job queue connections


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="VideoAnnotator API",
        description="Production-ready REST API for video annotation processing",
        version="1.2.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes
    app.include_router(api_router, prefix="/api/v1")
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Basic health check endpoint."""
        return {
            "status": "healthy",
            "api_version": "1.2.0",
            "videoannotator_version": videoannotator_version,
            "message": "VideoAnnotator API is running"
        }
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )