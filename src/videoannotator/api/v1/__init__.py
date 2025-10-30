"""VideoAnnotator API v1 endpoints."""

from fastapi import APIRouter

from .config import router as config_router
from .debug import router as debug_router
from .events import router as events_router
from .health import router as health_router
from .jobs import router as jobs_router
from .pipelines import router as pipelines_router
from .system import router as system_router

# Create main API router
api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(
    health_router, tags=["health"]
)  # No prefix - at /api/v1/health
api_router.include_router(jobs_router, prefix="/jobs", tags=["jobs"])
api_router.include_router(pipelines_router, prefix="/pipelines", tags=["pipelines"])
api_router.include_router(config_router, prefix="/config", tags=["config"])
api_router.include_router(system_router, prefix="/system", tags=["system"])
api_router.include_router(debug_router, prefix="/debug", tags=["debug"])
api_router.include_router(events_router, prefix="/events", tags=["events"])
