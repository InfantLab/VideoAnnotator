"""VideoAnnotator API v1 endpoints."""

from fastapi import APIRouter

from .auth import router as auth_router
from .batches import router as batches_router
from .config import router as config_router
from .datasets import router as datasets_router
from .debug import router as debug_router
from .endpoints.artifacts import router as artifacts_router
from .events import router as events_router
from .health import router as health_router
from .jobs import router as jobs_router
from .pipelines import router as pipelines_router
from .presets import router as presets_router
from .system import router as system_router

# Create main API router
api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(
    health_router, tags=["health"]
)  # No prefix - at /api/v1/health
api_router.include_router(auth_router, prefix="/auth", tags=["auth"])
api_router.include_router(jobs_router, prefix="/jobs", tags=["jobs"])
api_router.include_router(batches_router, prefix="/batches", tags=["batches"])
api_router.include_router(artifacts_router, prefix="/jobs", tags=["artifacts"])
api_router.include_router(pipelines_router, prefix="/pipelines", tags=["pipelines"])
api_router.include_router(datasets_router, prefix="/datasets", tags=["datasets"])
api_router.include_router(presets_router, prefix="/presets", tags=["presets"])
api_router.include_router(config_router, prefix="/config", tags=["config"])
api_router.include_router(system_router, prefix="/system", tags=["system"])
api_router.include_router(debug_router, prefix="/debug", tags=["debug"])
api_router.include_router(events_router, prefix="/events", tags=["events"])
