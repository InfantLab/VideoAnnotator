"""
VideoAnnotator API v1 endpoints
"""

from fastapi import APIRouter

from .jobs import router as jobs_router
from .pipelines import router as pipelines_router
from .system import router as system_router
from .debug import router as debug_router

# Create main API router
api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(jobs_router, prefix="/jobs", tags=["jobs"])
api_router.include_router(pipelines_router, prefix="/pipelines", tags=["pipelines"])
api_router.include_router(system_router, prefix="/system", tags=["system"])
api_router.include_router(debug_router, prefix="/debug", tags=["debug"])