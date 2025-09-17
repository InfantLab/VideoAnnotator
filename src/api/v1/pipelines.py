"""
Pipeline information endpoints for VideoAnnotator API
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel
from ...registry.pipeline_registry import get_registry, PipelineMetadata
import logging

logger = logging.getLogger("videoannotator.api")

# TODO: Import config system after fixing dependencies
# from ...config import load_config


router = APIRouter()


class PipelineInfo(BaseModel):
    """Information about an available pipeline (extended taxonomy)."""
    name: str
    display_name: str | None = None
    description: str
    enabled: bool = True
    pipeline_family: str | None = None
    variant: str | None = None
    tasks: List[str] = []
    modalities: List[str] = []
    capabilities: List[str] = []
    backends: List[str] = []
    stability: str | None = None
    outputs: List[Dict[str, Any]]
    config_schema: Dict[str, Any]
    examples: List[Dict[str, Any]] = []


class PipelineListResponse(BaseModel):
    """Response for pipeline listing."""
    pipelines: List[PipelineInfo]
    total: int


@router.get("/", response_model=PipelineListResponse)
async def list_pipelines():
    """
    List all available pipelines.
    
    Returns:
        List of available pipelines with their configurations
    """
    try:
        reg = get_registry()
        reg.load()  # idempotent
        metas = reg.list()
        if not metas:
            logger.warning("Registry returned no pipelines; falling back to empty list")
        pipeline_models: List[PipelineInfo] = [
            PipelineInfo(
                name=m.name,
                display_name=m.display_name,
                description=m.description,
                pipeline_family=m.pipeline_family,
                variant=m.variant,
                tasks=m.tasks,
                modalities=m.modalities,
                capabilities=m.capabilities,
                backends=m.backends,
                stability=m.stability,
                outputs=[{"format": o.format, "types": o.types} for o in m.outputs],
                config_schema={k: {"type": v.type, "default": v.default, "description": v.description} for k, v in m.config_schema.items()},
                examples=m.examples,
            ) for m in metas
        ]
        return PipelineListResponse(pipelines=pipeline_models, total=len(pipeline_models))
    except Exception as e:  # fallback: consistent error envelope start (minimal)
        logger.error("Failed to list pipelines via registry: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to list pipelines: {e}")


@router.get("/{pipeline_name}", response_model=PipelineInfo)
async def get_pipeline_info(pipeline_name: str):
    """
    Get detailed information about a specific pipeline.
    
    Args:
        pipeline_name: Name of the pipeline
        
    Returns:
        Detailed pipeline information
    """
    try:
        reg = get_registry()
        meta = reg.get(pipeline_name)
        if not meta:
            raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_name}' not found")
        return PipelineInfo(
            name=meta.name,
            display_name=meta.display_name,
            description=meta.description,
            pipeline_family=meta.pipeline_family,
            variant=meta.variant,
            tasks=meta.tasks,
            modalities=meta.modalities,
            capabilities=meta.capabilities,
            backends=meta.backends,
            stability=meta.stability,
            outputs=[{"format": o.format, "types": o.types} for o in meta.outputs],
            config_schema={k: {"type": v.type, "default": v.default, "description": v.description} for k, v in meta.config_schema.items()},
            examples=meta.examples,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get pipeline info '%s': %s", pipeline_name, e)
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline info: {e}")


@router.post("/{pipeline_name}/validate")
async def validate_pipeline_config(pipeline_name: str, config: Dict[str, Any]):
    """
    Validate a pipeline configuration.
    
    Args:
        pipeline_name: Name of the pipeline
        config: Configuration to validate
        
    Returns:
        Validation result
    """
    try:
        # Confirm pipeline exists
        _ = await get_pipeline_info(pipeline_name)

        # TODO: Implement proper config validation against schema
        # For now, just check if the pipeline exists and return success

        return {
            "valid": True,
            "pipeline": pipeline_name,
            "message": "Configuration is valid"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to validate config: {str(e)}"
        )