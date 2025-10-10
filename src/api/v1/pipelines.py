"""Pipeline information endpoints for VideoAnnotator API."""

import logging
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from api.errors import APIError
from registry.pipeline_registry import get_registry

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
    tasks: list[str] = []
    modalities: list[str] = []
    capabilities: list[str] = []
    backends: list[str] = []
    stability: str | None = None
    outputs: list[dict[str, Any]]
    config_schema: dict[str, Any]
    examples: list[dict[str, Any]] = []


class PipelineListResponse(BaseModel):
    """Response for pipeline listing."""

    pipelines: list[PipelineInfo]
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
        pipeline_models: list[PipelineInfo] = [
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
                config_schema={
                    k: {
                        "type": v.type,
                        "default": v.default,
                        "description": v.description,
                    }
                    for k, v in m.config_schema.items()
                },
                examples=m.examples,
            )
            for m in metas
        ]
        return PipelineListResponse(
            pipelines=pipeline_models, total=len(pipeline_models)
        )
    except APIError:
        raise
    except Exception as e:  # fallback
        logger.error("Failed to list pipelines via registry: %s", e)
        raise APIError(
            status_code=500,
            code="PIPELINES_LIST_FAILED",
            message="Failed to list pipelines",
            hint="Check server logs for details",
        )


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
            raise APIError(
                status_code=404,
                code="PIPELINE_NOT_FOUND",
                message=f"Pipeline '{pipeline_name}' not found",
                hint="Run 'videoannotator pipelines --detailed'",
            )
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
            config_schema={
                k: {"type": v.type, "default": v.default, "description": v.description}
                for k, v in meta.config_schema.items()
            },
            examples=meta.examples,
        )
    except APIError:
        raise
    except Exception as e:
        logger.error("Failed to get pipeline info '%s': %s", pipeline_name, e)
        raise APIError(
            status_code=500,
            code="PIPELINE_INFO_FAILED",
            message="Failed to get pipeline info",
            hint="Check server logs",
        )


@router.post("/{pipeline_name}/validate")
async def validate_pipeline_config(pipeline_name: str, config: dict[str, Any]):
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
            "message": "Configuration is valid",
        }

    except APIError:
        raise
    except Exception:
        raise APIError(
            status_code=500,
            code="PIPELINE_CONFIG_VALIDATE_FAILED",
            message="Failed to validate config",
            hint="Check schema / server logs",
        )
