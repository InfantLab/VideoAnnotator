"""Pipeline information endpoints for VideoAnnotator API."""

import logging
from typing import Any

from fastapi import APIRouter, Path
from pydantic import BaseModel

from ...registry.pipeline_registry import get_registry
from ..errors import APIError

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


@router.get("", include_in_schema=False)
@router.get(
    "/",
    response_model=PipelineListResponse,
    summary="List all available video annotation pipelines",
    description="""
Retrieves a comprehensive list of all registered pipelines available for video annotation,
including their metadata, configuration schemas, capabilities, and supported tasks.

Each pipeline includes:
- Basic metadata (name, display_name, description, family, variant)
- Task taxonomy (tasks, modalities, capabilities)
- Backend requirements and stability level
- Output formats and types
- Configuration schema with parameter types and defaults
- Usage examples

**Example Request:**
```bash
curl -X GET "http://localhost:18011/api/v1/pipelines" \\
  -H "X-API-Key: your-api-key-here"
```

**Success Response (200 OK):**
```json
{
  "pipelines": [
    {
      "name": "openface3_identity",
      "display_name": "OpenFace 3 - Identity",
      "description": "Face detection, tracking, and identity recognition",
      "pipeline_family": "openface",
      "variant": "identity",
      "tasks": ["face_detection", "face_tracking", "face_recognition"],
      "modalities": ["video"],
      "capabilities": ["detection", "tracking", "recognition"],
      "backends": ["openface"],
      "stability": "stable",
      "outputs": [
        {
          "format": "COCO",
          "types": ["detection", "tracking", "recognition"]
        }
      ],
      "config_schema": {
        "detection_confidence": {
          "type": "float",
          "default": 0.5,
          "description": "Minimum confidence threshold for face detection"
        }
      },
      "examples": [
        "videoannotator job submit --video input.mp4 --pipeline openface3_identity"
      ]
    }
  ],
  "total": 1
}
```

**Error Response (401 Unauthorized):**
```json
{
  "error": {
    "code": "UNAUTHORIZED",
    "message": "Invalid or missing API key"
  }
}
```

The pipeline list is dynamically loaded from the registry metadata, ensuring all
registered pipelines are discoverable. Use this endpoint to explore available
pipelines before submitting jobs.
""",
)
async def list_pipelines():
    """List all available pipelines."""
    try:
        reg = get_registry()
        reg.load()  # idempotent
        metas = reg.list()
        if not metas:
            logger.warning("Registry returned no pipelines; falling back to empty list")

        # Convert registry metadata to API response models
        # Note: We do NOT filter by available implementations here to avoid
        # heavy imports (torch, etc.) that would block the API for seconds/minutes.
        # The registry is the source of truth.
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
        ) from e


@router.get("/{pipeline_name}/", include_in_schema=False)
@router.get(
    "/{pipeline_name}",
    response_model=PipelineInfo,
    summary="Get detailed information about a specific pipeline",
    description="""
Retrieves comprehensive metadata and configuration details for a single pipeline
specified by name. Use this endpoint to explore pipeline capabilities, configuration
options, and usage examples before submitting jobs.

**Pipeline Information Includes:**
- Taxonomy: tasks, modalities, capabilities
- Backend requirements and stability level
- Output formats and annotation types
- Complete configuration schema with types, defaults, and descriptions
- Usage examples (CLI and API)

**Example Request:**
```bash
curl -X GET "http://localhost:18011/api/v1/pipelines/openface3_identity" \\
  -H "X-API-Key: your-api-key-here"
```

**Success Response (200 OK):**
```json
{
  "name": "openface3_identity",
  "display_name": "OpenFace 3 - Identity",
  "description": "Face detection, tracking, and identity recognition using OpenFace 3",
  "pipeline_family": "openface",
  "variant": "identity",
  "tasks": ["face_detection", "face_tracking", "face_recognition"],
  "modalities": ["video"],
  "capabilities": ["detection", "tracking", "recognition"],
  "backends": ["openface"],
  "stability": "stable",
  "outputs": [
    {
      "format": "COCO",
      "types": ["detection", "tracking", "recognition"]
    }
  ],
  "config_schema": {
    "detection_confidence": {
      "type": "float",
      "default": 0.5,
      "description": "Minimum confidence threshold for face detection (0.0-1.0)"
    },
    "enable_landmarks": {
      "type": "bool",
      "default": true,
      "description": "Extract facial landmarks for alignment"
    }
  },
  "examples": [
    "videoannotator job submit --video input.mp4 --pipeline openface3_identity",
    "videoannotator job submit --video input.mp4 --pipeline openface3_identity --config detection_confidence=0.7"
  ]
}
```

**Error Response (404 Not Found):**
```json
{
  "error": {
    "code": "PIPELINE_NOT_FOUND",
    "message": "Pipeline 'invalid_name' not found",
    "hint": "Run 'videoannotator pipelines --detailed' to list available pipelines"
  }
}
```

**Error Response (401 Unauthorized):**
```json
{
  "error": {
    "code": "UNAUTHORIZED",
    "message": "Invalid or missing API key"
  }
}
```

Use the configuration schema to validate parameters before job submission.
All configuration parameters are optional and have sensible defaults.
""",
)
async def get_pipeline_info(
    pipeline_name: str = Path(
        ...,
        description="The unique identifier/name of the pipeline (e.g., 'openface3_identity', 'whisper_transcription')",
        example="openface3_identity",
    ),
) -> PipelineInfo:
    """Get detailed information about a specific pipeline."""
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
        ) from e


class ConfigValidationRequest(BaseModel):
    """Request for config validation."""

    config: dict[str, Any]
    selected_pipelines: list[str] | None = None


class ConfigValidationResponse(BaseModel):
    """Response for config validation."""

    valid: bool
    errors: list[dict[str, Any]]
    warnings: list[dict[str, Any]]
    message: str


@router.post("/{pipeline_name}/validate", response_model=ConfigValidationResponse)
async def validate_pipeline_config(
    pipeline_name: str, request: ConfigValidationRequest
) -> ConfigValidationResponse:
    """Validate a pipeline configuration.

    Args:
        pipeline_name: Name of the pipeline
        request: Configuration validation request

    Returns:
        Validation result with errors and warnings
    """
    try:
        from videoannotator.validation.validator import ConfigValidator

        # Confirm pipeline exists
        _ = await get_pipeline_info(pipeline_name)

        # Validate the config
        validator = ConfigValidator()
        result = validator.validate(pipeline_name, request.config)

        # Build message
        if result.valid:
            message = "Configuration is valid"
            if result.warnings:
                message += f" ({len(result.warnings)} warning(s))"
        else:
            message = f"Configuration has {len(result.errors)} error(s)"

        return ConfigValidationResponse(
            valid=result.valid,
            errors=[e.model_dump() for e in result.errors],
            warnings=[w.model_dump() for w in result.warnings],
            message=message,
        )

    except APIError:
        raise
    except Exception as e:
        logger.error("Failed to validate config for '%s': %s", pipeline_name, e)
        raise APIError(
            status_code=500,
            code="PIPELINE_CONFIG_VALIDATE_FAILED",
            message="Failed to validate config",
            hint="Check schema / server logs",
        ) from e
