"""Pipeline information endpoints for VideoAnnotator API."""

import logging
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Path, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ...database.database import get_db
from ...database.models import ExtrasInstallJob, ExtrasInstallJobStatus
from ...registry.pipeline_loader import extras_available, install_hint, known_extras
from ...registry.pipeline_registry import get_registry
from .. import extras_install
from ..errors import APIError
from ..middleware.auth import require_admin

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
    available: bool = True
    install_hint: str | None = None


class PipelineListResponse(BaseModel):
    """Response for pipeline listing."""

    pipelines: list[PipelineInfo]
    total: int
    restart_required: bool = False


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
async def list_pipelines(
    include_unavailable: bool = Query(
        False,
        description=(
            "Include pipelines whose required extras aren't installed. "
            "Each such entry has available=false and an install_hint."
        ),
    ),
):
    """List available pipelines.

    By default, pipelines whose `requires_extras` aren't installed are
    omitted entirely (FR-005). Pass `?include_unavailable=true` to see them
    too, each flagged with `available: false` and an `install_hint`.

    The availability check is a metadata-only lookup (`importlib.metadata`),
    not an actual import of torch/etc., so it stays cheap even though the
    old version of this endpoint deliberately skipped availability checks
    to avoid blocking on heavy imports.
    """
    try:
        reg = get_registry()
        reg.load()  # idempotent
        metas = reg.list()
        if not metas:
            logger.warning("Registry returned no pipelines; falling back to empty list")

        pipeline_models: list[PipelineInfo] = []
        for m in metas:
            available = extras_available(m.requires_extras)
            if not available and not include_unavailable:
                continue
            pipeline_models.append(
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
                    available=available,
                    install_hint=None if available else install_hint(m.requires_extras),
                )
            )
        return PipelineListResponse(
            pipelines=pipeline_models,
            total=len(pipeline_models),
            restart_required=extras_install.restart_required(),
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
        examples={
            "openface3_identity": {
                "summary": "Example pipeline name",
                "value": "openface3_identity",
            }
        },
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
        available = extras_available(meta.requires_extras)
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
            available=available,
            install_hint=None if available else install_hint(meta.requires_extras),
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


class ExtrasInstallTriggerResponse(BaseModel):
    """Response for `POST /extras/{extra}/install`."""

    job_id: str
    extra_name: str
    status: str


class ConflictingDistribution(BaseModel):
    """An already-imported distribution an install changed (spec 011 FR-006)."""

    name: str
    old_version: str
    new_version: str | None = None  # None: the install removed it


class ExtrasInstallJobResponse(BaseModel):
    """Response for `GET /extras/install-jobs/{job_id}`."""

    job_id: str
    extra_name: str
    status: str
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    command_output: str | None = None
    restart_required: bool = False
    # spec 011: set once a job completes in this server process. `live`
    # means the pipelines are usable now; `restart_required` means
    # `conflicting_distributions` were already imported. None for a job
    # completed before the last restart (it is active by now either way).
    activation: str | None = None
    conflicting_distributions: list[ConflictingDistribution] = []


@router.post(
    "/extras/{extra}/install",
    response_model=ExtrasInstallTriggerResponse,
    status_code=202,
    summary="Trigger installation of a pipeline extras group",
    description="""
Admin-only. Triggers installing one named `[project.optional-dependencies]` extras group
(e.g. `face`, `audio`, `scene`, `all`) so its pipeline(s) become available, without needing
terminal/shell access (specs/005-pipeline-extras-install).

Returns immediately with a trackable job id rather than waiting for the (potentially
multi-minute) install to finish -- poll `GET /extras/install-jobs/{job_id}` for progress.
Most installs activate immediately (`activation: "live"` on the job). One that changed a
package the server had already imported reports `activation: "restart_required"` and sets
the top-level `restart_required` on `GET /api/v1/pipelines`; see `POST /api/v1/system/restart`.
""",
)
async def install_extra(
    extra: str = Path(
        ..., description="The extras-group name to install, e.g. 'face', 'audio', 'all'"
    ),
    user: dict[str, Any] = Depends(require_admin),
    db: Session = Depends(get_db),
) -> ExtrasInstallTriggerResponse:
    """Trigger installation of a named extras group."""
    known = known_extras()
    if extra not in known:
        raise APIError(
            status_code=422,
            code="UNKNOWN_EXTRAS_GROUP",
            message=f"Unknown extras group '{extra}'.",
            hint=f"Known extras groups: {', '.join(known)}",
        )

    # Already satisfied (FR-011): resolve immediately, no subprocess.
    if extras_available([extra]):
        job = ExtrasInstallJob(
            extra_name=extra,
            requested_by_user_id=user.get("id"),
            status=ExtrasInstallJobStatus.COMPLETED,
            command_output=f"Extras group '{extra}' is already installed; nothing to do.",
            started_at=datetime.now(),
            finished_at=datetime.now(),
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return ExtrasInstallTriggerResponse(
            job_id=str(job.id), extra_name=job.extra_name, status=job.status
        )

    # Dedup (FR-010): reserve this extra_name before creating anything. If
    # another install for the same extra is already in flight, its row may
    # not be committed yet -- report "pending" rather than treating a
    # not-yet-visible row as a stale dedup entry (avoids a race where a
    # losing request clears the winner's reservation).
    provisional_job_id = str(uuid.uuid4())
    existing_job_id = extras_install.try_begin_install(extra, provisional_job_id)
    if existing_job_id is not None:
        existing = (
            db.query(ExtrasInstallJob)
            .filter(ExtrasInstallJob.id == existing_job_id)
            .first()
        )
        status_value = (
            existing.status if existing is not None else ExtrasInstallJobStatus.PENDING
        )
        return ExtrasInstallTriggerResponse(
            job_id=existing_job_id, extra_name=extra, status=status_value
        )

    try:
        job = ExtrasInstallJob(
            id=provisional_job_id,
            extra_name=extra,
            requested_by_user_id=user.get("id"),
            status=ExtrasInstallJobStatus.PENDING,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        extras_install.start_install(str(job.id), extra)
    except Exception as e:
        extras_install._end_install(extra)
        logger.error("Failed to start install for extras group '%s': %s", extra, e)
        raise APIError(
            status_code=500,
            code="EXTRAS_INSTALL_START_FAILED",
            message=f"Failed to start install for extras group '{extra}'",
            hint="Check server logs",
        ) from e

    return ExtrasInstallTriggerResponse(
        job_id=str(job.id), extra_name=job.extra_name, status=job.status
    )


@router.get(
    "/extras/install-jobs/{job_id}",
    response_model=ExtrasInstallJobResponse,
    summary="Check the status of an extras-group install job",
    description="""
Admin-only. Returns the current state of an install job created by
`POST /extras/{extra}/install`: `pending`, `running`, `completed`, or `failed`. On
`failed`, `command_output` carries the captured error output for diagnosis.
""",
)
async def get_extras_install_job(
    job_id: str = Path(
        ..., description="The install job identifier returned by the trigger endpoint"
    ),
    user: dict[str, Any] = Depends(require_admin),
    db: Session = Depends(get_db),
) -> ExtrasInstallJobResponse:
    """Check the status of an extras-group install job."""
    job = db.query(ExtrasInstallJob).filter(ExtrasInstallJob.id == job_id).first()
    if job is None:
        raise APIError(
            status_code=404,
            code="EXTRAS_INSTALL_JOB_NOT_FOUND",
            message=f"Install job '{job_id}' not found",
        )
    outcome = extras_install.activation_for(str(job.id))
    return ExtrasInstallJobResponse(
        job_id=str(job.id),
        extra_name=job.extra_name,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        command_output=job.command_output,
        restart_required=(outcome or {}).get("activation") == "restart_required",
        activation=(outcome or {}).get("activation"),
        conflicting_distributions=(outcome or {}).get("conflicting_distributions", []),
    )


class PipelineParameterOption(BaseModel):
    """One valid choice for an enum/multiselect parameter."""

    value: str
    label: str | None = None


class PipelineParameterSchema(BaseModel):
    """One config field, shaped for the viewer's dynamic job-creation form
    (video-annotation-viewer's `PipelineParameterSchema`, `src/types/pipelines.ts`).
    """

    name: str
    type: str
    label: str | None = None
    description: str | None = None
    default: Any = None
    enum: list[PipelineParameterOption] | None = None


class PipelineSchemaDescriptor(BaseModel):
    id: str
    name: str
    description: str | None = None
    group: str | None = None


class PipelineSchemaResponse(BaseModel):
    """Response for `GET /{pipeline_name}/schema`."""

    pipeline: PipelineSchemaDescriptor
    parameters: list[PipelineParameterSchema]


# Registry `config_schema` field `type` strings -> the viewer's
# `PipelineParameterType` union (src/types/pipelines.ts). Anything not
# listed here (e.g. "list") falls back to "object", rendered as a raw-JSON
# textarea in DynamicPipelineParameters.tsx — a safe fallback rather than a
# crash for a config-schema type the viewer doesn't have a dedicated widget
# for yet.
_PARAM_TYPE_MAP: dict[str, str] = {
    "string": "string",
    "boolean": "boolean",
    "integer": "integer",
    "float": "number",
}


def _to_parameter_schema(name: str, field: Any) -> "PipelineParameterSchema":
    """Map one registry `PipelineConfigField` to the viewer's parameter shape."""
    if field.enum:
        param_type = "enum"
    elif field.widget == "textarea":
        param_type = "text"
    else:
        param_type = _PARAM_TYPE_MAP.get(field.type, "object")
    return PipelineParameterSchema(
        name=name,
        type=param_type,
        label=name.replace("_", " ").title(),
        description=field.description or None,
        default=field.default,
        enum=(
            [PipelineParameterOption(value=v, label=v) for v in field.enum]
            if field.enum
            else None
        ),
    )


@router.get("/{pipeline_name}/schema/", include_in_schema=False)
@router.get(
    "/{pipeline_name}/schema",
    response_model=PipelineSchemaResponse,
    summary="Get a pipeline's config schema as job-creation form parameters",
    description=(
        "Same underlying data as `config_schema` on `GET /{pipeline_name}`, "
        "reshaped into the form the video-annotation-viewer's dynamic "
        "job-creation UI expects (typed parameters with optional enum "
        "choices), rather than the raw type/default/description dict."
    ),
)
async def get_pipeline_schema(
    pipeline_name: str = Path(..., description="The pipeline's unique name"),
) -> PipelineSchemaResponse:
    """Get a pipeline's config schema, shaped for the job-creation form."""
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
        return PipelineSchemaResponse(
            pipeline=PipelineSchemaDescriptor(
                id=meta.name,
                name=meta.display_name,
                description=meta.description,
                group=meta.pipeline_family,
            ),
            parameters=[
                _to_parameter_schema(k, v) for k, v in meta.config_schema.items()
            ],
        )
    except APIError:
        raise
    except Exception as e:
        logger.error("Failed to get pipeline schema '%s': %s", pipeline_name, e)
        raise APIError(
            status_code=500,
            code="PIPELINE_SCHEMA_FAILED",
            message="Failed to get pipeline schema",
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
