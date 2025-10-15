"""Configuration validation endpoints for VideoAnnotator API."""

import logging
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from api.errors import APIError
from validation.validator import ConfigValidator

logger = logging.getLogger("videoannotator.api")

router = APIRouter()


class ConfigValidationRequest(BaseModel):
    """Request for full configuration validation."""

    config: dict[str, Any]
    selected_pipelines: list[str]


class ConfigValidationResponse(BaseModel):
    """Response for config validation."""

    valid: bool
    errors: list[dict[str, Any]]
    warnings: list[dict[str, Any]]
    message: str
    pipelines_validated: list[str]


@router.post("/validate", response_model=ConfigValidationResponse)
async def validate_config(request: ConfigValidationRequest) -> ConfigValidationResponse:
    """Validate a complete job configuration against selected pipelines.

    This endpoint validates configuration for multiple pipelines at once,
    providing comprehensive error and warning reporting.

    Args:
        request: Configuration validation request with config and pipeline list

    Returns:
        Validation result with errors, warnings, and helpful messages

    Example:
        ```json
        POST /api/v1/config/validate
        {
            "config": {
                "person_tracking": {"confidence_threshold": 0.5},
                "audio_processing": {"sample_rate": 16000}
            },
            "selected_pipelines": ["person_tracking", "whisper_transcription"]
        }
        ```
    """
    try:
        validator = ConfigValidator()

        # Build per-pipeline configs from the request
        # The config can have pipeline-specific sections, or be shared
        pipeline_configs = {}
        for pipeline_name in request.selected_pipelines:
            # Check if config has a pipeline-specific section
            if pipeline_name in request.config:
                pipeline_configs[pipeline_name] = request.config[pipeline_name]
            else:
                # Use the whole config for this pipeline
                pipeline_configs[pipeline_name] = request.config

        results = validator.validate_batch(pipeline_configs)

        # Aggregate results
        all_errors = []
        all_warnings = []
        valid_pipelines = []
        invalid_pipelines = []

        for pipeline_name, result in results.items():
            if result.valid:
                valid_pipelines.append(pipeline_name)
            else:
                invalid_pipelines.append(pipeline_name)

            # Add pipeline context to errors
            for error in result.errors:
                error_dict = error.model_dump()
                error_dict["pipeline"] = pipeline_name
                all_errors.append(error_dict)

            # Add pipeline context to warnings
            for warning in result.warnings:
                warning_dict = warning.model_dump()
                warning_dict["pipeline"] = pipeline_name
                all_warnings.append(warning_dict)

        # Build message
        overall_valid = len(invalid_pipelines) == 0
        if overall_valid:
            message = f"Configuration is valid for all {len(valid_pipelines)} pipeline(s)"
            if all_warnings:
                message += f" ({len(all_warnings)} warning(s) total)"
        else:
            message = f"Configuration has errors in {len(invalid_pipelines)} pipeline(s): {', '.join(invalid_pipelines)}"

        return ConfigValidationResponse(
            valid=overall_valid,
            errors=all_errors,
            warnings=all_warnings,
            message=message,
            pipelines_validated=request.selected_pipelines,
        )

    except Exception as e:
        logger.error("Failed to validate config: %s", e)
        raise APIError(
            status_code=500,
            code="CONFIG_VALIDATE_FAILED",
            message="Failed to validate configuration",
            hint="Check server logs for details",
        )
