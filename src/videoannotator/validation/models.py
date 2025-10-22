"""Validation models for configuration validation.

These models provide structured error and warning reporting for configuration
validation, following the error envelope pattern defined in v1.3.0 roadmap.
"""

from typing import Any, ClassVar

from pydantic import BaseModel, Field


class FieldError(BaseModel):
    """Represents a configuration field error."""

    field: str = Field(..., description="Dotted path to the invalid field")
    message: str = Field(..., description="Human-readable error message")
    code: str = Field(..., description="Machine-readable error code")
    hint: str | None = Field(None, description="Optional hint for fixing the error")

    json_schema_extra: ClassVar[dict[str, Any]] = {
        "examples": [
            {
                "field": "person_tracking.confidence_threshold",
                "message": "Value must be between 0.0 and 1.0",
                "code": "VALUE_OUT_OF_RANGE",
                "hint": "Try using 0.5 as a sensible default",
            }
        ]
    }


class FieldWarning(BaseModel):
    """Represents a configuration field warning."""

    field: str = Field(..., description="Dotted path to the field with warning")
    message: str = Field(..., description="Human-readable warning message")
    suggestion: str | None = Field(
        None, description="Optional suggestion for improvement"
    )

    json_schema_extra: ClassVar[dict[str, Any]] = {
        "examples": [
            {
                "field": "audio_processing.speech_recognition.model",
                "message": "Using default model 'openai/whisper-base'",
                "suggestion": "Consider 'openai/whisper-medium' for better accuracy",
            }
        ]
    }


class ValidationResult(BaseModel):
    """Result of configuration validation."""

    valid: bool = Field(..., description="Whether the configuration is valid")
    errors: list[FieldError] = Field(
        default_factory=list, description="List of validation errors"
    )
    warnings: list[FieldWarning] = Field(
        default_factory=list, description="List of validation warnings"
    )

    json_schema_extra: ClassVar[dict[str, Any]] = {
        "examples": [
            {
                "valid": False,
                "errors": [
                    {
                        "field": "person_tracking.confidence_threshold",
                        "message": "Value must be between 0.0 and 1.0",
                        "code": "VALUE_OUT_OF_RANGE",
                        "hint": "Try using 0.5 as a sensible default",
                    }
                ],
                "warnings": [],
            },
            {"valid": True, "errors": [], "warnings": []},
        ]
    }

    @property
    def error_count(self) -> int:
        """Get the number of errors."""
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """Get the number of warnings."""
        return len(self.warnings)
