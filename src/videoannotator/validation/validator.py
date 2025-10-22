"""Configuration validator for VideoAnnotator pipelines."""

from typing import Any, ClassVar

from .validation.models import FieldError, FieldWarning, ValidationResult


class ConfigValidator:
    """Validates pipeline configurations against known schemas.

    For v1.3.0, provides basic type and range validation with helpful error
    messages. Schema definitions are loaded from pipeline metadata in future versions.
    """

    # Common configuration schema rules (v1.3.0: basic validation)
    COMMON_RULES: ClassVar[dict[str, dict[str, Any]]] = {
        "confidence_threshold": {
            "type": (int, float),
            "range": (0.0, 1.0),
            "default": 0.5,
            "hint": "Confidence threshold should be between 0.0 and 1.0",
        },
        "iou_threshold": {
            "type": (int, float),
            "range": (0.0, 1.0),
            "default": 0.5,
            "hint": "IoU threshold should be between 0.0 and 1.0",
        },
        "threshold": {
            "type": (int, float),
            "range": (0.0, 100.0),
            "hint": "Threshold value should be a positive number",
        },
        "max_persons": {
            "type": int,
            "range": (1, 100),
            "hint": "Maximum persons should be between 1 and 100",
        },
        "max_faces": {
            "type": int,
            "range": (1, 100),
            "hint": "Maximum faces should be between 1 and 100",
        },
    }

    # Pipeline-specific field definitions
    # For v1.3.0: No strictly required fields - pipelines have sensible defaults
    PIPELINE_REQUIREMENTS: ClassVar[dict[str, dict[str, list[str]]]] = {
        "person_tracking": {
            "required": [],  # model_name has defaults
            "optional": [
                "model_name",
                "confidence_threshold",
                "iou_threshold",
                "track_mode",
                "tracker_type",
                "pose_format",
                "min_keypoint_confidence",
                "max_persons",
                "person_identity",
            ],
        },
        "scene_detection": {
            "required": [],  # threshold has defaults
            "optional": [
                "threshold",
                "min_scene_length",
                "model",
                "scene_labels",
                "extract_keyframes",
                "keyframe_format",
            ],
        },
        "face_analysis": {
            "required": [],  # backend has defaults
            "optional": [
                "backend",
                "face_confidence_threshold",
                "max_faces",
                "detect_emotions",
                "detect_age",
                "detect_gender",
                "detect_gaze",
                "detect_action_units",
                "deepface",
            ],
        },
        "audio_processing": {
            "required": [],
            "optional": [
                "speech_recognition",
                "speaker_diarization",
                "audio_classification",
            ],
        },
    }

    def __init__(self):
        """Initialize the config validator."""
        self._cached_schemas: dict[str, Any] = {}

    def validate(self, pipeline_name: str, config: dict[str, Any]) -> ValidationResult:
        """Validate a pipeline configuration.

        Args:
            pipeline_name: Name of the pipeline (e.g., 'person_tracking')
            config: Configuration dictionary to validate

        Returns:
            ValidationResult with errors and warnings
        """
        errors: list[FieldError] = []
        warnings: list[FieldWarning] = []

        # Check if pipeline is known
        if pipeline_name not in self.PIPELINE_REQUIREMENTS:
            errors.append(
                FieldError(
                    field="pipeline",
                    message=f"Unknown pipeline '{pipeline_name}'",
                    code="PIPELINE_NOT_FOUND",
                    hint=f"Available pipelines: {', '.join(self.PIPELINE_REQUIREMENTS.keys())}",
                )
            )
            return ValidationResult(valid=False, errors=errors, warnings=warnings)

        requirements = self.PIPELINE_REQUIREMENTS[pipeline_name]

        # Check required fields
        for required_field in requirements["required"]:
            if required_field not in config:
                errors.append(
                    FieldError(
                        field=f"{pipeline_name}.{required_field}",
                        message=f"Required field '{required_field}' is missing",
                        code="REQUIRED_FIELD_MISSING",
                        hint=f"Add '{required_field}' to your configuration",
                    )
                )

        # Validate field types and ranges
        for field, value in config.items():
            field_path = f"{pipeline_name}.{field}"

            # Skip nested objects for now (v1.3.0 simplification)
            if isinstance(value, dict):
                # Recursively validate nested configs
                nested_errors, nested_warnings = self._validate_nested(
                    field_path, value
                )
                errors.extend(nested_errors)
                warnings.extend(nested_warnings)
                continue

            # Check against common rules
            if field in self.COMMON_RULES:
                rule = self.COMMON_RULES[field]

                # Type check
                if "type" in rule:
                    expected_type = rule["type"]
                    if not isinstance(value, expected_type):
                        type_names = (
                            expected_type.__name__
                            if not isinstance(expected_type, tuple)
                            else " or ".join(t.__name__ for t in expected_type)
                        )
                        errors.append(
                            FieldError(
                                field=field_path,
                                message=f"Expected type {type_names}, got {type(value).__name__}",
                                code="INVALID_TYPE",
                                hint=rule.get("hint"),
                            )
                        )
                        continue

                # Range check
                if "range" in rule and isinstance(value, (int, float)):
                    min_val, max_val = rule["range"]
                    if not (min_val <= value <= max_val):
                        errors.append(
                            FieldError(
                                field=field_path,
                                message=f"Value {value} is out of range [{min_val}, {max_val}]",
                                code="VALUE_OUT_OF_RANGE",
                                hint=rule.get(
                                    "hint",
                                    f"Use a value between {min_val} and {max_val}",
                                ),
                            )
                        )

        # Check for unknown fields (warnings, not errors)
        all_known_fields = requirements["required"] + requirements["optional"]
        for field in config:
            if field not in all_known_fields and field not in self.COMMON_RULES:
                warnings.append(
                    FieldWarning(
                        field=f"{pipeline_name}.{field}",
                        message=f"Unknown field '{field}' will be ignored",
                        suggestion="Check spelling or remove if not needed",
                    )
                )

        return ValidationResult(
            valid=len(errors) == 0, errors=errors, warnings=warnings
        )

    def _validate_nested(
        self, parent_path: str, config: dict[str, Any]
    ) -> tuple[list[FieldError], list[FieldWarning]]:
        """Validate nested configuration objects.

        Args:
            parent_path: Dotted path to the parent field
            config: Nested configuration dictionary

        Returns:
            Tuple of (errors, warnings) lists
        """
        errors: list[FieldError] = []
        warnings: list[FieldWarning] = []

        for field, value in config.items():
            field_path = f"{parent_path}.{field}"

            # Skip nested objects (one level deep for v1.3.0)
            if isinstance(value, dict):
                continue

            # Check against common rules
            if field in self.COMMON_RULES:
                rule = self.COMMON_RULES[field]

                # Type check
                if "type" in rule:
                    expected_type = rule["type"]
                    if not isinstance(value, expected_type):
                        type_names = (
                            expected_type.__name__
                            if not isinstance(expected_type, tuple)
                            else " or ".join(t.__name__ for t in expected_type)
                        )
                        errors.append(
                            FieldError(
                                field=field_path,
                                message=f"Expected type {type_names}, got {type(value).__name__}",
                                code="INVALID_TYPE",
                                hint=rule.get("hint"),
                            )
                        )
                        continue

                # Range check
                if "range" in rule and isinstance(value, (int, float)):
                    min_val, max_val = rule["range"]
                    if not (min_val <= value <= max_val):
                        errors.append(
                            FieldError(
                                field=field_path,
                                message=f"Value {value} is out of range [{min_val}, {max_val}]",
                                code="VALUE_OUT_OF_RANGE",
                                hint=rule.get(
                                    "hint",
                                    f"Use a value between {min_val} and {max_val}",
                                ),
                            )
                        )

        return errors, warnings

    def validate_batch(
        self, configs: dict[str, dict[str, Any]]
    ) -> dict[str, ValidationResult]:
        """Validate multiple pipeline configurations at once.

        Args:
            configs: Dictionary mapping pipeline names to their configurations

        Returns:
            Dictionary mapping pipeline names to validation results
        """
        results = {}
        for pipeline_name, config in configs.items():
            results[pipeline_name] = self.validate(pipeline_name, config)
        return results
