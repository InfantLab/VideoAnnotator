"""Configuration validator for VideoAnnotator pipelines."""

from typing import Any, ClassVar

from .models import FieldError, FieldWarning, ValidationResult


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
        "vlm_annotation": {
            "required": [],  # every field has a default
            "optional": [
                "prompt",
                "base_url",
                "model",
                "sampling_mode",
                "frame_interval_sec",
                "burst_offsets",
                "think",
                "request_timeout_sec",
                "keep_alive",
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

        requirements = self._requirements_for(pipeline_name)
        if requirements is None:
            errors.append(
                FieldError(
                    field="pipeline",
                    message=f"Unknown pipeline '{pipeline_name}'",
                    code="PIPELINE_NOT_FOUND",
                    hint=f"Available pipelines: {', '.join(self._known_pipelines())}",
                )
            )
            return ValidationResult(valid=False, errors=errors, warnings=warnings)

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

        # Job submission passes the whole job config, keyed by pipeline name.
        section = config.get(pipeline_name)
        own = section if isinstance(section, dict) else config
        base_url_error = _base_url_error(pipeline_name, own.get("base_url"))
        if base_url_error:
            errors.append(base_url_error)

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

    def _requirements_for(self, pipeline_name: str) -> dict[str, list[str]] | None:
        """The hand-written rules above, else the registry's config schema.
        Pipelines added since v1.3.0 (speaker_diarization, speech_recognition,
        face_openface3_embedding, ...) exist only in the registry, and job
        submission rejected them as unknown."""
        if pipeline_name in self.PIPELINE_REQUIREMENTS:
            return self.PIPELINE_REQUIREMENTS[pipeline_name]
        meta = _registry().get(pipeline_name)
        if meta is None:
            return None
        return {"required": [], "optional": list(meta.config_schema)}

    def _known_pipelines(self) -> list[str]:
        names = list(self.PIPELINE_REQUIREMENTS)
        names += [m.name for m in _registry().list() if m.name not in names]
        return names

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


def _base_url_error(pipeline_name: str, value: Any) -> FieldError | None:
    """A non-empty `base_url` must be an http(s) URL. Checked here so a bad
    one fails at submission, not minutes into the job as an opaque client
    error (e.g. a viewer that sent the prompt text as the URL)."""
    if value in (None, ""):
        return None
    from urllib.parse import urlsplit

    ok = isinstance(value, str) and not any(c.isspace() for c in value)
    if ok:
        try:
            parts = urlsplit(value)
            _ = parts.port  # raises on a non-numeric or out-of-range port
            ok = parts.scheme in ("http", "https") and bool(parts.hostname)
        except ValueError:
            ok = False
    if ok:
        return None
    shown = value if len(str(value)) <= 60 else f"{str(value)[:57]}..."
    return FieldError(
        field=f"{pipeline_name}.base_url",
        message=f"base_url {shown!r} is not an http(s) URL",
        code="INVALID_URL",
        hint="Use e.g. http://localhost:11434, or leave it empty for the server default",
    )


def _registry():
    from ..registry.pipeline_registry import get_registry

    registry = get_registry()
    registry.load()
    return registry
