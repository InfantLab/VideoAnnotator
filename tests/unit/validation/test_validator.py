"""Unit tests for configuration validator."""

import pytest

from videoannotator.validation.models import FieldError, FieldWarning, ValidationResult
from videoannotator.validation.validator import ConfigValidator


class TestValidationModels:
    """Test validation model classes."""

    def test_field_error_creation(self):
        """Test creating a FieldError."""
        error = FieldError(
            field="person_tracking.confidence_threshold",
            message="Value must be between 0.0 and 1.0",
            code="VALUE_OUT_OF_RANGE",
            hint="Try using 0.5 as a sensible default",
        )
        assert error.field == "person_tracking.confidence_threshold"
        assert error.code == "VALUE_OUT_OF_RANGE"
        assert error.hint is not None

    def test_field_warning_creation(self):
        """Test creating a FieldWarning."""
        warning = FieldWarning(
            field="audio_processing.model",
            message="Using default model",
            suggestion="Consider using a larger model",
        )
        assert warning.field == "audio_processing.model"
        assert warning.suggestion is not None

    def test_validation_result_valid(self):
        """Test creating a valid ValidationResult."""
        result = ValidationResult(valid=True, errors=[], warnings=[])
        assert result.valid is True
        assert result.error_count == 0
        assert result.warning_count == 0

    def test_validation_result_with_errors(self):
        """Test ValidationResult with errors."""
        error = FieldError(
            field="test.field",
            message="Test error",
            code="TEST_ERROR",
            hint=None,
        )
        result = ValidationResult(valid=False, errors=[error], warnings=[])
        assert result.valid is False
        assert result.error_count == 1
        assert result.warning_count == 0


class TestConfigValidator:
    """Test ConfigValidator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ConfigValidator()

    def test_validator_initialization(self):
        """Test validator initializes correctly."""
        assert self.validator is not None
        assert hasattr(self.validator, "_cached_schemas")

    def test_valid_person_tracking_config(self):
        """Test valid person_tracking configuration."""
        config = {
            "model_name": "yolo11n-pose.pt",
            "confidence_threshold": 0.5,
            "iou_threshold": 0.5,
            "track_mode": True,
        }
        result = self.validator.validate("person_tracking", config)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_missing_required_field(self):
        """Test that configs without explicitly required fields are valid (use defaults)."""
        # v1.3.0: No strictly required fields - pipelines have sensible defaults
        config = {
            "confidence_threshold": 0.5,
            # model_name not provided - will use default
        }
        result = self.validator.validate("person_tracking", config)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_confidence_threshold_out_of_range(self):
        """Test confidence threshold out of range."""
        config = {
            "model_name": "yolo11n-pose.pt",
            "confidence_threshold": 1.5,  # Invalid: > 1.0
        }
        result = self.validator.validate("person_tracking", config)
        assert result.valid is False
        assert len(result.errors) == 1
        assert result.errors[0].code == "VALUE_OUT_OF_RANGE"
        assert "0.0" in result.errors[0].message
        assert "1.0" in result.errors[0].message

    def test_confidence_threshold_negative(self):
        """Test negative confidence threshold."""
        config = {
            "model_name": "yolo11n-pose.pt",
            "confidence_threshold": -0.5,  # Invalid: < 0.0
        }
        result = self.validator.validate("person_tracking", config)
        assert result.valid is False
        assert len(result.errors) == 1
        assert result.errors[0].code == "VALUE_OUT_OF_RANGE"

    def test_invalid_type_for_confidence(self):
        """Test invalid type for confidence threshold."""
        config = {
            "model_name": "yolo11n-pose.pt",
            "confidence_threshold": "high",  # Invalid: should be float
        }
        result = self.validator.validate("person_tracking", config)
        assert result.valid is False
        assert len(result.errors) == 1
        assert result.errors[0].code == "INVALID_TYPE"
        assert "int or float" in result.errors[0].message

    def test_unknown_pipeline(self):
        """Test unknown pipeline returns error."""
        config = {"model_name": "test"}
        result = self.validator.validate("nonexistent_pipeline", config)
        assert result.valid is False
        assert len(result.errors) == 1
        assert result.errors[0].code == "PIPELINE_NOT_FOUND"
        assert "nonexistent_pipeline" in result.errors[0].message
        assert result.errors[0].hint is not None

    def test_unknown_field_warning(self):
        """Test unknown field generates warning."""
        config = {
            "model_name": "yolo11n-pose.pt",
            "unknown_field": "value",  # Not a known field
        }
        result = self.validator.validate("person_tracking", config)
        assert result.valid is True  # Still valid, just a warning
        assert len(result.warnings) == 1
        assert "unknown_field" in result.warnings[0].message

    def test_scene_detection_valid_config(self):
        """Test valid scene_detection configuration."""
        config = {
            "threshold": 30.0,
            "min_scene_length": 2.0,
            "model": "ViT-B/32",
        }
        result = self.validator.validate("scene_detection", config)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_scene_detection_missing_required(self):
        """Test scene_detection without threshold uses defaults."""
        # v1.3.0: No strictly required fields - pipelines have sensible defaults
        config = {
            "min_scene_length": 2.0,
            # threshold not provided - will use default
        }
        result = self.validator.validate("scene_detection", config)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_face_analysis_valid_config(self):
        """Test valid face_analysis configuration."""
        config = {
            "backend": "deepface",
            "face_confidence_threshold": 0.7,
            "max_faces": 10,
        }
        result = self.validator.validate("face_analysis", config)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_max_faces_out_of_range(self):
        """Test max_faces out of range."""
        config = {
            "backend": "deepface",
            "max_faces": 150,  # Invalid: > 100
        }
        result = self.validator.validate("face_analysis", config)
        assert result.valid is False
        assert len(result.errors) == 1
        assert result.errors[0].code == "VALUE_OUT_OF_RANGE"

    def test_nested_config_validation(self):
        """Test validation of nested configuration."""
        config = {
            "model_name": "yolo11n-pose.pt",
            "person_identity": {
                "enabled": True,
                "confidence_threshold": 1.5,  # Invalid nested value
            },
        }
        result = self.validator.validate("person_tracking", config)
        assert result.valid is False
        assert len(result.errors) == 1
        assert "person_identity.confidence_threshold" in result.errors[0].field

    def test_multiple_errors(self):
        """Test configuration with multiple errors."""
        config = {
            # model_name not provided - will use default (no error)
            "confidence_threshold": 1.5,  # Out of range
            "iou_threshold": -0.1,  # Out of range
        }
        result = self.validator.validate("person_tracking", config)
        assert result.valid is False
        assert len(result.errors) == 2  # 2 out of range errors

    def test_validate_batch(self):
        """Test validating multiple pipeline configs at once."""
        configs = {
            "person_tracking": {
                "model_name": "yolo11n-pose.pt",
                "confidence_threshold": 0.5,
            },
            "scene_detection": {
                "threshold": 30.0,
            },
        }
        results = self.validator.validate_batch(configs)
        assert len(results) == 2
        assert results["person_tracking"].valid is True
        assert results["scene_detection"].valid is True

    def test_validate_batch_with_errors(self):
        """Test batch validation with some invalid configs."""
        configs = {
            "person_tracking": {
                # model_name not provided - will use default (valid)
                "confidence_threshold": 0.5,
            },
            "scene_detection": {
                "threshold": 30.0,  # Valid
            },
        }
        results = self.validator.validate_batch(configs)
        # Both should be valid now (no strictly required fields)
        assert results["person_tracking"].valid is True
        assert results["scene_detection"].valid is True

    def test_audio_processing_minimal_config(self):
        """Test audio_processing with minimal config (no required fields)."""
        config = {}
        result = self.validator.validate("audio_processing", config)
        assert result.valid is True  # No required fields for audio_processing

    def test_audio_processing_with_nested(self):
        """Test audio_processing with nested speech_recognition config."""
        config = {
            "speech_recognition": {
                "model": "openai/whisper-base",
                "language": "auto",
            }
        }
        result = self.validator.validate("audio_processing", config)
        assert result.valid is True
