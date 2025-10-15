"""Integration tests for configuration validation API endpoints.

Tests the POST /api/v1/config/validate and /api/v1/pipelines/{name}/validate endpoints.
"""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


class TestConfigValidationEndpoints:
    """Integration tests for configuration validation endpoints."""

    def test_validate_single_pipeline_valid_config(self):
        """Test validating a valid configuration for a single pipeline."""
        response = client.post(
            "/api/v1/pipelines/person_tracking/validate",
            json={
                "config": {
                    "confidence_threshold": 0.5,
                    "tracking_algorithm": "sort",
                }
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert len(data["errors"]) == 0
        assert "valid" in data["message"].lower()

    def test_validate_single_pipeline_invalid_config(self):
        """Test validating an invalid configuration for a single pipeline."""
        response = client.post(
            "/api/v1/pipelines/person_tracking/validate",
            json={
                "config": {
                    "confidence_threshold": 1.5,  # Out of range
                }
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert len(data["errors"]) > 0
        assert any("confidence_threshold" in e["field"] for e in data["errors"])

    def test_validate_single_pipeline_with_warnings(self):
        """Test validation that produces warnings."""
        response = client.post(
            "/api/v1/pipelines/audio_processing/validate",
            json={
                "config": {
                    "sample_rate": 8000,  # Low sample rate - warning
                }
            },
        )

        assert response.status_code == 200
        data = response.json()
        # Should be valid but with warnings
        assert data["valid"] is True
        assert len(data["warnings"]) > 0

    def test_validate_unknown_pipeline(self):
        """Test validating config for an unknown pipeline."""
        response = client.post(
            "/api/v1/pipelines/nonexistent_pipeline/validate",
            json={"config": {}},
        )

        # Should return 404 for unknown pipeline
        assert response.status_code == 404

    def test_validate_batch_config_all_valid(self):
        """Test batch validation with all valid configs."""
        response = client.post(
            "/api/v1/config/validate",
            json={
                "config": {
                    "person_tracking": {"confidence_threshold": 0.5},
                    "audio_processing": {"sample_rate": 16000},
                },
                "selected_pipelines": ["person_tracking", "audio_processing"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert len(data["errors"]) == 0
        assert len(data["pipelines_validated"]) == 2

    def test_validate_batch_config_mixed_results(self):
        """Test batch validation with some valid and some invalid configs."""
        response = client.post(
            "/api/v1/config/validate",
            json={
                "config": {
                    "person_tracking": {"confidence_threshold": 1.5},  # Invalid
                    "audio_processing": {"sample_rate": 16000},  # Valid
                },
                "selected_pipelines": ["person_tracking", "audio_processing"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert len(data["errors"]) > 0
        # Errors should have pipeline context
        assert any(e.get("pipeline") == "person_tracking" for e in data["errors"])

    def test_validate_batch_config_shared_config(self):
        """Test batch validation with shared config (not pipeline-specific)."""
        response = client.post(
            "/api/v1/config/validate",
            json={
                "config": {
                    "confidence_threshold": 0.5,
                    "output_format": "json",
                },
                "selected_pipelines": ["person_tracking"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        # Should validate even with shared config structure
        assert "valid" in data

    def test_validate_empty_config(self):
        """Test validation with empty config."""
        response = client.post(
            "/api/v1/pipelines/person_tracking/validate",
            json={"config": {}},
        )

        assert response.status_code == 200
        data = response.json()
        # Empty config should be valid (uses defaults)
        assert data["valid"] is True

    def test_validate_config_with_extra_fields(self):
        """Test validation with extra unknown fields."""
        response = client.post(
            "/api/v1/pipelines/person_tracking/validate",
            json={
                "config": {
                    "confidence_threshold": 0.5,
                    "unknown_field": "value",
                }
            },
        )

        assert response.status_code == 200
        data = response.json()
        # Should have warning about unknown field
        assert len(data["warnings"]) > 0
        assert any("unknown_field" in w["field"] for w in data["warnings"])

    def test_validation_response_structure(self):
        """Test that validation response has correct structure."""
        response = client.post(
            "/api/v1/pipelines/person_tracking/validate",
            json={"config": {"confidence_threshold": 0.5}},
        )

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "valid" in data
        assert "errors" in data
        assert "warnings" in data
        assert "message" in data
        assert isinstance(data["errors"], list)
        assert isinstance(data["warnings"], list)

    def test_validation_error_structure(self):
        """Test that validation errors have correct structure."""
        response = client.post(
            "/api/v1/pipelines/person_tracking/validate",
            json={"config": {"confidence_threshold": 1.5}},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False

        # Check error structure
        error = data["errors"][0]
        assert "field" in error
        assert "message" in error
        assert "code" in error
        # May have optional hint
        if "hint" in error:
            assert isinstance(error["hint"], (str, type(None)))

    def test_batch_validation_pipeline_context(self):
        """Test that batch validation includes pipeline context in errors."""
        response = client.post(
            "/api/v1/config/validate",
            json={
                "config": {
                    "person_tracking": {"confidence_threshold": 1.5},
                    "audio_processing": {"sample_rate": -1},
                },
                "selected_pipelines": ["person_tracking", "audio_processing"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False

        # Each error should have pipeline context
        for error in data["errors"]:
            assert "pipeline" in error
            assert error["pipeline"] in ["person_tracking", "audio_processing"]

    def test_validate_complex_nested_config(self):
        """Test validation with complex nested configuration."""
        response = client.post(
            "/api/v1/pipelines/audio_processing/validate",
            json={
                "config": {
                    "sample_rate": 16000,
                    "speech_recognition": {
                        "model": "openai/whisper-base",
                        "language": "en",
                    },
                }
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "valid" in data
        # Complex configs should be handled gracefully

    def test_validate_multiple_pipelines_same_config(self):
        """Test validating multiple pipelines with the same config."""
        response = client.post(
            "/api/v1/config/validate",
            json={
                "config": {"confidence_threshold": 0.5},
                "selected_pipelines": ["person_tracking", "laion_face_detection"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        # Should validate the config for each pipeline
        assert len(data["pipelines_validated"]) == 2
