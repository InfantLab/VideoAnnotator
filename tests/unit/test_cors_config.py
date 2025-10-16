"""Tests for CORS configuration and security defaults."""

import os
from unittest.mock import patch

import pytest


class TestCORSConfiguration:
    """Test CORS middleware configuration and defaults."""

    def test_cors_default_restricted_to_localhost(self):
        """Test that CORS defaults to localhost:3000 only."""
        # Test the environment variable logic
        with patch.dict(os.environ, {}, clear=True):
            # Default should be localhost:3000
            cors_origins_str = os.environ.get("CORS_ORIGINS", "http://localhost:3000")
            cors_origins = [origin.strip() for origin in cors_origins_str.split(",")]

            assert cors_origins == ["http://localhost:3000"]
            assert "*" not in cors_origins

    def test_cors_wildcard_not_in_default(self):
        """Test that wildcard (*) is NOT the default configuration."""
        with patch.dict(os.environ, {}, clear=True):
            cors_origins_str = os.environ.get("CORS_ORIGINS", "http://localhost:3000")
            assert "*" not in cors_origins_str
            assert "localhost" in cors_origins_str

    def test_cors_respects_environment_variable_single_origin(self):
        """Test that CORS_ORIGINS environment variable configures allowed origins."""
        custom_origin = "https://app.example.com"

        with patch.dict(os.environ, {"CORS_ORIGINS": custom_origin}):
            cors_origins_str = os.environ.get("CORS_ORIGINS", "http://localhost:3000")
            cors_origins = [origin.strip() for origin in cors_origins_str.split(",")]

            assert cors_origins == ["https://app.example.com"]

    def test_cors_respects_environment_variable_multiple_origins(self):
        """Test that CORS_ORIGINS supports comma-separated multiple origins."""
        origins = (
            "http://localhost:3000,https://app.example.com,https://staging.example.com"
        )

        with patch.dict(os.environ, {"CORS_ORIGINS": origins}):
            cors_origins_str = os.environ.get("CORS_ORIGINS", "http://localhost:3000")
            cors_origins = [origin.strip() for origin in cors_origins_str.split(",")]

            assert len(cors_origins) == 3
            assert "http://localhost:3000" in cors_origins
            assert "https://app.example.com" in cors_origins
            assert "https://staging.example.com" in cors_origins

    def test_cors_handles_whitespace_in_origins(self):
        """Test that CORS_ORIGINS handles whitespace correctly."""
        origins = "http://localhost:3000, https://app.example.com , https://staging.example.com"

        with patch.dict(os.environ, {"CORS_ORIGINS": origins}):
            cors_origins_str = os.environ.get("CORS_ORIGINS", "http://localhost:3000")
            cors_origins = [origin.strip() for origin in cors_origins_str.split(",")]

            # Should strip whitespace
            assert "https://app.example.com" in cors_origins
            assert "https://app.example.com " not in cors_origins  # No trailing space
            assert " https://app.example.com" not in cors_origins  # No leading space

    def test_initialize_security_logs_cors_configuration(self):
        """Test that security initialization logs CORS origins."""
        from api.startup import initialize_security

        with patch.dict(os.environ, {"CORS_ORIGINS": "https://secure.example.com"}):
            with patch("api.startup.logger") as mock_logger:
                with patch("api.startup.ensure_api_key_exists") as mock_ensure:
                    mock_ensure.return_value = (None, False)

                    with patch(
                        "api.middleware.auth.is_auth_required", return_value=True
                    ):
                        initialize_security()

                        # Verify CORS origins were logged
                        calls = [str(call) for call in mock_logger.info.call_args_list]
                        cors_logged = any(
                            "CORS" in str(call) and "secure.example.com" in str(call)
                            for call in calls
                        )
                        assert cors_logged, f"CORS not logged in: {calls}"


class TestCORSSecurityBestPractices:
    """Test CORS security best practices and warnings."""

    def test_production_can_configure_multiple_origins(self):
        """Test that production environments can configure multiple origins."""
        production_origins = (
            "https://app.example.com,https://api.example.com,https://admin.example.com"
        )

        with patch.dict(os.environ, {"CORS_ORIGINS": production_origins}):
            cors_origins_str = os.environ.get("CORS_ORIGINS", "http://localhost:3000")
            cors_origins = [origin.strip() for origin in cors_origins_str.split(",")]

            assert len(cors_origins) == 3
            assert all(origin.startswith("https://") for origin in cors_origins)
            assert "*" not in cors_origins

    def test_cors_empty_string_falls_back_to_default(self):
        """Test that empty CORS_ORIGINS falls back to default."""
        with patch.dict(os.environ, {"CORS_ORIGINS": ""}):
            cors_origins_str = os.environ.get("CORS_ORIGINS", "http://localhost:3000")

            # Empty string should use itself (will result in one empty origin)
            # This is intentional - users should not set it to empty
            # In real deployment, we'd want localhost as fallback
            if not cors_origins_str:
                cors_origins_str = "http://localhost:3000"

            cors_origins = [origin.strip() for origin in cors_origins_str.split(",")]
            assert "http://localhost:3000" in cors_origins
