"""Tests for API startup and auto-generation functionality."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from videoannotator.api.startup import ensure_api_key_exists, initialize_security
from videoannotator.auth.token_manager import SecureTokenManager, TokenType


class TestAutoAPIKeyGeneration:
    """Test automatic API key generation on first startup."""

    def test_ensure_api_key_exists_with_existing_keys(self, tmp_path):
        """Test that no key is generated when keys already exist."""
        # Setup token manager with existing key
        tokens_dir = tmp_path / "tokens"
        tokens_dir.mkdir()

        with patch("api.startup.get_token_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager._token_cache = {
                "va_api_existing": MagicMock(
                    token_type=TokenType.API_KEY, is_active=True
                )
            }
            mock_get_manager.return_value = mock_manager

            # Mock Path.exists to return True
            with patch("api.startup.Path") as mock_path:
                mock_file = MagicMock()
                mock_file.exists.return_value = True
                mock_file.stat.return_value = MagicMock(st_size=100)
                mock_path.return_value = mock_file

                api_key, is_new = ensure_api_key_exists()

        # Should not generate new key
        assert api_key is None
        assert is_new is False

    def test_ensure_api_key_exists_generates_new_key(self, tmp_path):
        """Test that a new key is generated when none exist."""
        # Setup empty token manager
        tokens_dir = tmp_path / "tokens"
        tokens_dir.mkdir()

        with patch("api.startup.get_token_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager._token_cache = {}

            # Mock generate_api_key to return a test key
            test_key = "va_api_test123456"
            test_token_info = MagicMock()
            mock_manager.generate_api_key.return_value = (test_key, test_token_info)
            mock_get_manager.return_value = mock_manager

            # Mock Path to return empty/non-existent file
            with patch("api.startup.Path") as mock_path:
                mock_file = MagicMock()
                mock_file.exists.return_value = False
                mock_path.return_value = mock_file

                with patch("builtins.print"):  # Suppress console output
                    api_key, is_new = ensure_api_key_exists()

        # Should generate new key
        assert api_key == test_key
        assert is_new is True

        # Verify generate_api_key was called correctly
        mock_manager.generate_api_key.assert_called_once()
        call_kwargs = mock_manager.generate_api_key.call_args.kwargs
        assert call_kwargs["user_id"] == "admin"
        assert call_kwargs["username"] == "admin"
        assert call_kwargs["scopes"] == ["read", "write", "admin"]
        assert call_kwargs["expires_in_days"] is None

    def test_ensure_api_key_exists_respects_auto_generate_env(self, tmp_path):
        """Test that AUTO_GENERATE_API_KEY=false disables generation."""
        with patch.dict(os.environ, {"AUTO_GENERATE_API_KEY": "false"}):
            api_key, is_new = ensure_api_key_exists()

        assert api_key is None
        assert is_new is False

    def test_ensure_api_key_exists_handles_generation_failure(self, tmp_path):
        """Test graceful handling of key generation failure."""
        with patch("api.startup.get_token_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager._token_cache = {}
            mock_manager.generate_api_key.side_effect = Exception("Database error")
            mock_get_manager.return_value = mock_manager

            with patch("api.startup.Path") as mock_path:
                mock_file = MagicMock()
                mock_file.exists.return_value = False
                mock_path.return_value = mock_file

                with patch("builtins.print"):  # Suppress console output
                    api_key, is_new = ensure_api_key_exists()

        # Should return None on failure
        assert api_key is None
        assert is_new is False

    def test_initialize_security_calls_ensure_api_key(self):
        """Test that initialize_security calls ensure_api_key_exists."""
        with patch("api.startup.ensure_api_key_exists") as mock_ensure:
            mock_ensure.return_value = ("va_api_test", True)

            with patch("api.middleware.auth.is_auth_required", return_value=True):
                with patch.dict(os.environ, {"CORS_ORIGINS": "http://localhost:3000"}):
                    initialize_security()

        # Verify ensure_api_key_exists was called
        mock_ensure.assert_called_once()

    def test_initialize_security_logs_auth_disabled(self):
        """Test that security initialization logs when auth is disabled."""
        with patch("api.startup.ensure_api_key_exists") as mock_ensure:
            mock_ensure.return_value = (None, False)

            with patch("api.middleware.auth.is_auth_required", return_value=False):
                with patch("api.startup.logger") as mock_logger:
                    initialize_security()

                    # Verify warning logged
                    mock_logger.warning.assert_called_once()
                    warning_msg = mock_logger.warning.call_args[0][0]
                    assert "Authentication DISABLED" in warning_msg

    def test_generated_api_key_format(self):
        """Test that generated API keys have correct format."""
        from videoannotator.auth.token_manager import SecureTokenManager

        manager = SecureTokenManager()
        api_key, token_info = manager.generate_api_key(
            user_id="test", username="test", email="test@example.com"
        )

        # Verify format
        assert api_key.startswith("va_api_")
        assert len(api_key) > 40  # va_api_ + 32 urlsafe chars = ~47 chars

        # Verify token info
        assert token_info.token_type == TokenType.API_KEY
        assert token_info.user_id == "test"
        assert token_info.is_active is True
