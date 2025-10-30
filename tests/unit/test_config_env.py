"""Tests for configuration environment variable management.

Validates that configuration values can be overridden via environment variables
and that defaults are correctly applied.

v1.3.0: Phase 11 - T064
"""

import os
from unittest.mock import patch

import pytest


class TestConfigEnvHelpers:
    """Test configuration helper functions."""

    def test_get_int_env_with_valid_value(self):
        """Test get_int_env returns integer from environment variable."""
        from videoannotator.config_env import get_int_env

        with patch.dict(os.environ, {"TEST_INT": "42"}):
            assert get_int_env("TEST_INT", 10) == 42

    def test_get_int_env_with_default(self):
        """Test get_int_env returns default when variable not set."""
        from videoannotator.config_env import get_int_env

        with patch.dict(os.environ, {}, clear=True):
            assert get_int_env("NONEXISTENT", 99) == 99

    def test_get_int_env_with_invalid_value(self):
        """Test get_int_env returns default for non-numeric string."""
        from videoannotator.config_env import get_int_env

        with patch.dict(os.environ, {"TEST_INT": "not_a_number"}):
            assert get_int_env("TEST_INT", 15) == 15

    def test_get_bool_env_with_true_values(self):
        """Test get_bool_env recognizes truthy strings."""
        from videoannotator.config_env import get_bool_env

        for val in ["true", "True", "TRUE", "1", "yes", "YES", "on", "ON"]:
            with patch.dict(os.environ, {"TEST_BOOL": val}):
                assert get_bool_env("TEST_BOOL", False) is True, (
                    f"Expected True for value: {val}"
                )

    def test_get_bool_env_with_false_values(self):
        """Test get_bool_env recognizes falsy strings."""
        from videoannotator.config_env import get_bool_env

        for val in ["false", "False", "FALSE", "0", "no", "NO", "off", "OFF"]:
            with patch.dict(os.environ, {"TEST_BOOL": val}):
                assert get_bool_env("TEST_BOOL", True) is False, (
                    f"Expected False for value: {val}"
                )

    def test_get_bool_env_with_default(self):
        """Test get_bool_env returns default when variable not set."""
        from videoannotator.config_env import get_bool_env

        with patch.dict(os.environ, {}, clear=True):
            assert get_bool_env("NONEXISTENT", True) is True
            assert get_bool_env("NONEXISTENT", False) is False

    def test_get_str_env_with_value(self):
        """Test get_str_env returns string from environment."""
        from videoannotator.config_env import get_str_env

        with patch.dict(os.environ, {"TEST_STR": "hello"}):
            assert get_str_env("TEST_STR", "default") == "hello"

    def test_get_str_env_with_default(self):
        """Test get_str_env returns default when variable not set."""
        from videoannotator.config_env import get_str_env

        with patch.dict(os.environ, {}, clear=True):
            assert get_str_env("NONEXISTENT", "fallback") == "fallback"


class TestWorkerConfiguration:
    """Test worker-related configuration values."""

    def test_max_concurrent_jobs_default(self):
        """Test MAX_CONCURRENT_JOBS has correct default."""
        # Reimport to get fresh value (in case env was already set)
        import importlib

        import videoannotator.config_env

        with patch.dict(os.environ, {}, clear=True):
            importlib.reload(videoannotator.config_env)
            from videoannotator.config_env import MAX_CONCURRENT_JOBS

            assert MAX_CONCURRENT_JOBS == 2

    def test_max_concurrent_jobs_from_env(self):
        """Test MAX_CONCURRENT_JOBS reads from environment."""
        import importlib

        import videoannotator.config_env

        with patch.dict(os.environ, {"MAX_CONCURRENT_JOBS": "5"}):
            importlib.reload(videoannotator.config_env)
            from videoannotator.config_env import MAX_CONCURRENT_JOBS

            assert MAX_CONCURRENT_JOBS == 5

    def test_worker_poll_interval_default(self):
        """Test WORKER_POLL_INTERVAL has correct default."""
        import importlib

        import videoannotator.config_env

        with patch.dict(os.environ, {}, clear=True):
            importlib.reload(videoannotator.config_env)
            from videoannotator.config_env import WORKER_POLL_INTERVAL

            assert WORKER_POLL_INTERVAL == 5

    def test_worker_poll_interval_from_env(self):
        """Test WORKER_POLL_INTERVAL reads from environment."""
        import importlib

        import videoannotator.config_env

        with patch.dict(os.environ, {"WORKER_POLL_INTERVAL": "10"}):
            importlib.reload(videoannotator.config_env)
            from videoannotator.config_env import WORKER_POLL_INTERVAL

            assert WORKER_POLL_INTERVAL == 10

    def test_max_job_retries_default(self):
        """Test MAX_JOB_RETRIES has correct default."""
        import importlib

        import videoannotator.config_env

        with patch.dict(os.environ, {}, clear=True):
            importlib.reload(videoannotator.config_env)
            from videoannotator.config_env import MAX_JOB_RETRIES

            assert MAX_JOB_RETRIES == 3

    def test_retry_delay_base_default(self):
        """Test RETRY_DELAY_BASE has correct default."""
        import importlib

        import videoannotator.config_env

        with patch.dict(os.environ, {}, clear=True):
            importlib.reload(videoannotator.config_env)
            from videoannotator.config_env import RETRY_DELAY_BASE

            assert RETRY_DELAY_BASE == 2.0


class TestConfigIntegration:
    """Test configuration integration with worker components."""

    def test_job_processor_allows_explicit_args(self):
        """Test JobProcessor accepts explicit configuration arguments."""
        from videoannotator.worker.job_processor import JobProcessor

        # Create processor with explicit arguments
        processor = JobProcessor(max_concurrent_jobs=4, poll_interval=10)

        assert processor.max_concurrent_jobs == 4
        assert processor.poll_interval == 10

    def test_job_processor_allows_override(self):
        """Test JobProcessor allows explicit override of env vars."""
        from videoannotator.worker.job_processor import JobProcessor

        # Create processor with explicit override
        processor = JobProcessor(max_concurrent_jobs=8)

        assert processor.max_concurrent_jobs == 8

    def test_background_manager_accepts_explicit_args(self):
        """Test BackgroundJobManager accepts explicit configuration arguments."""
        from videoannotator.api.background_tasks import BackgroundJobManager

        # Create manager with explicit arguments
        manager = BackgroundJobManager(max_concurrent_jobs=3, poll_interval=7)

        assert manager.max_concurrent_jobs == 3
        assert manager.poll_interval == 7

    def test_background_manager_allows_override(self):
        """Test BackgroundJobManager allows explicit override."""
        from videoannotator.api.background_tasks import BackgroundJobManager

        # Create manager with explicit override
        manager = BackgroundJobManager(max_concurrent_jobs=6)

        assert manager.max_concurrent_jobs == 6


class TestConfigPrinting:
    """Test configuration printing utility."""

    def test_print_config_runs_without_error(self):
        """Test print_config executes without errors."""
        from videoannotator.config_env import print_config

        # Should not raise
        print_config()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
