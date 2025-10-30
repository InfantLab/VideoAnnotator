"""Tests for diagnostic modules.

v1.3.0: Phase 11 - T070-T074
"""

import pytest


class TestSystemDiagnostics:
    """Tests for system diagnostics."""

    def test_diagnose_system_returns_dict(self):
        """Test that diagnose_system returns a dictionary."""
        from videoannotator.diagnostics.system import diagnose_system

        result = diagnose_system()

        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] in ["ok", "warning", "error"]

    def test_diagnose_system_includes_python_info(self):
        """Test that system diagnostics include Python information."""
        from videoannotator.diagnostics.system import diagnose_system

        result = diagnose_system()

        assert "python" in result
        assert "version" in result["python"]
        assert "executable" in result["python"]
        assert "platform" in result["python"]

    def test_diagnose_system_includes_ffmpeg_info(self):
        """Test that system diagnostics include FFmpeg information."""
        from videoannotator.diagnostics.system import diagnose_system

        result = diagnose_system()

        assert "ffmpeg" in result
        assert "installed" in result["ffmpeg"]
        # FFmpeg might not be installed, but key should exist

    def test_diagnose_system_includes_os_info(self):
        """Test that system diagnostics include OS information."""
        from videoannotator.diagnostics.system import diagnose_system

        result = diagnose_system()

        assert "os" in result
        assert "system" in result["os"]
        assert "release" in result["os"]


class TestGPUDiagnostics:
    """Tests for GPU diagnostics."""

    def test_diagnose_gpu_returns_dict(self):
        """Test that diagnose_gpu returns a dictionary."""
        from videoannotator.diagnostics.gpu import diagnose_gpu

        result = diagnose_gpu()

        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] in ["ok", "warning", "error"]

    def test_diagnose_gpu_includes_cuda_info(self):
        """Test that GPU diagnostics include CUDA information."""
        from videoannotator.diagnostics.gpu import diagnose_gpu

        result = diagnose_gpu()

        assert "cuda_available" in result
        assert "device_count" in result
        assert isinstance(result["cuda_available"], bool)
        assert isinstance(result["device_count"], int)

    def test_diagnose_gpu_handles_no_pytorch(self):
        """Test that GPU diagnostics handle missing PyTorch gracefully."""
        from videoannotator.diagnostics.gpu import diagnose_gpu

        result = diagnose_gpu()

        # Should not raise exception
        assert "status" in result
        # If PyTorch not installed, should have warning
        if not result.get("pytorch_version"):
            assert result["status"] in ["warning", "error"]


class TestStorageDiagnostics:
    """Tests for storage diagnostics."""

    def test_diagnose_storage_returns_dict(self):
        """Test that diagnose_storage returns a dictionary."""
        from videoannotator.diagnostics.storage import diagnose_storage

        result = diagnose_storage()

        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] in ["ok", "warning", "error"]

    def test_diagnose_storage_includes_disk_usage(self):
        """Test that storage diagnostics include disk usage information."""
        from videoannotator.diagnostics.storage import diagnose_storage

        result = diagnose_storage()

        assert "disk_usage" in result
        disk = result["disk_usage"]
        assert "total_gb" in disk
        assert "used_gb" in disk
        assert "free_gb" in disk
        assert "percent_used" in disk

    def test_diagnose_storage_checks_writable(self):
        """Test that storage diagnostics check write permissions."""
        from videoannotator.diagnostics.storage import diagnose_storage

        result = diagnose_storage()

        assert "writable" in result
        assert isinstance(result["writable"], bool)


class TestDatabaseDiagnostics:
    """Tests for database diagnostics."""

    def test_diagnose_database_returns_dict(self):
        """Test that diagnose_database returns a dictionary."""
        from videoannotator.diagnostics.database import diagnose_database

        result = diagnose_database()

        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] in ["ok", "warning", "error"]

    def test_diagnose_database_includes_connection_status(self):
        """Test that database diagnostics include connection status."""
        from videoannotator.diagnostics.database import diagnose_database

        result = diagnose_database()

        assert "connected" in result
        assert isinstance(result["connected"], bool)

    def test_diagnose_database_includes_path(self):
        """Test that database diagnostics include database path."""
        from videoannotator.diagnostics.database import diagnose_database

        result = diagnose_database()

        assert "database_path" in result


class TestDiagnosticsIntegration:
    """Integration tests for diagnostics."""

    def test_all_diagnostics_have_errors_warnings_lists(self):
        """Test that all diagnostic functions return errors and warnings lists."""
        from videoannotator.diagnostics import (
            diagnose_database,
            diagnose_gpu,
            diagnose_storage,
            diagnose_system,
        )

        for diag_func in [
            diagnose_system,
            diagnose_gpu,
            diagnose_storage,
            diagnose_database,
        ]:
            result = diag_func()
            assert "errors" in result
            assert "warnings" in result
            assert isinstance(result["errors"], list)
            assert isinstance(result["warnings"], list)

    def test_all_diagnostics_status_matches_errors(self):
        """Test that status reflects presence of errors."""
        from videoannotator.diagnostics import (
            diagnose_database,
            diagnose_gpu,
            diagnose_storage,
            diagnose_system,
        )

        for diag_func in [
            diagnose_system,
            diagnose_gpu,
            diagnose_storage,
            diagnose_database,
        ]:
            result = diag_func()
            if result["errors"]:
                assert result["status"] == "error"
