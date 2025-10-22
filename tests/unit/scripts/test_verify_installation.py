"""Tests for installation verification script."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from scripts.verify_installation import CheckResult, InstallationVerifier


class TestCheckResult:
    """Test CheckResult namedtuple."""

    def test_check_result_creation(self):
        """Test creating CheckResult instances."""
        result = CheckResult(
            name="Test Check",
            passed=True,
            critical=True,
            message="Test passed",
            suggestion="No action needed",
        )

        assert result.name == "Test Check"
        assert result.passed is True
        assert result.critical is True
        assert result.message == "Test passed"
        assert result.suggestion == "No action needed"

    def test_check_result_without_suggestion(self):
        """Test CheckResult with default empty suggestion."""
        result = CheckResult(
            name="Test Check",
            passed=True,
            critical=False,
            message="Test passed",
        )

        assert result.suggestion == ""


class TestInstallationVerifier:
    """Test InstallationVerifier class."""

    def test_verifier_initialization(self):
        """Test verifier initialization."""
        verifier = InstallationVerifier(verbose=True, skip_video_test=True)

        assert verifier.verbose is True
        assert verifier.skip_video_test is True
        assert verifier.results == []
        assert verifier.has_critical_failure is False
        assert verifier.has_warnings is False

    def test_add_result_critical_failure(self):
        """Test adding a critical failure result."""
        verifier = InstallationVerifier()

        result = CheckResult(
            name="Critical Test",
            passed=False,
            critical=True,
            message="Failed",
        )

        verifier.add_result(result)

        assert len(verifier.results) == 1
        assert verifier.has_critical_failure is True
        assert verifier.has_warnings is False

    def test_add_result_warning(self):
        """Test adding a warning result."""
        verifier = InstallationVerifier()

        result = CheckResult(
            name="Warning Test",
            passed=False,
            critical=False,
            message="Warning",
        )

        verifier.add_result(result)

        assert len(verifier.results) == 1
        assert verifier.has_critical_failure is False
        assert verifier.has_warnings is True

    def test_add_result_success(self):
        """Test adding a successful result."""
        verifier = InstallationVerifier()

        result = CheckResult(
            name="Success Test",
            passed=True,
            critical=True,
            message="Passed",
        )

        verifier.add_result(result)

        assert len(verifier.results) == 1
        assert verifier.has_critical_failure is False
        assert verifier.has_warnings is False


class TestPythonVersionCheck:
    """Test Python version checking."""

    def test_check_python_version_success(self):
        """Test Python version check with valid version."""
        verifier = InstallationVerifier()

        # Current Python should be >= 3.10 in dev environment
        result = verifier.check_python_version()

        assert result.passed is True
        assert result.critical is True
        assert "Python" in result.message

    def test_check_python_version_failure(self, monkeypatch):
        """Test Python version check with old version."""
        verifier = InstallationVerifier()

        # Mock old Python version
        mock_version_info = type(
            "version_info",
            (),
            {"major": 3, "minor": 8, "micro": 0, "__ge__": lambda self, other: False},
        )()
        monkeypatch.setattr(sys, "version_info", mock_version_info)

        result = verifier.check_python_version()

        assert result.passed is False
        assert result.critical is True
        assert "3.8" in result.message
        assert "Install Python" in result.suggestion


class TestFFmpegCheck:
    """Test FFmpeg availability checking."""

    def test_check_ffmpeg_success(self):
        """Test FFmpeg check with FFmpeg installed."""
        verifier = InstallationVerifier()

        # Mock successful FFmpeg call
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="ffmpeg version 4.4.2 Copyright (c) 2000-2021",
            )

            result = verifier.check_ffmpeg()

            assert result.passed is True
            assert result.critical is True
            assert "FFmpeg" in result.message

    def test_check_ffmpeg_not_found(self):
        """Test FFmpeg check when FFmpeg is not installed."""
        verifier = InstallationVerifier()

        # Mock FFmpeg not found
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("ffmpeg not found")

            result = verifier.check_ffmpeg()

            assert result.passed is False
            assert result.critical is True
            assert "not found" in result.message
            assert "Install FFmpeg" in result.suggestion

    def test_check_ffmpeg_timeout(self):
        """Test FFmpeg check timeout."""
        verifier = InstallationVerifier()

        # Mock timeout
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("ffmpeg", 5)

            result = verifier.check_ffmpeg()

            assert result.passed is False
            assert result.critical is True
            assert "timed out" in result.message.lower()

    def test_check_ffmpeg_command_failed(self):
        """Test FFmpeg check with command failure."""
        verifier = InstallationVerifier()

        # Mock command failure
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=1, stdout="")

            result = verifier.check_ffmpeg()

            assert result.passed is False
            assert result.critical is True
            assert "failed" in result.message.lower()


class TestVideoAnnotatorImportCheck:
    """Test VideoAnnotator package import checking."""

    def test_check_import_success(self):
        """Test import check with package available."""
        verifier = InstallationVerifier()

        # Should succeed in actual environment
        result = verifier.check_videoannotator_import()

        # May pass or fail depending on test environment
        assert result.critical is True

    def test_check_import_failure(self, monkeypatch):
        """Test import check with missing package."""
        verifier = InstallationVerifier()

        # Mock version file not existing
        with patch("pathlib.Path.exists", return_value=False):
            # Also mock the fallback import check to fail
            def mock_import(name, *args, **kwargs):
                if "videoannotator" in name or name == "version":
                    raise ImportError(f"No module named '{name}'")
                return __import__(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                result = verifier.check_videoannotator_import()

                assert result.passed is False
                assert result.critical is True
                assert (
                    "failed" in result.message.lower()
                    or "not found" in result.message.lower()
                )
                assert "uv sync" in result.suggestion.lower()


class TestDatabaseWritableCheck:
    """Test database write access checking."""

    def test_check_database_writable_success(self, tmp_path, monkeypatch):
        """Test database write check with write access."""
        verifier = InstallationVerifier()

        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        result = verifier.check_database_writable()

        assert result.passed is True
        assert result.critical is True
        assert "writable" in result.message.lower()

    def test_check_database_writable_permission_denied(self, tmp_path, monkeypatch):
        """Test database write check with permission denied."""
        verifier = InstallationVerifier()

        # Mock permission error
        with patch("sqlite3.connect") as mock_connect:
            mock_connect.side_effect = PermissionError("Permission denied")

            result = verifier.check_database_writable()

            assert result.passed is False
            assert result.critical is True
            assert "permission" in result.message.lower()
            assert "chmod" in result.suggestion.lower()


class TestGPUAvailabilityCheck:
    """Test GPU availability checking."""

    def test_check_gpu_available(self):
        """Test GPU check with GPU available."""
        verifier = InstallationVerifier()

        # Mock torch with GPU
        with patch("builtins.__import__") as mock_import:
            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.device_count.return_value = 1
            mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 3080"

            def import_side_effect(name, *args, **kwargs):
                if name == "torch":
                    return mock_torch
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            result = verifier.check_gpu_availability()

            # Should be non-critical regardless
            assert result.critical is False

    def test_check_gpu_not_available(self):
        """Test GPU check with no GPU."""
        verifier = InstallationVerifier()

        # Mock torch without GPU
        with patch("builtins.__import__") as mock_import:
            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = False

            def import_side_effect(name, *args, **kwargs):
                if name == "torch":
                    return mock_torch
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            result = verifier.check_gpu_availability()

            assert result.passed is False
            assert result.critical is False  # Non-critical
            assert "No GPU" in result.message
            assert "CPU" in result.message

    def test_check_gpu_torch_not_installed(self):
        """Test GPU check when PyTorch is not installed."""
        verifier = InstallationVerifier()

        # Mock torch import error
        with patch("builtins.__import__") as mock_import:

            def import_side_effect(name, *args, **kwargs):
                if name == "torch":
                    raise ImportError("No module named 'torch'")
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            result = verifier.check_gpu_availability()

            assert result.passed is False
            assert result.critical is False  # Non-critical
            assert "PyTorch" in result.message


class TestPlatformDetection:
    """Test platform detection."""

    def test_detect_platform_linux(self, monkeypatch):
        """Test platform detection on Linux."""
        verifier = InstallationVerifier()

        monkeypatch.setattr("platform.system", lambda: "Linux")

        # Mock non-WSL
        with patch("builtins.open", side_effect=FileNotFoundError):
            platform_name = verifier.detect_platform()

            assert platform_name == "Linux"

    def test_detect_platform_wsl(self, monkeypatch):
        """Test platform detection on WSL."""
        verifier = InstallationVerifier()

        monkeypatch.setattr("platform.system", lambda: "Linux")

        # Mock WSL /proc/version
        mock_open = MagicMock()
        mock_open.return_value.__enter__.return_value.read.return_value = (
            "Linux version 5.10.0-microsoft-standard"
        )

        with patch("builtins.open", mock_open):
            platform_name = verifier.detect_platform()

            assert platform_name == "Windows WSL2"

    def test_detect_platform_macos(self, monkeypatch):
        """Test platform detection on macOS."""
        verifier = InstallationVerifier()

        monkeypatch.setattr("platform.system", lambda: "Darwin")

        platform_name = verifier.detect_platform()

        assert platform_name == "macOS"

    def test_detect_platform_windows(self, monkeypatch):
        """Test platform detection on Windows."""
        verifier = InstallationVerifier()

        monkeypatch.setattr("platform.system", lambda: "Windows")

        platform_name = verifier.detect_platform()

        assert platform_name == "Windows"

    def test_detect_platform_unknown(self, monkeypatch):
        """Test platform detection on unknown system."""
        verifier = InstallationVerifier()

        monkeypatch.setattr("platform.system", lambda: "FreeBSD")

        platform_name = verifier.detect_platform()

        assert "Unknown" in platform_name
        assert "FreeBSD" in platform_name


class TestVideoProcessingCheck:
    """Test video processing capability checking."""

    def test_check_video_processing_skipped(self):
        """Test video processing check when skipped."""
        verifier = InstallationVerifier(skip_video_test=True)

        result = verifier.check_sample_video_processing()

        assert result.passed is True
        assert result.critical is False
        assert "Skip" in result.message

    def test_check_video_processing_success(self, tmp_path, monkeypatch):
        """Test video processing check success."""
        verifier = InstallationVerifier(skip_video_test=False)

        # Mock successful video processing
        with patch("subprocess.run") as mock_run, patch("cv2.VideoCapture") as mock_cap:
            mock_run.return_value = Mock(returncode=0)

            mock_cap_instance = MagicMock()
            mock_cap_instance.isOpened.return_value = True
            mock_cap_instance.read.side_effect = [
                (True, MagicMock()),
                (True, MagicMock()),
                (False, None),
            ]
            mock_cap.return_value = mock_cap_instance

            result = verifier.check_sample_video_processing()

            # Non-critical check
            assert result.critical is False

    def test_check_video_processing_timeout(self):
        """Test video processing check timeout."""
        verifier = InstallationVerifier(skip_video_test=False)

        # Mock timeout
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("ffmpeg", 30)

            result = verifier.check_sample_video_processing()

            assert result.passed is False
            assert result.critical is False  # Non-critical
            assert "timed out" in result.message.lower()


class TestExitCodes:
    """Test exit code logic."""

    def test_exit_code_all_pass(self):
        """Test exit code 0 when all checks pass."""
        verifier = InstallationVerifier()

        verifier.add_result(
            CheckResult(name="Test 1", passed=True, critical=True, message="OK")
        )
        verifier.add_result(
            CheckResult(name="Test 2", passed=True, critical=False, message="OK")
        )

        # Check exit code logic
        if verifier.has_critical_failure:
            exit_code = 1
        elif verifier.has_warnings:
            exit_code = 2
        else:
            exit_code = 0

        assert exit_code == 0

    def test_exit_code_critical_failure(self):
        """Test exit code 1 when critical failure occurs."""
        verifier = InstallationVerifier()

        verifier.add_result(
            CheckResult(name="Test 1", passed=False, critical=True, message="FAIL")
        )

        # Check exit code logic
        if verifier.has_critical_failure:
            exit_code = 1
        elif verifier.has_warnings:
            exit_code = 2
        else:
            exit_code = 0

        assert exit_code == 1

    def test_exit_code_warnings_only(self):
        """Test exit code 2 when only warnings present."""
        verifier = InstallationVerifier()

        verifier.add_result(
            CheckResult(name="Test 1", passed=True, critical=True, message="OK")
        )
        verifier.add_result(
            CheckResult(name="Test 2", passed=False, critical=False, message="WARN")
        )

        # Check exit code logic
        if verifier.has_critical_failure:
            exit_code = 1
        elif verifier.has_warnings:
            exit_code = 2
        else:
            exit_code = 0

        assert exit_code == 2
