"""
Tests for model loading utilities with enhanced progress logging.

Tests the enhanced model download logging functionality introduced in v1.2.0,
including progress display, first-run information, and error handling.
"""

import pytest
import logging
import sys
import time
from unittest.mock import Mock, patch, call, MagicMock
from pathlib import Path
from io import StringIO

from src.utils.model_loader import (
    log_model_download,
    setup_download_logging,
    log_first_run_info
)


class TestLogModelDownload:
    """Test the log_model_download function."""
    
    def test_successful_model_loading(self, caplog):
        """Test successful model loading with logging."""
        # Mock loader function
        mock_model = Mock()
        mock_loader = Mock(return_value=mock_model)
        
        with caplog.at_level(logging.INFO):
            result = log_model_download(
                "Test Model",
                "test-model.pt",
                mock_loader,
                "model_arg"
            )
        
        # Check return value
        assert result == mock_model
        
        # Check loader was called correctly
        mock_loader.assert_called_once_with("model_arg")
        
        # Check logging output
        log_messages = [record.message for record in caplog.records]
        assert any("🤖 Loading Test Model" in msg for msg in log_messages)
        assert any("📁 Model: test-model.pt" in msg for msg in log_messages)
        assert any("✅ Test Model loaded successfully!" in msg for msg in log_messages)

    def test_model_loading_with_kwargs(self, caplog):
        """Test model loading with keyword arguments."""
        mock_model = Mock()
        mock_loader = Mock(return_value=mock_model)
        
        with caplog.at_level(logging.INFO):
            result = log_model_download(
                "Test Model",
                "test-model.pt", 
                mock_loader,
                "model_arg",
                device="cuda",
                precision="fp16"
            )
        
        # Check loader was called with correct arguments
        mock_loader.assert_called_once_with("model_arg", device="cuda", precision="fp16")
        assert result == mock_model

    def test_model_loading_failure(self, caplog):
        """Test model loading failure handling."""
        mock_loader = Mock(side_effect=RuntimeError("Model load failed"))
        
        with pytest.raises(RuntimeError, match="Model load failed"):
            with caplog.at_level(logging.INFO):
                log_model_download(
                    "Test Model",
                    "test-model.pt",
                    mock_loader,
                    "model_arg"
                )
        
        # Check error logging
        log_messages = [record.message for record in caplog.records]
        assert any("❌ Failed to load Test Model" in msg for msg in log_messages)

    def test_timing_measurement(self, caplog):
        """Test that loading time is measured and logged."""
        def slow_loader(*args, **kwargs):
            time.sleep(0.1)  # Simulate loading time
            return Mock()
        
        with caplog.at_level(logging.INFO):
            log_model_download(
                "Slow Model",
                "slow-model.pt",
                slow_loader,
                "model_arg"
            )
        
        # Check timing in logs
        log_messages = [record.message for record in caplog.records]
        timing_logs = [msg for msg in log_messages if "Load time:" in msg and "seconds" in msg]
        assert len(timing_logs) > 0
        
        # Extract timing - should be > 0.1 seconds
        timing_msg = timing_logs[0]
        assert "0." in timing_msg  # Should show fractional seconds

    def test_empty_args(self, caplog):
        """Test model loading with no additional arguments."""
        mock_model = Mock()
        mock_loader = Mock(return_value=mock_model)
        
        with caplog.at_level(logging.INFO):
            result = log_model_download(
                "Simple Model",
                "simple.pt",
                mock_loader
            )
        
        mock_loader.assert_called_once_with()
        assert result == mock_model

    def test_complex_model_path(self, caplog):
        """Test with complex model path."""
        mock_model = Mock()
        mock_loader = Mock(return_value=mock_model)
        
        complex_path = "models/yolo/yolo11n-pose.pt"
        
        with caplog.at_level(logging.INFO):
            log_model_download(
                "YOLO Pose Model",
                complex_path,
                mock_loader,
                complex_path
            )
        
        log_messages = [record.message for record in caplog.records]
        assert any(complex_path in msg for msg in log_messages)


class TestSetupDownloadLogging:
    """Test the setup_download_logging function."""
    
    def test_logging_setup(self):
        """Test logging configuration setup."""
        with patch('logging.StreamHandler') as mock_handler:
            mock_handler_instance = Mock()
            mock_handler.return_value = mock_handler_instance
            
            setup_download_logging()
            
            # Check handler was created
            mock_handler.assert_called_once()
            mock_handler_instance.setLevel.assert_called_with(logging.INFO)

    def test_logging_setup_with_custom_level(self):
        """Test logging setup with handler configuration."""
        with patch('logging.StreamHandler') as mock_handler:
            mock_handler_instance = Mock()
            mock_handler.return_value = mock_handler_instance
            
            setup_download_logging()
            
            # Check handler configuration
            mock_handler_instance.setLevel.assert_called_with(logging.INFO)
            mock_handler_instance.setFormatter.assert_called_once()

    def test_handler_setup(self):
        """Test that stream handler is configured."""
        # Create a temporary logger to test
        logger = logging.getLogger("test_model_loader")
        
        with patch('logging.basicConfig'):
            setup_download_logging()
        
        # Test that we can log without errors
        logger.info("Test message")


class TestLogFirstRunInfo:
    """Test the log_first_run_info function."""
    
    def test_first_run_info_display(self, caplog):
        """Test first-run information display."""
        with caplog.at_level(logging.INFO):
            log_first_run_info()
        
        log_messages = [record.message for record in caplog.records]
        
        # Check for key first-run messages
        assert any("🎉 Welcome to VideoAnnotator" in msg for msg in log_messages)
        assert any("FIRST RUN:" in msg for msg in log_messages)
        assert any("downloading" in msg.lower() or "download" in msg.lower() for msg in log_messages)

    def test_info_contains_helpful_tips(self, caplog):
        """Test that first-run info contains helpful information."""
        with caplog.at_level(logging.INFO):
            log_first_run_info()
        
        log_messages = [record.message for record in caplog.records]
        combined_message = " ".join(log_messages).lower()
        
        # Check for helpful information
        assert "models" in combined_message
        assert any(word in combined_message for word in ["download", "cache", "storage"])

    def test_formatting_consistency(self, caplog):
        """Test that first-run info formatting is consistent."""
        with caplog.at_level(logging.INFO):
            log_first_run_info()
        
        log_messages = [record.message for record in caplog.records]
        
        # Check for consistent formatting patterns
        separator_lines = [msg for msg in log_messages if "=" in msg and len(msg) > 20]
        assert len(separator_lines) >= 1  # Should have at least one separator
        
        # Check for emojis in welcome message
        welcome_messages = [msg for msg in log_messages if "Welcome" in msg]
        assert len(welcome_messages) >= 1
        assert any("🎉" in msg or "🤖" in msg for msg in welcome_messages)


class TestModelLoaderIntegration:
    """Integration tests for model loader functionality."""
    
    def test_realistic_yolo_loading_simulation(self, caplog):
        """Test simulating realistic YOLO model loading."""
        def mock_yolo_loader(model_path, device="cpu"):
            # Simulate YOLO loading behavior
            if not model_path.endswith('.pt'):
                raise ValueError(f"Invalid model path: {model_path}")
            
            model = Mock()
            model.device = device
            model.names = {0: 'person'}
            return model
        
        with caplog.at_level(logging.INFO):
            result = log_model_download(
                "YOLO11 Pose Detection Model",
                "yolo11n-pose.pt",
                mock_yolo_loader,
                "yolo11n-pose.pt",
                device="cuda"
            )
        
        assert result.device == "cuda"
        assert result.names == {0: 'person'}
        
        # Check logging output
        log_messages = [record.message for record in caplog.records]
        assert any("YOLO11 Pose Detection Model" in msg for msg in log_messages)

    def test_realistic_whisper_loading_simulation(self, caplog):
        """Test simulating realistic Whisper model loading."""
        def mock_whisper_loader(model_size, device="cpu", download_root=None):
            model = Mock()
            model.dims = Mock()
            model.dims.n_mels = 80
            model.dims.n_audio_ctx = 1500
            return model
        
        with caplog.at_level(logging.INFO):
            result = log_model_download(
                "OpenAI Whisper BASE Model",
                "whisper-base",
                mock_whisper_loader,
                "base",
                device="cuda",
                download_root="./models/whisper"
            )
        
        assert result.dims.n_mels == 80
        
        # Check logging includes model details
        log_messages = [record.message for record in caplog.records]
        assert any("Whisper" in msg for msg in log_messages)

    def test_error_recovery_simulation(self, caplog):
        """Test error recovery in model loading."""
        call_count = 0
        
        def flaky_loader(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Download failed")
            return Mock(name="recovered_model")
        
        # First call should fail
        with pytest.raises(ConnectionError):
            with caplog.at_level(logging.INFO):
                log_model_download(
                    "Flaky Model",
                    "flaky.pt",
                    flaky_loader
                )
        
        # Second call should succeed
        with caplog.at_level(logging.INFO):
            result = log_model_download(
                "Flaky Model",
                "flaky.pt", 
                flaky_loader
            )
        
        assert str(result) == "recovered_model" or result._mock_name == "recovered_model"

    @patch('pathlib.Path.exists')
    def test_first_run_detection(self, mock_exists, caplog):
        """Test first-run detection logic."""
        # Simulate models directory doesn't exist (first run)
        mock_exists.return_value = False
        
        with caplog.at_level(logging.INFO):
            log_first_run_info()
        
        log_messages = [record.message for record in caplog.records]
        assert any("FIRST RUN" in msg for msg in log_messages)


class TestLoggingConfiguration:
    """Test logging configuration and output formatting."""
    
    def test_log_level_filtering(self, caplog):
        """Test that different log levels work correctly."""
        mock_loader = Mock(return_value=Mock())
        
        # Test with INFO level
        with caplog.at_level(logging.INFO):
            log_model_download("Test Model", "test.pt", mock_loader)
        
        info_messages = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(info_messages) > 0
        
        # Test with WARNING level (should see fewer messages)
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            log_model_download("Test Model", "test.pt", mock_loader)
        
        warning_messages = [r for r in caplog.records if r.levelno >= logging.WARNING]
        info_messages_filtered = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(info_messages_filtered) == 0  # INFO messages should be filtered out

    def test_unicode_emoji_handling(self, caplog):
        """Test that unicode emojis are handled correctly."""
        mock_loader = Mock(return_value=Mock())
        
        with caplog.at_level(logging.INFO):
            log_model_download("Test Model", "test.pt", mock_loader)
        
        log_messages = [record.message for record in caplog.records]
        
        # Check that emojis are present and properly formatted
        emoji_messages = [msg for msg in log_messages if any(emoji in msg for emoji in ["🤖", "📁", "✅", "❌"])]
        assert len(emoji_messages) >= 2  # At least loading and success messages

    def test_long_model_names(self, caplog):
        """Test handling of very long model names."""
        long_name = "Very Long Model Name That Exceeds Normal Length Expectations For Testing Purposes"
        mock_loader = Mock(return_value=Mock())
        
        with caplog.at_level(logging.INFO):
            log_model_download(long_name, "long-name.pt", mock_loader)
        
        log_messages = [record.message for record in caplog.records]
        
        # Verify long name is handled gracefully
        assert any(long_name in msg for msg in log_messages)
        
        # Check that formatting isn't broken
        loading_messages = [msg for msg in log_messages if "🤖 Loading" in msg]
        assert len(loading_messages) >= 1