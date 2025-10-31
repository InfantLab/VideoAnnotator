"""Integration tests for enhanced logging functionality in v1.2.0.

Tests the integration of enhanced model download logging with actual
pipeline components, ensuring logging works correctly in realistic
scenarios.
"""

import logging
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Test the integration without importing heavy pipeline modules
from videoannotator.utils.model_loader import (
    log_first_run_info,
    log_model_download,
    setup_download_logging,
)


@pytest.mark.integration
class TestEnhancedLoggingIntegration:
    """Integration tests for enhanced logging with pipeline components."""

    def test_api_server_logging_setup(self, caplog):
        """Test that API server logging setup works correctly."""
        # Simulate what API server does
        with caplog.at_level(logging.INFO):
            setup_download_logging()
            log_first_run_info()

        # Check welcome message appears (note: should be ASCII-safe now)
        text = caplog.text
        assert "Welcome to VideoAnnotator" in text
        assert "FIRST RUN" in text

    def test_cli_logging_setup(self, caplog):
        """Test that CLI logging setup works correctly."""
        # Simulate what CLI does
        with caplog.at_level(logging.INFO):
            setup_download_logging()
            # Simulate checking if models directory exists
            models_dir = Path("./models")
            if not models_dir.exists():
                log_first_run_info()

    def test_person_pipeline_logging_integration(self, caplog):
        """Test enhanced logging integration with person pipeline loading.

        pattern.
        """

        def mock_yolo_constructor(model_path):
            """Mock YOLO constructor that simulates model loading behavior."""
            if not model_path.endswith(".pt"):
                raise ValueError(f"Invalid model path: {model_path}")

            # Simulate YOLO model structure
            model = Mock()
            model.names = {0: "person"}
            model.device = Mock()
            model.device.type = "cpu"
            return model

        with caplog.at_level(logging.INFO):
            result = log_model_download(
                "YOLO11 Pose Detection Model",
                "yolo11n-pose.pt",
                mock_yolo_constructor,
                "yolo11n-pose.pt",
            )

        # Check that the model was loaded correctly
        assert result.names == {0: "person"}

        # Check logging output matches person pipeline expectations
        log_messages = [record.message for record in caplog.records]
        assert any("YOLO11 Pose Detection Model" in msg for msg in log_messages)
        assert any("yolo11n-pose.pt" in msg for msg in log_messages)
        assert any("[OK]" in msg or "SUCCESS" in msg for msg in log_messages)  # Success indicator

    def test_whisper_pipeline_logging_integration(self, caplog):
        """Test enhanced logging integration with Whisper pipeline loading.

        pattern.
        """

        def mock_whisper_load_model(
            model_size, device="cpu", download_root=None, in_memory=True
        ):
            """Mock whisper.load_model that simulates Whisper loading
            behavior.
            """
            if model_size not in ["tiny", "base", "small", "medium", "large"]:
                raise ValueError(f"Invalid model size: {model_size}")

            # Simulate Whisper model structure
            model = Mock()
            model.dims = Mock()
            model.dims.n_mels = 80
            model.dims.n_audio_ctx = 1500 if model_size == "base" else 3000
            model.dims.n_vocab = 51864
            return model

        with caplog.at_level(logging.INFO):
            result = log_model_download(
                "OpenAI Whisper BASE Model",
                "whisper-base",
                mock_whisper_load_model,
                "base",
                device="cpu",
                download_root="./models/whisper",
                in_memory=True,
            )

        # Check that the model was loaded correctly
        assert result.dims.n_mels == 80
        assert result.dims.n_audio_ctx == 1500

        # Check logging output matches Whisper pipeline expectations
        log_messages = [record.message for record in caplog.records]
        assert any("Whisper" in msg for msg in log_messages)
        assert any("BASE" in msg for msg in log_messages)

    def test_multiple_pipeline_loading_sequence(self, caplog):
        """Test logging when multiple pipelines load models in sequence."""

        # Mock loaders for different model types
        def mock_yolo_loader(model_path):
            return Mock(names={0: "person"})

        def mock_whisper_loader(model_size, **kwargs):
            return Mock(dims=Mock(n_mels=80))

        def mock_clip_loader(model_name, **kwargs):
            return Mock(visual=Mock(), transformer=Mock())

        with caplog.at_level(logging.INFO):
            # Simulate loading sequence like in actual VideoAnnotator startup
            yolo_model = log_model_download(
                "YOLO11 Pose Detection",
                "yolo11n-pose.pt",
                mock_yolo_loader,
                "yolo11n-pose.pt",
            )

            whisper_model = log_model_download(
                "OpenAI Whisper Base", "whisper-base", mock_whisper_loader, "base"
            )

            clip_model = log_model_download(
                "OpenAI CLIP ViT-B/32", "ViT-B-32", mock_clip_loader, "ViT-B-32"
            )

        # Verify all models loaded
        assert yolo_model.names == {0: "person"}
        assert whisper_model.dims.n_mels == 80
        assert hasattr(clip_model, "visual")

        # Check logging shows clear sequence
        log_messages = [record.message for record in caplog.records]

        # Should see loading messages for all three models
        assert any("YOLO11" in msg for msg in log_messages)
        assert any("Whisper" in msg for msg in log_messages)
        assert any("CLIP" in msg for msg in log_messages)

        # Should see success messages for all (no emoji - Windows compat)
        success_messages = [msg for msg in log_messages if "[OK]" in msg or "SUCCESS" in msg]
        assert len(success_messages) >= 3

    def test_error_handling_in_pipeline_context(self, caplog):
        """Test error handling and logging in realistic pipeline failure.

        scenarios.
        """

        def failing_model_loader(*args, **kwargs):
            raise RuntimeError("CUDA out of memory")

        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            with caplog.at_level(logging.INFO):
                log_model_download(
                    "Large GPU Model",
                    "large-model.pt",
                    failing_model_loader,
                    "large-model.pt",
                )

        log_messages = [record.message for record in caplog.records]
        error_messages = [msg for msg in log_messages if "[ERROR]" in msg or "ERROR" in msg]
        assert len(error_messages) >= 1
        assert any("CUDA out of memory" in msg for msg in log_messages)

    def test_logging_with_actual_file_system(self, tmp_path, caplog):
        """Test logging behavior with real file system operations."""
        # Create a fake model file
        model_file = tmp_path / "test_model.pt"
        model_file.write_bytes(b"fake model data")

        def mock_loader_with_file_check(model_path):
            # Loader that checks if file exists
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            return Mock(loaded_from=model_path)

        with caplog.at_level(logging.INFO):
            result = log_model_download(
                "Test Model",
                str(model_file),
                mock_loader_with_file_check,
                str(model_file),
            )

        assert result.loaded_from == str(model_file)

        # Should show that file was found locally
        log_messages = [record.message for record in caplog.records]
        assert any("found locally" in msg for msg in log_messages)

    def test_logging_performance_impact(self, caplog):
        """Test that enhanced logging doesn't significantly impact.

        performance.
        """
        import time

        def fast_loader(*args, **kwargs):
            # Simulate a very fast model loader
            return Mock()

        # Time the loading with logging
        start_time = time.time()
        with caplog.at_level(logging.INFO):
            for i in range(10):
                log_model_download(f"Fast Model {i}", f"fast-{i}.pt", fast_loader)
        end_time = time.time()

        # Should complete quickly (less than 1 second for 10 loads)
        total_time = end_time - start_time
        assert total_time < 1.0

        # Should have logged all attempts
        log_messages = [record.message for record in caplog.records]
        success_messages = [msg for msg in log_messages if "[OK]" in msg or "SUCCESS" in msg]
        assert len(success_messages) == 10


@pytest.mark.integration
class TestLoggingInAPIContext:
    """Test logging integration in API server context."""

    def test_api_server_logging_compatibility(self, caplog):
        """Test that enhanced logging works correctly in API server context."""
        # Simulate API server startup logging
        with caplog.at_level(logging.INFO):
            setup_download_logging()

            # Simulate model loading that might happen in API endpoints
            def mock_on_demand_loader(*args, **kwargs):
                return Mock(api_ready=True)

            api_model = log_model_download(
                "API On-Demand Model", "api-model.pt", mock_on_demand_loader
            )

        assert api_model.api_ready is True

        # Should work without interfering with API logging
        log_messages = [record.message for record in caplog.records]
        assert any("API On-Demand Model" in msg for msg in log_messages)

    def test_concurrent_logging_safety(self, caplog):
        """Test that logging is safe for concurrent model loading scenarios."""
        import queue
        import threading

        results_queue = queue.Queue()

        def load_model_in_thread(model_id):
            try:

                def thread_loader(*args, **kwargs):
                    return Mock(thread_id=model_id)

                with caplog.at_level(logging.INFO):
                    result = log_model_download(
                        f"Thread Model {model_id}",
                        f"thread-{model_id}.pt",
                        thread_loader,
                    )
                results_queue.put((model_id, result, None))
            except Exception as e:
                results_queue.put((model_id, None, str(e)))

        # Start multiple threads loading models
        threads = []
        for i in range(3):
            thread = threading.Thread(target=load_model_in_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Check all succeeded
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        assert len(results) == 3
        for model_id, result, error in results:
            assert error is None
            assert result.thread_id == model_id


@pytest.mark.integration
class TestFirstRunExperience:
    """Test the complete first-run user experience with enhanced logging."""

    @patch("pathlib.Path.exists")
    def test_complete_first_run_flow(self, mock_exists, caplog, tmp_path):
        """Test complete first-run experience with model downloads."""
        # Simulate no models directory exists
        mock_exists.return_value = False

        with caplog.at_level(logging.INFO):
            # Step 1: Setup logging (like demo.py does)
            setup_download_logging()

            # Step 2: Show first-run info
            log_first_run_info()

            # Step 3: Simulate loading multiple models in sequence
            def create_mock_loader(model_type):
                def mock_loader(*args, **kwargs):
                    return Mock(model_type=model_type, ready=True)

                return mock_loader

            # Load models in typical startup order
            models_loaded = []

            # YOLO for person detection
            yolo_model = log_model_download(
                "YOLO11 Person Detection",
                "yolo11n-pose.pt",
                create_mock_loader("yolo"),
                "yolo11n-pose.pt",
            )
            models_loaded.append(yolo_model)

            # Whisper for speech recognition
            whisper_model = log_model_download(
                "Whisper Speech Recognition",
                "whisper-base",
                create_mock_loader("whisper"),
                "base",
            )
            models_loaded.append(whisper_model)

        # Verify all models loaded successfully
        assert len(models_loaded) == 2
        assert all(model.ready for model in models_loaded)

        # Check complete user experience in logs
        log_messages = [record.message for record in caplog.records]
        combined_log = " ".join(log_messages)

        # Should have welcome message (no emoji - Windows compat)
        assert any("Welcome" in msg or "WELCOME" in msg for msg in log_messages)

        # Should explain first-run process
        assert any("FIRST RUN" in msg for msg in log_messages)

        # Should show model loading progress
        assert "YOLO11" in combined_log
        assert "Whisper" in combined_log

        # Should show success for both models
        success_count = len([msg for msg in log_messages if "[OK]" in msg or "SUCCESS" in msg])
        assert success_count >= 2

    def test_subsequent_run_experience(self, caplog):
        """Test that subsequent runs have minimal logging overhead."""
        # Simulate models already exist (no first-run info)
        with caplog.at_level(logging.INFO):
            setup_download_logging()
            # Don't call log_first_run_info() - simulate already installed

            # Load a model that exists locally
            def mock_existing_loader(*args, **kwargs):
                return Mock(cached=True)

            model = log_model_download(
                "Cached Model", "/existing/path/model.pt", mock_existing_loader
            )

        assert model.cached is True

        # Should have minimal logging for subsequent runs
        log_messages = [record.message for record in caplog.records]

        # Should not have first-run welcome messages
        assert not any("Welcome" in msg for msg in log_messages)
        assert not any("FIRST RUN" in msg for msg in log_messages)

        # Should still show model loading progress
        assert any("Cached Model" in msg for msg in log_messages)
        assert any("[OK]" in msg or "SUCCESS" in msg for msg in log_messages)
