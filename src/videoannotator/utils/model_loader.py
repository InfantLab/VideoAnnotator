"""Model loading utilities with enhanced progress logging."""

import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def log_model_download(
    model_name: str, model_path: str, loader_func: Callable, *args, **kwargs
) -> Any:
    """Load a model with enhanced download progress logging.

    Args:
        model_name: Human-readable name of the model (e.g., "YOLO11 Pose Detection")
        model_path: Path/identifier of the model being loaded
        loader_func: Function to call for loading the model
        *args, **kwargs: Arguments to pass to loader_func

    Returns:
        The loaded model object
    """
    logger.info("=" * 60)
    logger.info(f"[LOAD] Loading {model_name}")
    logger.info(f"[PATH] Model: {model_path}")

    # Check if model file exists locally
    if isinstance(model_path, (str, Path)):
        model_file = Path(model_path)
        if model_file.exists():
            logger.info(f"[OK] Model file found locally: {model_file}")
        else:
            logger.info(
                f"[DOWNLOAD] Model not found locally - will download: {model_path}"
            )
            logger.info("[WAIT] This may take a few minutes on first run...")

            # Show expected download info
            if "yolo" in model_path.lower():
                logger.info("   Expected download size: ~50-200MB")
            elif "whisper" in model_path.lower():
                logger.info(
                    "   Expected download size: ~150MB-1.5GB (depends on model size)"
                )
            elif "clip" in model_path.lower():
                logger.info("   Expected download size: ~300-600MB")

    start_time = time.time()

    try:
        # Load the model
        logger.info(f"[INIT] Initializing {model_name}...")
        model = loader_func(*args, **kwargs)

        load_time = time.time() - start_time
        logger.info(f"[OK] {model_name} loaded successfully!")
        logger.info(f"[TIME] Load time: {load_time:.1f} seconds")
        logger.info("=" * 60)

        return model

    except Exception as e:
        load_time = time.time() - start_time
        logger.error(
            f"[ERROR] Failed to load {model_name} after {load_time:.1f} seconds"
        )
        logger.error(f"[ERROR] {e}")
        logger.info("=" * 60)
        raise


def setup_download_logging():
    """Configure logging to show model download progress clearly."""
    # Set up console handler with custom format for model downloads
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Custom formatter for model loading
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(formatter)

    # Configure loggers for key packages that download models
    for logger_name in [
        "ultralytics",
        "transformers",
        "whisper",
        "sentence_transformers",
        "huggingface_hub",
        "torch.hub",
        "__main__",
    ]:
        pkg_logger = logging.getLogger(logger_name)
        pkg_logger.setLevel(logging.INFO)
        if not pkg_logger.handlers:
            pkg_logger.addHandler(console_handler)
            pkg_logger.propagate = False


def log_first_run_info():
    """Display helpful information about first-run model downloads."""
    logger.info("[WELCOME] Welcome to VideoAnnotator v1.2.0!")
    logger.info("")
    logger.info("[FIRST RUN] Downloading AI Models")
    logger.info("   VideoAnnotator uses several AI models for video analysis:")
    logger.info("   - YOLO11: Person detection & pose estimation (~100MB)")
    logger.info("   - Whisper: Speech recognition (~150MB-1.5GB)")
    logger.info("   - OpenCLIP: Scene understanding (~300MB)")
    logger.info("   - PyAnnote: Audio diarization (~200MB)")
    logger.info("")
    logger.info("   [INFO] Models download automatically and are cached for future use")
    logger.info("   [INFO] Subsequent runs will be much faster!")
    logger.info("")
    logger.info("=" * 60)


class ModelDownloadProgress:
    """Context manager for showing model download progress."""

    def __init__(self, model_name: str):
        """Initialize progress tracker for the provided model name."""
        self.model_name = model_name
        self.start_time = None

    def __enter__(self):
        """Start timing the download and emit initial log messages."""
        self.start_time = time.time()
        logger.info(f"[START] Starting download: {self.model_name}")
        logger.info("   [WAIT] Please wait... (this may take several minutes)")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log the outcome of the download when the context exits."""
        if self.start_time:
            elapsed = time.time() - self.start_time
            if exc_type is None:
                logger.info(
                    f"[OK] {self.model_name} downloaded successfully in {elapsed:.1f} seconds"
                )
            else:
                logger.error(
                    f"[ERROR] {self.model_name} download failed after {elapsed:.1f} seconds"
                )
