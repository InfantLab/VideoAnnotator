"""Whisper Base Pipeline for VideoAnnotator.

Loads a Whisper model on the faster-whisper runtime (CTranslate2) and extracts
the audio it transcribes. `SpeechPipeline` does the transcription.
"""

import gc
import logging
import tempfile
import warnings
from pathlib import Path
from typing import Any

# Initialize librosa with safer settings to avoid numba crashes
import librosa
import numpy as np
import torch

from videoannotator.pipelines.base_pipeline import BasePipeline
from videoannotator.utils.model_loader import log_model_download

from ...provenance import ModelRef
from .ffmpeg_utils import check_ffmpeg_available
from .ffmpeg_utils import extract_audio_from_video as ffmpeg_extract

try:
    # Disable numba JIT compilation for librosa if causing issues
    import numba

    # Configure numba to use threading layer that's more stable
    numba.config.THREADING_LAYER = "omp"
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Suppress librosa warnings that might indicate instability
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

# Download sizes for the first-run message (Hugging Face Hub, model.bin).
MODEL_SIZES_MB = {
    "tiny": 75,
    "base": 148,
    "small": 484,
    "medium": 1528,
    "large-v3-turbo": 1622,
    "turbo": 1622,
    "large-v3": 3087,
    "large": 3087,
}

CUDA_ERROR_MARKERS = (
    "not compiled with cuda",
    "cuda driver",
    # torch built with CUDA on a machine without the NVIDIA driver
    "no nvidia driver",
    "no cuda gpus are available",
    "cublas",
    "cudnn",
    "cuda error",
    "cuda out of memory",
    "cuda failed",
)


def faster_whisper_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("faster_whisper") is not None


def model_repository(model: str) -> str:
    """The Hugging Face repository faster-whisper loads for a size name or id."""
    if "/" in model or Path(model).is_dir():
        return model
    from faster_whisper.utils import _MODELS

    if model not in _MODELS:
        raise ValueError(
            f"Unknown Whisper model '{model}'. Use one of "
            f"{', '.join(_MODELS)}, or a CTranslate2 repository on Hugging Face."
        )
    return _MODELS[model]


class WhisperBasePipeline(BasePipeline):
    """Base pipeline for Whisper-based audio processing tasks.

    Loads the model on the faster-whisper runtime and extracts audio from
    video. Subclasses transcribe.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the Whisper base pipeline with configuration settings.

        Args:
            config: Configuration dictionary with the following options:
                - whisper_model: faster-whisper size name or CTranslate2 repository
                  (default: "base")
                - compute_type: CTranslate2 compute type (default: "default", the
                  weights' own type where the device supports it)
                - sample_rate: Audio sample rate (default: 16000)
                - device: Device to use ("cpu", "cuda", "auto") (default: "auto")
                - use_fp16: False forces float32 (default: True)
                - normalize_audio: Normalize audio during preprocessing (default: True)
        """
        default_config = {
            "whisper_model": "base",
            "compute_type": "default",
            "sample_rate": 16000,  # Whisper's preferred sample rate
            "device": "auto",  # "cpu", "cuda", or "auto"
            "use_fp16": True,
            "normalize_audio": True,  # Normalize audio during preprocessing
        }

        merged_config = default_config.copy()
        if config:
            merged_config.update(config)

        super().__init__(merged_config)

        self.logger = logging.getLogger(__name__)
        self.is_initialized: bool = False
        self.whisper_model: Any | None = None
        self.device: torch.device = torch.device("cpu")
        # Fallback state flags (for diagnostics/tests)
        self.used_cuda_initially = False
        self.fell_back_to_cpu = False

    def initialize(self) -> None:
        """Initialize the Whisper model and resources."""
        if self.is_initialized:
            return

        # Setup device
        requested_device = self.config.get("device", "auto")
        explicit_cuda = requested_device == "cuda"
        if requested_device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.used_cuda_initially = True
            else:
                self.device = torch.device("cpu")
        else:
            if explicit_cuda:
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                    self.used_cuda_initially = True
                else:
                    # Explicit CUDA request but unavailable: fail fast (no silent fallback)
                    raise RuntimeError(
                        "CUDA device explicitly requested but not available"
                    )
            else:
                self.device = torch.device(requested_device)

        self.logger.info(f"Initializing WhisperBasePipeline with device: {self.device}")

        try:
            self._load_whisper_model()
        except ImportError as e:
            # Optional dependency: allow the pipeline to exist but remain uninitialized.
            # Downstream pipelines check `self.whisper_model` before inference.
            self.logger.warning(f"Whisper dependency unavailable: {e}")
            self.is_initialized = True
            return
        except RuntimeError as e:
            # Fallback only if CUDA was opportunistic (auto) and failed with a
            # CUDA-specific issue, such as CTranslate2 not finding cuDNN.
            if (
                self.used_cuda_initially
                and requested_device == "auto"
                and any(m in str(e).lower() for m in CUDA_ERROR_MARKERS)
            ):
                self.logger.warning(
                    f"Speech recognition couldn't use the GPU ({e}); running on CPU."
                )
                self.device = torch.device("cpu")
                self.fell_back_to_cpu = True
                self._model_refs.clear()
                try:
                    self._load_whisper_model()
                except Exception as retry_e:
                    raise RuntimeError(
                        f"Failed to load Whisper model after CPU fallback: {retry_e}"
                    ) from retry_e
            else:
                raise

        self.is_initialized = True
        self.logger.info("WhisperBasePipeline initialized successfully")

    def _compute_type(self) -> str:
        if not self.config.get("use_fp16", True):
            return "float32"
        return str(self.config.get("compute_type") or "default")

    def _load_whisper_model(self) -> None:
        """Load the configured model on the faster-whisper runtime.

        Weights go to the Hugging Face cache (inside the models directory).
        One second of silence is transcribed straight away: CTranslate2 loads
        some CUDA libraries only when it first runs, and a missing one should
        fail here, where it can fall back to CPU, not halfway through a job.
        """
        try:
            from faster_whisper import WhisperModel
            from faster_whisper.utils import download_model
        except ImportError as e:
            raise ImportError(
                "faster-whisper not available. Install the audio extra: "
                "pip install 'videoannotator[audio]'"
            ) from e

        model = str(self.config["whisper_model"])
        repository = model_repository(model)
        size = MODEL_SIZES_MB.get(model)
        hint = ""
        if self.device.type == "cpu" and model not in ("tiny", "base", "small"):
            hint = " On CPU a smaller model is much faster: whisper_model small."
        self.logger.info(
            f"Speech recognition model: {model} ({repository}"
            + (f", about {size} MB to download on first use" if size else "")
            + f").{hint}"
        )
        path = (
            model
            if Path(model).is_dir()
            else log_model_download(
                f"Whisper {model} (faster-whisper)", repository, download_model, model
            )
        )
        try:
            self.whisper_model = WhisperModel(
                path, device=self.device.type, compute_type=self._compute_type()
            )
            segments, _info = self.whisper_model.transcribe(
                np.zeros(16000, dtype=np.float32), beam_size=1, language="en"
            )
            list(segments)
        except Exception as e:
            self.whisper_model = None
            raise RuntimeError(str(e)) from e

        # CTranslate2 runs float16 weights as float32 where the device can't.
        quantization = (
            str(getattr(self.whisper_model.model, "compute_type", "")) or None
        )
        revision = None if Path(model).is_dir() else Path(path).name
        self._model_refs.append(
            ModelRef(
                repository,
                "huggingface",
                revision,
                "git" if revision else "unknown",
                None if revision else "local directory",
                quantization=quantization,
            )
        )
        self.logger.info(
            f"Whisper '{model}' loaded on {self.device} ({quantization or 'default'})"
        )

    def extract_audio_from_video(
        self, video_path: str | Path
    ) -> tuple[np.ndarray, int]:
        """Extract audio from video file and preprocess it.

        Uses FFmpeg for safer extraction, then librosa for processing.

        Args:
            video_path: Path to the video file

        Returns:
            Tuple of (audio_waveform, sample_rate)
        """
        try:
            video_path = Path(video_path)
            self.logger.info(f"Extracting audio from {video_path}")

            target_sr = self.config["sample_rate"]

            # Method 1: Try FFmpeg extraction first (safer for video files)
            if check_ffmpeg_available():
                self.logger.debug("Using FFmpeg for audio extraction")
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_audio_path = Path(temp_dir) / "temp_audio.wav"

                    # Extract audio to temporary WAV file using FFmpeg
                    extracted_path = ffmpeg_extract(
                        video_path=video_path,
                        output_path=temp_audio_path,
                        sample_rate=target_sr,
                        channels=1,  # Mono
                        format="wav",
                    )

                    if extracted_path and extracted_path.exists():
                        # Load the extracted audio file with librosa (safer than direct video loading)
                        audio, _loaded_sr = librosa.load(
                            str(extracted_path), sr=target_sr, mono=True
                        )

                        # Clean up is automatic with tempfile.TemporaryDirectory
                        self.logger.info(
                            f"Audio extracted via FFmpeg: {len(audio) / target_sr:.2f} seconds at {target_sr}Hz"
                        )
                    else:
                        raise RuntimeError("FFmpeg extraction failed")
            else:
                self.logger.warning(
                    "FFmpeg not available, falling back to librosa direct loading"
                )
                # Method 2: Fallback to direct librosa loading (may crash on some systems)
                audio, orig_sr = librosa.load(
                    str(video_path),
                    sr=None,  # Use original sample rate initially
                    mono=True,
                )

                # Resample if necessary
                if orig_sr != target_sr:
                    self.logger.info(
                        f"Resampling audio from {orig_sr}Hz to {target_sr}Hz"
                    )
                    audio = librosa.resample(
                        audio, orig_sr=orig_sr, target_sr=target_sr
                    )

            # Apply normalization if configured
            if self.config.get("normalize_audio", True):
                self.logger.debug("Normalizing audio")
                # Use safer normalization approach
                if np.max(np.abs(audio)) > 0:
                    audio = audio / np.max(np.abs(audio))

            self.logger.info(
                f"Audio extraction complete: {len(audio) / target_sr:.2f} seconds at {target_sr}Hz"
            )
            return audio, target_sr

        except Exception as e:
            self.logger.error(f"Error extracting audio from {video_path}: {e}")
            # Try to provide more specific error information
            if "librosa" in str(e).lower() or "numba" in str(e).lower():
                raise RuntimeError(
                    f"Audio processing library error. This may be due to librosa/numba compatibility issues. "
                    f"Consider installing FFmpeg for more stable audio extraction. Original error: {e}"
                ) from e
            else:
                raise RuntimeError(f"Failed to extract audio: {e}") from e

    def cleanup(self) -> None:
        """Clean up resources used by the pipeline."""
        # CTranslate2 frees its GPU memory when the model is released.
        self.whisper_model = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.is_initialized = False
        self.logger.info("WhisperBasePipeline resources cleaned up")

    def get_schema(self) -> dict[str, Any]:
        """Return the output schema for this pipeline."""
        return {
            "type": "whisper_base",
            "description": "Base Whisper pipeline for audio processing",
            "output_format": {"audio_info": "Dict[str, Any]"},
        }

    def process(
        self,
        video_path: str,
        start_time: float = 0.0,
        end_time: float | None = None,
        pps: float = 1.0,
        output_dir: str | None = None,
    ) -> list[dict[str, Any]]:
        """Extract the audio; subclasses transcribe it.

        Returns:
            One entry with the audio's duration and sample rate.
        """
        if not self.is_initialized:
            self.initialize()
        audio, sample_rate = self.extract_audio_from_video(video_path)
        return [{"duration": len(audio) / sample_rate, "sample_rate": sample_rate}]
