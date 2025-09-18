"""
Whisper Base Pipeline for VideoAnnotator

This module provides a base pipeline for Whisper-based audio processing tasks.
It serves as a foundation for both speech recognition and voice emotion analysis.
"""

import logging
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import tempfile
import os
import warnings

import numpy as np
import torch

# Initialize librosa with safer settings to avoid numba crashes
import librosa
try:
    # Disable numba JIT compilation for librosa if causing issues
    import numba
    # Configure numba to use threading layer that's more stable
    numba.config.THREADING_LAYER = 'omp'
except ImportError:
    pass

# Suppress librosa warnings that might indicate instability
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')

from pipelines.base_pipeline import BasePipeline
from .ffmpeg_utils import check_ffmpeg_available, extract_audio_from_video as ffmpeg_extract
from utils.model_loader import log_model_download

# Try to import both standard Whisper and HF Whisper
try:
    import whisper
    STANDARD_WHISPER_AVAILABLE = True
except ImportError:
    STANDARD_WHISPER_AVAILABLE = False
    logging.warning("Standard whisper not available. Install with: pip install openai-whisper")

try:
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    HF_WHISPER_AVAILABLE = True
except ImportError:
    HF_WHISPER_AVAILABLE = False
    logging.warning("Hugging Face transformers not available. Install with: pip install transformers")


class WhisperBasePipeline(BasePipeline):
    """
    Base pipeline for Whisper-based audio processing tasks.
    
    This class provides common functionality for loading Whisper models,
    processing audio, and extracting embeddings. It supports both the
    standard OpenAI Whisper package and Hugging Face Whisper models.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Whisper base pipeline with configuration settings.
        
        Args:
            config: Configuration dictionary with the following options:
                - whisper_model: Model ID or size (default: "base")
                - sample_rate: Audio sample rate (default: 16000)
                - device: Device to use ("cpu", "cuda", "auto") (default: "auto")
                - use_fp16: Use half precision when possible (default: True)
                - cache_dir: Model cache directory (default: "./models/whisper")
                - use_auth_token: Use HF auth token for gated models (default: False)
                - normalize_audio: Normalize audio during preprocessing (default: True)
        """
        default_config = {
            "whisper_model": "base",  # Standard Whisper model or HF model ID
            "sample_rate": 16000,     # Whisper's preferred sample rate
            "device": "auto",         # "cpu", "cuda", or "auto"
            "use_fp16": True,         # Use half precision when possible
            "cache_dir": "./models/whisper",  # Local cache for models
            "use_auth_token": False,  # Use HF auth token for gated models
            "normalize_audio": True,  # Normalize audio during preprocessing
        }
        
        merged_config = default_config.copy()
        if config:
            merged_config.update(config)
        
        super().__init__(merged_config)
        
        self.logger = logging.getLogger(__name__)
        self.whisper_model = None
        self.whisper_processor = None
        self.device = None
        self.model_type = None  # "standard" or "huggingface"
        # Fallback state flags (for diagnostics/tests)
        self.used_cuda_initially = False
        self.fell_back_to_cpu = False
        
    def initialize(self) -> None:
        """Initialize the Whisper model and resources."""
        if self.is_initialized:
            return
            
        # Setup device
        requested_device = self.config.get("device", "auto")
        explicit_cuda = (requested_device == "cuda")
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
                    raise RuntimeError("CUDA device explicitly requested but not available")
            else:
                self.device = torch.device(requested_device)
            
        self.logger.info(f"Initializing WhisperBasePipeline with device: {self.device}")
            
        # Load Whisper model with CUDA fallback handling
        try:
            self._load_whisper_model()
        except RuntimeError as e:
            err_lower = str(e).lower()
            cuda_markers = [
                "not compiled with cuda",
                "cuda driver",
                "no cuda gpus are available",
                "cublas",
                "cudnn",
                "cuda error",
                "cuda out of memory"
            ]
            # Fallback only if CUDA was opportunistic (auto) and failed with a CUDA-specific issue
            if self.used_cuda_initially and requested_device == "auto" and any(m in err_lower for m in cuda_markers):
                self.logger.warning("Opportunistic CUDA initialization failed (%s). Falling back to CPU and retrying...", e)
                self.device = torch.device("cpu")
                self.fell_back_to_cpu = True
                try:
                    self._load_whisper_model()
                except Exception as retry_e:
                    raise RuntimeError(f"Failed to load Whisper model after CPU fallback: {retry_e}") from retry_e
            else:
                # Explicit CUDA request or non-CUDA error: re-raise
                raise
        
        self.is_initialized = True
        self.logger.info(f"WhisperBasePipeline initialized successfully")
    
    def _load_whisper_model(self) -> None:
        """
        Load the appropriate Whisper model based on configuration.
        
        This method supports both standard Whisper and Hugging Face variants.
        """
        model_id = self.config["whisper_model"]
        cache_dir = self.config.get("cache_dir")
        
        # Determine model type (HuggingFace or standard)
        if "/" in model_id:  # Likely a Hugging Face model
            if not HF_WHISPER_AVAILABLE:
                raise ImportError(
                    "Hugging Face transformers not available. "
                    "Install with: pip install transformers"
                )
            self._load_hf_whisper_model(model_id, cache_dir)
            self.model_type = "huggingface"
        else:  # Standard Whisper model
            if not STANDARD_WHISPER_AVAILABLE:
                raise ImportError(
                    "Standard whisper not available. "
                    "Install with: pip install openai-whisper"
                )
            self._load_standard_whisper_model(model_id)
            self.model_type = "standard"
    
    def _load_hf_whisper_model(self, model_id: str, cache_dir: Optional[str] = None) -> None:
        """
        Load a Whisper model from Hugging Face.
        
        Args:
            model_id: Hugging Face model ID
            cache_dir: Optional cache directory for model files
        """
        try:
            self.logger.info(f"Loading Hugging Face Whisper model: {model_id}")
            
            # Check for auth token if needed
            use_auth_token = self.config.get("use_auth_token", False)
            auth_token = None
            
            if use_auth_token:
                import os
                auth_token = os.getenv("HF_AUTH_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
                if not auth_token:
                    self.logger.warning("HF_AUTH_TOKEN environment variable not set but use_auth_token=True")
            
            # Load processor
            processor_kwargs = {"cache_dir": cache_dir}
            if auth_token:
                processor_kwargs["use_auth_token"] = auth_token
                
            self.whisper_processor = WhisperProcessor.from_pretrained(
                model_id, **processor_kwargs
            )
            
            # Load model
            model_kwargs = {"cache_dir": cache_dir}
            if auth_token:
                model_kwargs["use_auth_token"] = auth_token
                
            # Add FP16 if requested and on GPU
            if self.config.get("use_fp16", True) and self.device.type == "cuda":
                model_kwargs["torch_dtype"] = torch.float16
                
            model = WhisperForConditionalGeneration.from_pretrained(
                model_id, **model_kwargs
            )
            # Handle meta tensor properly for newer PyTorch versions
            try:
                self.whisper_model = model.to(self.device)
            except RuntimeError as e:
                if "meta tensor" in str(e):
                    self.whisper_model = model.to_empty(device=self.device)
                else:
                    raise
            
            # Set to evaluation mode
            self.whisper_model.eval()
            
            self.logger.info(f"HF Whisper model '{model_id}' loaded successfully to {self.device}")
            
        except Exception as e:
            self.logger.error(f"Error loading HF Whisper model '{model_id}': {e}")
            raise RuntimeError(f"Failed to load HF Whisper model: {e}")
    
    def _load_standard_whisper_model(self, model_size: str) -> None:
        """
        Load a standard Whisper model.
        
        Args:
            model_size: Model size ("tiny", "base", "small", "medium", "large")
        """
        try:
            self.logger.info(f"Loading standard Whisper model: {model_size}")
            
            # Handle FP16 for CUDA
            fp16 = self.config.get("use_fp16", True) and self.device.type == "cuda"
            
            # Load model with enhanced logging
            self.whisper_model = log_model_download(
                f"OpenAI Whisper {model_size.upper()} Model",
                f"whisper-{model_size}",
                whisper.load_model,
                model_size,
                device=self.device.type,
                download_root=self.config.get("cache_dir"),
                in_memory=True
            )
            
            self.logger.info(f"Standard Whisper model '{model_size}' loaded successfully to {self.device}")
            
        except Exception as e:
            # Propagate to outer initialize logic for potential fallback handling
            self.logger.error(f"Error loading standard Whisper model '{model_size}': {e}")
            raise RuntimeError(e)
    
    def extract_audio_from_video(self, video_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        Extract audio from video file and preprocess it.
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
                        format="wav"
                    )
                    
                    if extracted_path and extracted_path.exists():
                        # Load the extracted audio file with librosa (safer than direct video loading)
                        audio, loaded_sr = librosa.load(
                            str(extracted_path),
                            sr=target_sr,
                            mono=True
                        )
                        
                        # Clean up is automatic with tempfile.TemporaryDirectory
                        self.logger.info(f"Audio extracted via FFmpeg: {len(audio)/target_sr:.2f} seconds at {target_sr}Hz")
                    else:
                        raise RuntimeError("FFmpeg extraction failed")
            else:
                self.logger.warning("FFmpeg not available, falling back to librosa direct loading")
                # Method 2: Fallback to direct librosa loading (may crash on some systems)
                audio, orig_sr = librosa.load(
                    str(video_path), 
                    sr=None,  # Use original sample rate initially
                    mono=True
                )
                
                # Resample if necessary
                if orig_sr != target_sr:
                    self.logger.info(f"Resampling audio from {orig_sr}Hz to {target_sr}Hz")
                    audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
            
            # Apply normalization if configured
            if self.config.get("normalize_audio", True):
                self.logger.debug("Normalizing audio")
                # Use safer normalization approach
                if np.max(np.abs(audio)) > 0:
                    audio = audio / np.max(np.abs(audio))
                
            self.logger.info(f"Audio extraction complete: {len(audio)/target_sr:.2f} seconds at {target_sr}Hz")
            return audio, target_sr
            
        except Exception as e:
            self.logger.error(f"Error extracting audio from {video_path}: {e}")
            # Try to provide more specific error information
            if "librosa" in str(e).lower() or "numba" in str(e).lower():
                raise RuntimeError(f"Audio processing library error. This may be due to librosa/numba compatibility issues. "
                                 f"Consider installing FFmpeg for more stable audio extraction. Original error: {e}")
            else:
                raise RuntimeError(f"Failed to extract audio: {e}")
    
    @torch.no_grad()
    def get_whisper_embedding(self, audio: np.ndarray, pad_or_trim: bool = True, 
                              target_seq_len: int = 1500) -> Optional[torch.Tensor]:
        """
        Extract Whisper embeddings from audio.
        
        Args:
            audio: Audio waveform as numpy array
            pad_or_trim: Whether to pad/trim embedding to standard sequence length
            target_seq_len: Target sequence length for embedding (default: 1500)
            
        Returns:
            Whisper embedding tensor
        """
        if not self.is_initialized:
            self.initialize()
            
        try:
            # Process differently based on model type
            if self.model_type == "huggingface":
                # Process through HF Whisper
                input_features = self.whisper_processor(
                    audio,
                    sampling_rate=self.config["sample_rate"],
                    return_tensors="pt"
                ).input_features.to(self.device)
                
                # Ensure input features match model dtype for FP16 compatibility
                if self.whisper_model.dtype != input_features.dtype:
                    input_features = input_features.to(dtype=self.whisper_model.dtype)
                
                # Get encoder outputs
                encoder_outputs = self.whisper_model.get_encoder()(input_features)
                embedding = encoder_outputs.last_hidden_state
                
                # Handle sequence length if needed
                if pad_or_trim:
                    return self._pad_or_trim_embedding(embedding, target_seq_len)
                return embedding
                
            else:  # Standard Whisper
                # Convert audio to float32 if needed
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)
                    
                # Use standard Whisper's encoder
                audio_tensor = torch.from_numpy(audio).to(self.device)
                embedding = self.whisper_model.embed_audio(audio_tensor)
                
                # For standard Whisper, return the embedding
                return embedding
                
        except Exception as e:
            self.logger.error(f"Error extracting Whisper embedding: {e}")
            return None
    
    def _pad_or_trim_embedding(self, embedding: torch.Tensor, target_seq_len: int = 1500) -> torch.Tensor:
        """
        Pad or trim embedding to the target sequence length.
        
        Args:
            embedding: Whisper embedding tensor
            target_seq_len: Target sequence length
            
        Returns:
            Padded or trimmed embedding tensor
        """
        current_seq_len = embedding.shape[1]
        embed_dim = embedding.shape[2]
        
        if current_seq_len < target_seq_len:
            # Pad with zeros using the same dtype and device as the embedding
            padding = torch.zeros(
                (1, target_seq_len - current_seq_len, embed_dim),
                device=embedding.device,
                dtype=embedding.dtype
            )
            return torch.cat((embedding, padding), dim=1)
        elif current_seq_len > target_seq_len:
            # Trim to target length
            return embedding[:, :target_seq_len, :]
        else:
            return embedding
    
    def segment_audio(self, audio: np.ndarray, sample_rate: int, pps: float = 0.2,
                     start_time: float = 0.0, end_time: Optional[float] = None,
                     min_segment_duration: float = 1.0, max_segment_duration: float = 30.0,
                     segment_overlap: float = 0.0) -> List[Dict[str, Any]]:
        """
        Segment audio based on a fixed interval strategy.
        
        Args:
            audio: Audio waveform as numpy array
            sample_rate: Sample rate of the audio
            pps: Predictions per second (used for fixed interval segmentation)
            start_time: Start time in seconds
            end_time: End time in seconds (None = end of audio)
            min_segment_duration: Minimum segment duration in seconds
            max_segment_duration: Maximum segment duration in seconds
            segment_overlap: Overlap between segments in seconds
            
        Returns:
            List of segment dictionaries with audio data and metadata
        """
        segments = []
        audio_duration = len(audio) / sample_rate
        
        if end_time is None or end_time > audio_duration:
            end_time = audio_duration
        
        self.logger.info(f"Segmenting audio using fixed interval mode")
        
        # Calculate segment duration based on pps
        if pps <= 0:
            # If pps is 0 or negative, use one segment for the entire audio
            segment_duration = end_time - start_time
        else:
            # Normal case: segment duration is 1/pps
            segment_duration = 1.0 / pps
        
        # Ensure segment duration is within bounds
        segment_duration = max(min_segment_duration, min(segment_duration, max_segment_duration))
        
        self.logger.info(f"Fixed interval segmentation: {segment_duration:.2f}s segments")
        
        # Create segments
        current_start = start_time
        while current_start < end_time:
            current_end = min(current_start + segment_duration, end_time)
            
            # Skip segments that are too short
            if current_end - current_start < min_segment_duration:
                break
            
            # Calculate audio indices
            start_idx = int(current_start * sample_rate)
            end_idx = int(current_end * sample_rate)
            
            if start_idx >= len(audio) or start_idx >= end_idx:
                break
            
            # Create segment
            segment = {
                "start_time": current_start,
                "end_time": current_end,
                "audio": audio[start_idx:end_idx],
                "speaker_id": None  # No speaker ID for fixed interval
            }
            
            segments.append(segment)
            
            # Move to next segment with potential overlap
            current_start += segment_duration - segment_overlap
        
        self.logger.info(f"Created {len(segments)} audio segments")
        return segments
    
    def cleanup(self) -> None:
        """Clean up resources used by the pipeline."""
        try:
            # Release CUDA memory
            if self.device and self.device.type == "cuda":
                if self.whisper_model is not None:
                    if hasattr(self.whisper_model, "to"):
                        self.whisper_model = self.whisper_model.to("cpu")
                    
                # Clear CUDA cache
                torch.cuda.empty_cache()
                gc.collect()
            
            # Set models to None to help garbage collection
            self.whisper_model = None
            self.whisper_processor = None
            
            self.logger.info("WhisperBasePipeline resources cleaned up")
            self.is_initialized = False
            
        except Exception as e:
            self.logger.error(f"Error cleaning up resources: {e}")

    def get_schema(self) -> Dict[str, Any]:
        """
        Return the output schema for this pipeline.
        
        Returns:
            Dictionary describing the output format
        """
        return {
            "type": "whisper_base",
            "description": "Base Whisper pipeline for audio processing",
            "output_format": {
                "embeddings": "torch.Tensor",
                "segments": "List[Dict[str, Any]]",
                "audio_info": "Dict[str, Any]"
            }
        }

    def process(self, video_path: str, start_time: float = 0.0, 
                end_time: Optional[float] = None, pps: float = 1.0,
                output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Process video/audio file to extract embeddings and segments.
        
        Args:
            video_path: Path to the video/audio file
            start_time: Start time in seconds
            end_time: End time in seconds (None = end of file)
            pps: Predictions per second for segmentation
            output_dir: Output directory (unused in base implementation)
            
        Returns:
            List of processing results
        """
        try:
            # Ensure initialization
            if not self.is_initialized:
                self.initialize()
            
            # Extract audio from video
            audio, sample_rate = self.extract_audio_from_video(video_path)
            
            # Segment the audio
            segments = self.segment_audio(
                audio=audio, 
                sample_rate=sample_rate,
                pps=pps,
                start_time=start_time,
                end_time=end_time
            )
            
            # Process each segment (basic implementation)
            results = []
            for i, segment in enumerate(segments):
                try:
                    # Extract embedding for this segment
                    embedding = self.get_whisper_embedding(segment["audio"])
                    
                    result = {
                        "segment_id": i,
                        "start_time": segment["start_time"],
                        "end_time": segment["end_time"],
                        "duration": segment["end_time"] - segment["start_time"],
                        "embedding_shape": list(embedding.shape) if embedding is not None else None,
                        "has_embedding": embedding is not None,
                        "audio_length": len(segment["audio"]),
                    }
                    results.append(result)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process segment {i}: {e}")
                    result = {
                        "segment_id": i,
                        "start_time": segment["start_time"],
                        "end_time": segment["end_time"],
                        "error": str(e),
                        "has_embedding": False
                    }
                    results.append(result)
            
            self.logger.info(f"Processed {len(results)} segments from {video_path}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing {video_path}: {e}")
            raise RuntimeError(f"Failed to process audio: {e}")
