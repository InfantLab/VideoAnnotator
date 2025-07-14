"""
Whisper Base Pipeline for VideoAnnotator

This module provides a base pipeline for Whisper-based audio processing tasks.
It serves as a foundation for both speech recognition and voice emotion analysis.
"""

import logging
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import torch
import librosa

from ..base_pipeline import BasePipeline

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
        
    def initialize(self) -> None:
        """Initialize the Whisper model and resources."""
        if self.is_initialized:
            return
            
        # Setup device
        if self.config["device"] == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config["device"])
            
        self.logger.info(f"Initializing WhisperBasePipeline with device: {self.device}")
            
        # Load Whisper model
        self._load_whisper_model()
        
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
                
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
                model_id, **model_kwargs
            ).to(self.device)
            
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
            
            # Load model
            self.whisper_model = whisper.load_model(
                model_size, 
                device=self.device.type,
                download_root=self.config.get("cache_dir"),
                in_memory=True,
            )
            
            self.logger.info(f"Standard Whisper model '{model_size}' loaded successfully to {self.device}")
            
        except Exception as e:
            self.logger.error(f"Error loading standard Whisper model '{model_size}': {e}")
            raise RuntimeError(f"Failed to load standard Whisper model: {e}")
    
    def extract_audio_from_video(self, video_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        Extract audio from video file and preprocess it.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple of (audio_waveform, sample_rate)
        """
        try:
            video_path = str(video_path)
            self.logger.info(f"Extracting audio from {video_path}")
            
            # Use librosa to load audio
            audio, orig_sr = librosa.load(
                video_path, 
                sr=None,  # Use original sample rate initially
                mono=True
            )
            
            target_sr = self.config["sample_rate"]
            
            # Resample if necessary
            if orig_sr != target_sr:
                self.logger.info(f"Resampling audio from {orig_sr}Hz to {target_sr}Hz")
                audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
            
            # Apply normalization if configured
            if self.config.get("normalize_audio", True):
                self.logger.debug("Normalizing audio")
                audio = librosa.util.normalize(audio)
            
            self.logger.info(f"Audio extracted: {len(audio)/target_sr:.2f} seconds at {target_sr}Hz")
            return audio, target_sr
        
        except Exception as e:
            self.logger.error(f"Error extracting audio from {video_path}: {e}")
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
