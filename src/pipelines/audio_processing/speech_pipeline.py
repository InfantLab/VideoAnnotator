"""
Speech Recognition Pipeline for VideoAnnotator

This pipeline focuses specifically on speech recognition using OpenAI Whisper.
It's designed to be separable from other audio processing functionality.
"""

import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import numpy as np
import torch
import librosa
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logging.warning("whisper not available. Speech recognition will be disabled.")

from .whisper_base_pipeline import WhisperBasePipeline
from version import __version__


class SpeechPipeline(WhisperBasePipeline):
    """
    Speech recognition pipeline using OpenAI Whisper.
    
    This implementation extends WhisperBasePipeline to leverage the shared
    Whisper functionality while focusing specifically on transcription.
    """

    @property
    def output_format(self) -> str:
        """Return the output format for this pipeline."""
        return "webvtt"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Speech-specific config defaults
        speech_config = {
            "whisper_model": "base",  # Will use this as model ID for WhisperBasePipeline
            "language": None,
            "task": "transcribe",
            "beam_size": 5,
            "best_of": 5,
            "temperature": 0.0,
            "patience": 1.0,
            "length_penalty": 1.0,
            "word_timestamps": True,
            "prepend_punctuations": "\"'\"¿([{-",
            "append_punctuations": "\"'.。,，!！?？:：\")]}、",
            "suppress_tokens": None,
            "min_segment_duration": 1.0
        }
        
        # Update with user-provided config
        if config:
            speech_config.update(config)
        
        # Convert legacy "model" key to "whisper_model" if present
        if config and "model" in config and "whisper_model" not in config:
            speech_config["whisper_model"] = config["model"]
        
        # Initialize base pipeline with merged config
        super().__init__(speech_config)
        self.logger = logging.getLogger(__name__)

    def process(
        self,
        video_path: str,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        pps: float = 0.0,
        output_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process video and return speech recognition results.

        Args:
            video_path: Path to video file
            start_time: Segment start time in seconds
            end_time: Segment end time in seconds
            pps: Predictions per second
            output_dir: Optional output directory

        Returns:
            List containing SpeechRecognition result
        """
        if not self.is_initialized:
            self.initialize()

        try:
            # Convert path to Path object
            video_path = Path(video_path)
            
            # Extract audio using base pipeline functionality
            audio, sample_rate = self.extract_audio_from_video(video_path)
            
            # Transcribe the audio
            result = self.transcribe_audio(audio)
            return [result] if result else []

        except Exception as e:
            self.logger.error(f"Error processing speech recognition: {e}")
            return []

    def transcribe_audio(self, audio: Union[str, Path, np.ndarray]) -> Optional[Dict[str, Any]]:
        """
        Perform speech recognition on audio.

        Args:
            audio: Path to audio file or audio waveform

        Returns:
            SpeechRecognition object or None if failed
        """
        if not self.is_initialized:
            self.initialize()

        if not self.whisper_model:
            self.logger.error("Whisper model not initialized")
            return None

        try:
            self.logger.info("Performing speech recognition")
            
            # Handle different audio input types
            if isinstance(audio, (str, Path)):
                audio_path = Path(audio)
                if not audio_path.exists():
                    self.logger.error(f"Audio file not found: {audio_path}")
                    return None
                
                # For standard Whisper, use the built-in transcribe method
                if self.model_type == "standard":
                    # Prepare Whisper options
                    options = {
                        "language": self.config.get("language"),
                        "task": self.config.get("task", "transcribe"),
                        "beam_size": self.config.get("beam_size", 5),
                        "best_of": self.config.get("best_of", 5),
                        "temperature": self.config.get("temperature", 0.0),
                        "patience": self.config.get("patience", 1.0),
                        "length_penalty": self.config.get("length_penalty", 1.0),
                        "word_timestamps": self.config.get("word_timestamps", True),
                        "prepend_punctuations": self.config.get("prepend_punctuations", "\"'\"¿([{-"),
                        "append_punctuations": self.config.get("append_punctuations", "\"'.。,，!！?？:：\")]}、"),
                        "fp16": self.config.get("use_fp16", True) and torch.cuda.is_available(),
                    }

                    if self.config.get("suppress_tokens"):
                        options["suppress_tokens"] = self.config["suppress_tokens"]

                    # Transcribe audio
                    result = self.whisper_model.transcribe(str(audio_path), **options)
                    return self._process_transcription_result(result, audio_path.stem)
                
                # For HuggingFace Whisper, load the audio and process
                else:
                    # Use the base pipeline's audio loading
                    audio, sr = librosa.load(
                        audio_path, 
                        sr=self.config["sample_rate"],
                        mono=True
                    )
            
            # Handle numpy array input (already loaded audio)
            if isinstance(audio, np.ndarray):
                if self.model_type == "standard":
                    # For standard Whisper with waveform
                    options = {
                        "language": self.config.get("language"),
                        "task": self.config.get("task", "transcribe"),
                        "beam_size": self.config.get("beam_size", 5),
                        "best_of": self.config.get("best_of", 5),
                        "temperature": self.config.get("temperature", 0.0),
                        "patience": self.config.get("patience", 1.0),
                        "length_penalty": self.config.get("length_penalty", 1.0),
                        "word_timestamps": self.config.get("word_timestamps", True),
                        "prepend_punctuations": self.config.get("prepend_punctuations", "\"'\"¿([{-"),
                        "append_punctuations": self.config.get("append_punctuations", "\"'.。,，!！?？:：\")]}、"),
                        "fp16": self.config.get("use_fp16", True) and torch.cuda.is_available(),
                    }
                    
                    if self.config.get("suppress_tokens"):
                        options["suppress_tokens"] = self.config["suppress_tokens"]
                    
                    # Transcribe audio waveform
                    result = self.whisper_model.transcribe(audio, **options)
                    return self._process_transcription_result(result, "audio_segment")
                
                # For HuggingFace Whisper, process directly
                else:
                    # TODO: Implement HuggingFace Whisper transcription
                    # This requires using the tokenizer and generation utilities
                    self.logger.error("HuggingFace Whisper transcription not yet implemented")
                    return None

        except Exception as e:
            self.logger.error(f"Speech recognition failed: {e}")
            return None

    def _process_transcription_result(self, result: Dict[str, Any], video_id: str) -> Dict[str, Any]:
        """
        Process Whisper transcription result into standard format.
        
        Args:
            result: Raw Whisper result
            video_id: Identifier for the video/audio
            
        Returns:
            Processed transcription result
        """
        # Extract word-level timestamps
        word_timestamps = []
        if result.get("segments"):
            for segment in result["segments"]:
                if "words" in segment:
                    for word_info in segment["words"]:
                        word_timestamps.append(
                            {
                                "word": word_info["word"].strip(),
                                "start": float(word_info["start"]),
                                "end": float(word_info["end"]),
                                "confidence": float(word_info.get("probability", 0.0)),
                            }
                        )

        # Extract segment-level information
        segments = []
        if result.get("segments"):
            for segment in result["segments"]:
                segments.append(
                    {
                        "id": segment["id"],
                        "start": float(segment["start"]),
                        "end": float(segment["end"]),
                        "text": segment["text"].strip(),
                        "tokens": segment.get("tokens", []),
                        "temperature": float(segment.get("temperature", 0.0)),
                        "avg_logprob": float(segment.get("avg_logprob", 0.0)),
                        "compression_ratio": float(segment.get("compression_ratio", 0.0)),
                        "no_speech_prob": float(segment.get("no_speech_prob", 0.0)),
                    }
                )

        # Create result using dictionary format
        speech_result = {
            "video_id": video_id,
            "timestamp": 0.0,  # Start time
            "transcript": result["text"].strip(),
            "language": result.get("language", "unknown"),
            "confidence": 0.0,  # Whisper doesn't provide overall confidence
            "words": word_timestamps,
            "metadata": {
                "model": self.config["whisper_model"],
                "task": self.config.get("task", "transcribe"),
                "num_segments": len(segments),
                "num_words": len(word_timestamps),
                "start_time": 0.0,
                "end_time": segments[-1]["end"] if segments else 0.0,
                "total_duration": segments[-1]["end"] if segments else 0.0,
                "segments": segments,  # Store segment details in metadata
            }
        }

        self.logger.info(
            f"Speech recognition completed. "
            f"Transcribed {len(segments)} segments with {len(word_timestamps)} words"
        )
        return speech_result

    def get_schema(self) -> Dict[str, Any]:
        """Get the output schema for this pipeline."""
        return {
            "type": "speech_recognition",
            "description": "Speech recognition results with word and segment timestamps",
            "properties": {
                "transcript": {"type": "string", "description": "Full transcribed text"},
                "language": {"type": "string", "description": "Detected or specified language"},
                "confidence": {"type": "number", "description": "Overall confidence score"},
                "words": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "word": {"type": "string"},
                            "start": {"type": "number"},
                            "end": {"type": "number"},
                            "confidence": {"type": "number"},
                        },
                    },
                    "description": "Word-level timestamps with confidence",
                },
            },
        }

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the speech recognition pipeline."""
        return {
            "name": "SpeechPipeline",
            "version": __version__,
            "capabilities": {
                "speech_recognition": WHISPER_AVAILABLE,
                "word_timestamps": True,
                "segment_timestamps": True,
                "language_detection": True,
                "translation": True,
            },
            "models": {
                "whisper_model": self.config["whisper_model"] if WHISPER_AVAILABLE else None,
            },
            "config": {
                "model_name": self.config["whisper_model"],
                "language": self.config.get("language"),
                "task": self.config.get("task", "transcribe"),
                "word_timestamps": self.config.get("word_timestamps", True),
                "device": self.config.get("device", "auto"),
                "use_fp16": self.config.get("use_fp16", True),
            },
            "requirements": {
                "whisper_available": WHISPER_AVAILABLE,
                "cuda_available": torch.cuda.is_available(),
            },
        }
