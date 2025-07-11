"""
Speech Recognition Pipeline for VideoAnnotator

This pipeline focuses specifically on speech recognition using OpenAI Whisper.
It's designed to be separable from other audio processing functionality.
"""

import logging
import os
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import torch
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logging.warning("whisper not available. Speech recognition will be disabled.")

from ..base_pipeline import BasePipeline
from ...schemas.audio_schema import SpeechRecognition


@dataclass
class SpeechPipelineConfig:
    """Configuration for speech recognition pipeline."""
    
    # Whisper model configuration
    model_name: str = "base"  # tiny, base, small, medium, large, large-v2, large-v3
    language: Optional[str] = None  # Auto-detect if None
    task: str = "transcribe"  # transcribe or translate
    
    # Processing options
    beam_size: int = 5
    best_of: int = 5
    temperature: float = 0.0
    patience: float = 1.0
    length_penalty: float = 1.0
    suppress_tokens: Optional[List[int]] = None
    
    # Word-level timestamps
    word_timestamps: bool = True
    prepend_punctuations: str = "\"'¿([{-"
    append_punctuations: str = "\"'.。,，!！?？:：\")]}、"
    
    # Audio preprocessing
    use_vad: bool = True  # Voice Activity Detection
    vad_threshold: float = 0.5
    chunk_length: int = 30  # seconds
    
    # Processing options
    use_gpu: bool = True  # Use GPU if available
    fp16: bool = True  # Use half precision
    
    def __post_init__(self):
        """Validate configuration."""
        valid_models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        if self.model_name not in valid_models:
            raise ValueError(f"Invalid model_name: {self.model_name}. Must be one of {valid_models}")
        
        if self.task not in ["transcribe", "translate"]:
            raise ValueError(f"Invalid task: {self.task}. Must be 'transcribe' or 'translate'")


class SpeechPipeline(BasePipeline):
    """
    Speech recognition pipeline using OpenAI Whisper.
    """
    
    def __init__(self, config: Optional[SpeechPipelineConfig] = None):
        super().__init__()
        self.config = config or SpeechPipelineConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self._whisper_model = None
        
    def initialize(self) -> None:
        """Initialize the speech recognition pipeline."""
        if not WHISPER_AVAILABLE:
            raise ImportError("whisper is not available. Please install it with: pip install openai-whisper")
        
        try:
            self.logger.info(f"Loading Whisper model: {self.config.model_name}")
            
            # Load model with device selection
            device = "cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu"
            self._whisper_model = whisper.load_model(
                self.config.model_name, 
                device=device
            )
            
            if device == "cuda":
                self.logger.info("Whisper model loaded on GPU")
            else:
                if self.config.use_gpu:
                    self.logger.warning("GPU requested but CUDA not available, using CPU")
                else:
                    self.logger.info("Whisper model loaded on CPU")
            
            self.is_initialized = True
            self.logger.info("Speech recognition pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing speech recognition pipeline: {e}")
            raise
    
    def process(
        self, 
        video_path: str, 
        start_time: float = 0.0, 
        end_time: Optional[float] = None,
        pps: float = 0.0,
        output_dir: Optional[str] = None
    ) -> List[SpeechRecognition]:
        """
        Process video and return speech recognition results.
        
        Args:
            video_path: Path to video file
            start_time: Segment start time in seconds (not used for speech recognition)
            end_time: Segment end time in seconds (not used for speech recognition)
            pps: Predictions per second (not used for speech recognition)
            output_dir: Optional output directory
            
        Returns:
            List containing SpeechRecognition result
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            # First extract audio from video if needed
            video_path = Path(video_path)
            
            # Use FFmpeg to extract audio
            from .ffmpeg_utils import extract_audio_from_video
            
            if output_dir:
                output_path = Path(output_dir) / f"{video_path.stem}_audio.wav"
            else:
                output_path = video_path.parent / f"{video_path.stem}_audio.wav"
            
            # Extract audio if it doesn't exist
            if not output_path.exists():
                extracted_audio = extract_audio_from_video(
                    video_path=video_path,
                    output_path=output_path,
                    sample_rate=16000,  # Whisper's native sample rate
                    channels=1  # Mono audio for speech recognition
                )
                
                if not extracted_audio:
                    raise RuntimeError(f"Failed to extract audio from {video_path}")
                
                audio_path = Path(extracted_audio)
            else:
                audio_path = output_path
            
            # Perform speech recognition
            result = self.transcribe_audio(audio_path)
            return [result] if result else []
            
        except Exception as e:
            self.logger.error(f"Error processing speech recognition: {e}")
            return []
    
    def transcribe_audio(self, audio_path: Union[str, Path]) -> Optional[SpeechRecognition]:
        """
        Perform speech recognition on an audio file.
        
        Args:
            audio_path: Path to audio file (preferably WAV format)
            
        Returns:
            SpeechRecognition object or None if failed
        """
        if not self.is_initialized:
            self.initialize()
        
        if not self._whisper_model:
            self.logger.error("Whisper model not initialized")
            return None
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            self.logger.error(f"Audio file not found: {audio_path}")
            return None
        
        try:
            self.logger.info(f"Performing speech recognition on: {audio_path}")
            
            # Prepare Whisper options
            options = {
                "language": self.config.language,
                "task": self.config.task,
                "beam_size": self.config.beam_size,
                "best_of": self.config.best_of,
                "temperature": self.config.temperature,
                "patience": self.config.patience,
                "length_penalty": self.config.length_penalty,
                "word_timestamps": self.config.word_timestamps,
                "prepend_punctuations": self.config.prepend_punctuations,
                "append_punctuations": self.config.append_punctuations,
                "fp16": self.config.fp16 and torch.cuda.is_available(),
            }
            
            if self.config.suppress_tokens:
                options["suppress_tokens"] = self.config.suppress_tokens
            
            # Transcribe audio
            result = self._whisper_model.transcribe(str(audio_path), **options)
            
            # Extract word-level timestamps
            word_timestamps = []
            if result.get('segments'):
                for segment in result['segments']:
                    if 'words' in segment:
                        for word_info in segment['words']:
                            word_timestamps.append({
                                'word': word_info['word'].strip(),
                                'start': float(word_info['start']),
                                'end': float(word_info['end']),
                                'confidence': float(word_info.get('probability', 0.0))
                            })
            
            # Extract segment-level information
            segments = []
            if result.get('segments'):
                for segment in result['segments']:
                    segments.append({
                        'id': segment['id'],
                        'start': float(segment['start']),
                        'end': float(segment['end']),
                        'text': segment['text'].strip(),
                        'tokens': segment.get('tokens', []),
                        'temperature': float(segment.get('temperature', 0.0)),
                        'avg_logprob': float(segment.get('avg_logprob', 0.0)),
                        'compression_ratio': float(segment.get('compression_ratio', 0.0)),
                        'no_speech_prob': float(segment.get('no_speech_prob', 0.0))
                    })
            
            # Create result using the schema format
            speech_result = SpeechRecognition(
                video_id=str(audio_path.stem),  # Use filename as video_id
                timestamp=0.0,  # Start time
                transcript=result['text'].strip(),
                language=result.get('language', 'unknown'),
                confidence=0.0,  # Whisper doesn't provide overall confidence
                words=word_timestamps
            )
            
            # Add metadata
            speech_result.metadata = {
                'model': self.config.model_name,
                'audio_file': str(audio_path),
                'task': self.config.task,
                'num_segments': len(segments),
                'num_words': len(word_timestamps),
                'start_time': 0.0,
                'end_time': segments[-1]['end'] if segments else 0.0,
                'total_duration': segments[-1]['end'] if segments else 0.0,
                'segments': segments  # Store segment details in metadata
            }
            
            self.logger.info(f"Speech recognition completed. "
                           f"Transcribed {len(segments)} segments with {len(word_timestamps)} words")
            return speech_result
            
        except Exception as e:
            self.logger.error(f"Speech recognition failed: {e}")
            return None
    
    def cleanup(self) -> None:
        """Clean up pipeline resources."""
        if self._whisper_model:
            # Move to CPU to free GPU memory
            if hasattr(self._whisper_model, 'to'):
                self._whisper_model = self._whisper_model.to('cpu')
        
        self._whisper_model = None
        self.is_initialized = False
        self.logger.info("Speech recognition pipeline cleaned up")
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the output schema for this pipeline."""
        return {
            "type": "speech_recognition",
            "description": "Speech recognition results with word and segment timestamps",
            "properties": {
                "transcript": {
                    "type": "string",
                    "description": "Full transcribed text"
                },
                "language": {
                    "type": "string",
                    "description": "Detected or specified language"
                },
                "confidence": {
                    "type": "number",
                    "description": "Overall confidence score"
                },
                "words": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "word": {"type": "string"},
                            "start": {"type": "number"},
                            "end": {"type": "number"},
                            "confidence": {"type": "number"}
                        }
                    },
                    "description": "Word-level timestamps with confidence"
                }
            }
        }
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the speech recognition pipeline."""
        return {
            'name': 'SpeechPipeline',
            'version': '1.0.0',
            'capabilities': {
                'speech_recognition': WHISPER_AVAILABLE,
                'word_timestamps': True,
                'segment_timestamps': True,
                'language_detection': True,
                'translation': True
            },
            'models': {
                'whisper_model': self.config.model_name if WHISPER_AVAILABLE else None,
            },
            'config': {
                'model_name': self.config.model_name,
                'language': self.config.language,
                'task': self.config.task,
                'word_timestamps': self.config.word_timestamps,
                'use_gpu': self.config.use_gpu,
                'fp16': self.config.fp16
            },
            'requirements': {
                'whisper_available': WHISPER_AVAILABLE,
                'cuda_available': torch.cuda.is_available()
            }
        }
