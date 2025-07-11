"""
Speaker Diarization Pipeline for VideoAnnotator

This pipeline focuses specifically on speaker diarization using PyAnnote.
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
    from pyannote.audio import Pipeline as PyannoteePipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    logging.warning("pyannote.audio not available. Speaker diarization will be disabled.")

from ..base_pipeline import BasePipeline
from ...schemas.audio_schema import SpeakerDiarization


@dataclass
class DiarizationPipelineConfig:
    """Configuration for speaker diarization pipeline."""
    
    # Authentication
    huggingface_token: Optional[str] = None  # For PyAnnote models
    
    # Speaker diarization
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    min_speakers: int = 1
    max_speakers: int = 10
    
    # Processing options
    use_gpu: bool = True  # Use GPU if available
    
    def __post_init__(self):
        """Set token from environment if not provided."""
        if not self.huggingface_token:
            # Try HF_AUTH_TOKEN first (standard), then HUGGINGFACE_TOKEN (legacy)
            self.huggingface_token = os.getenv("HF_AUTH_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")


class DiarizationPipeline(BasePipeline):
    """
    Speaker diarization pipeline using PyAnnote.
    """
    
    def __init__(self, config: Optional[DiarizationPipelineConfig] = None):
        super().__init__()
        self.config = config or DiarizationPipelineConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self._diarization_pipeline = None
        
    def initialize(self) -> None:
        """Initialize the diarization pipeline."""
        if not PYANNOTE_AVAILABLE:
            raise ImportError("pyannote.audio is not available. Please install it with: pip install pyannote.audio")
        
        if not self.config.huggingface_token:
            raise ValueError(
                "HuggingFace token is required for speaker diarization. "
                "Set HUGGINGFACE_TOKEN environment variable or pass it in config."
            )
        
        try:
            self.logger.info(f"Loading diarization model: {self.config.diarization_model}")
            self._diarization_pipeline = PyannoteePipeline.from_pretrained(
                self.config.diarization_model,
                use_auth_token=self.config.huggingface_token
            )
            
            # Send to GPU if available and requested
            if self.config.use_gpu and torch.cuda.is_available():
                self._diarization_pipeline.to(torch.device("cuda"))
                self.logger.info("Diarization pipeline moved to GPU")
            elif self.config.use_gpu:
                self.logger.warning("GPU requested but CUDA not available, using CPU")
            
            self.is_initialized = True
            self.logger.info("Diarization pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing diarization pipeline: {e}")
            raise
    
    def process(
        self, 
        video_path: str, 
        start_time: float = 0.0, 
        end_time: Optional[float] = None,
        pps: float = 0.0,
        output_dir: Optional[str] = None
    ) -> List[SpeakerDiarization]:
        """
        Process video and return speaker diarization results.
        
        Args:
            video_path: Path to video file
            start_time: Segment start time in seconds (not used for diarization)
            end_time: Segment end time in seconds (not used for diarization)
            pps: Predictions per second (not used for diarization)
            output_dir: Optional output directory
            
        Returns:
            List containing SpeakerDiarization result
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            # First extract audio from video if needed
            video_path = Path(video_path)
            audio_path = video_path.with_suffix('.wav')
            
            # Extract audio if it doesn't exist
            if not audio_path.exists():
                from .ffmpeg_utils import extract_audio_from_video
                
                if output_dir:
                    output_path = Path(output_dir) / f"{video_path.stem}_audio.wav"
                else:
                    output_path = video_path.parent / f"{video_path.stem}_audio.wav"
                
                extracted_audio = extract_audio_from_video(
                    video_path=video_path,
                    output_path=output_path,
                    sample_rate=16000,  # PyAnnote works well with 16kHz
                    channels=1  # Mono audio for diarization
                )
                
                if extracted_audio:
                    audio_path = Path(extracted_audio)
                else:
                    raise RuntimeError(f"Failed to extract audio from {video_path}")
            
            # Perform diarization
            result = self.diarize_audio(audio_path)
            return [result] if result else []
            
        except Exception as e:
            self.logger.error(f"Error processing diarization: {e}")
            return []
    
    def diarize_audio(self, audio_path: Union[str, Path]) -> Optional[SpeakerDiarization]:
        """
        Perform speaker diarization on an audio file.
        
        Args:
            audio_path: Path to audio file (preferably WAV format)
            
        Returns:
            SpeakerDiarization object or None if failed
        """
        if not self.is_initialized:
            self.initialize()
        
        if not self._diarization_pipeline:
            self.logger.error("Diarization pipeline not initialized")
            return None
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            self.logger.error(f"Audio file not found: {audio_path}")
            return None
        
        try:
            self.logger.info(f"Performing speaker diarization on: {audio_path}")
            diarization = self._diarization_pipeline(str(audio_path))
            
            speakers = {}
            segments = []
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_id = f"speaker_{speaker}"
                if speaker_id not in speakers:
                    speakers[speaker_id] = {
                        'speaker_id': speaker_id,
                        'total_duration': 0.0,
                        'segments': []
                    }
                
                segment = {
                    'start_time': float(turn.start),
                    'end_time': float(turn.end),
                    'speaker_id': speaker_id,
                    'confidence': 1.0  # pyannote doesn't provide confidence scores
                }
                
                speakers[speaker_id]['segments'].append(segment)
                speakers[speaker_id]['total_duration'] += turn.end - turn.start
                segments.append(segment)
            
            # Create result using the legacy dataclass format
            result = SpeakerDiarization(
                type="speaker_diarization",
                video_id=str(audio_path.stem),  # Use filename as video_id
                timestamp=0.0,  # Start time
                speakers=list(speakers.keys()),
                segments=segments,
                total_speech_time=sum(s['total_duration'] for s in speakers.values()),
                speaker_change_points=[seg['start_time'] for seg in segments[1:]]
            )
            
            # Add metadata
            result.metadata = {
                'num_speakers': len(speakers),
                'audio_file': str(audio_path),
                'model': self.config.diarization_model
            }
            
            self.logger.info(f"Diarization completed. Found {len(speakers)} speakers in {len(segments)} segments")
            return result
            
        except Exception as e:
            self.logger.error(f"Speaker diarization failed: {e}")
            return None
    
    def cleanup(self) -> None:
        """Clean up pipeline resources."""
        if self._diarization_pipeline:
            # Move to CPU to free GPU memory
            if hasattr(self._diarization_pipeline, 'to'):
                import torch
                self._diarization_pipeline.to(torch.device('cpu'))
        
        self._diarization_pipeline = None
        self.is_initialized = False
        self.logger.info("Diarization pipeline cleaned up")
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the output schema for this pipeline."""
        return {
            "type": "speaker_diarization",
            "description": "Speaker diarization results",
            "properties": {
                "speakers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of detected speaker IDs"
                },
                "segments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "speaker_id": {"type": "string"},
                            "start_time": {"type": "number"},
                            "end_time": {"type": "number"},
                            "confidence": {"type": "number"}
                        }
                    },
                    "description": "Speaker segments with timing"
                },
                "total_speech_time": {
                    "type": "number",
                    "description": "Total time with speech detected"
                },
                "speaker_change_points": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Timestamps where speakers change"
                }
            }
        }
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the diarization pipeline."""
        return {
            'name': 'DiarizationPipeline',
            'version': '1.0.0',
            'capabilities': {
                'speaker_diarization': PYANNOTE_AVAILABLE,
            },
            'models': {
                'diarization_model': self.config.diarization_model if PYANNOTE_AVAILABLE else None,
            },
            'config': {
                'diarization_model': self.config.diarization_model,
                'min_speakers': self.config.min_speakers,
                'max_speakers': self.config.max_speakers,
                'use_gpu': self.config.use_gpu,
                'has_token': bool(self.config.huggingface_token)
            },
            'requirements': {
                'pyannote_available': PYANNOTE_AVAILABLE,
                'cuda_available': torch.cuda.is_available(),
                'has_auth_token': bool(self.config.huggingface_token)
            }
        }
