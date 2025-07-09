"""
Audio processing schemas for speech, music, and sound analysis.
"""

from typing import Dict, Any, Optional, List, Tuple, ClassVar
from dataclasses import dataclass
from pydantic import BaseModel, Field

from .base_schema import AnnotationBase, BaseAnnotation


class AudioSegment(BaseAnnotation):
    """Audio segment annotation (modern Pydantic version)."""
    
    start_time: float = Field(..., description="Segment start time in seconds")
    end_time: float = Field(..., description="Segment end time in seconds")
    audio_type: str = Field(..., description="Type of audio (speech, music, noise, silence)")
    volume_level: float = Field(default=0.0, ge=0.0, le=1.0, description="Volume level")
    
    @property
    def duration(self) -> float:
        """Duration of the audio segment."""
        return self.end_time - self.start_time


class SpeechRecognition(BaseAnnotation):
    """Speech recognition and transcription."""
    
    speaker_id: Optional[int] = Field(None, description="Speaker identifier")
    transcript: str = Field(..., description="Transcribed text")
    language: str = Field(default="en", description="Detected language")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Transcription confidence")
    
    # Word-level timing
    words: Optional[List[Dict[str, Any]]] = Field(None, description="Word-level timing and confidence")
    
    # Speech characteristics
    speech_rate: Optional[float] = Field(None, description="Speech rate (words per minute)")
    pitch_mean: Optional[float] = Field(None, description="Mean pitch in Hz")
    pitch_std: Optional[float] = Field(None, description="Pitch standard deviation")
    
    # Emotion from speech
    speech_emotion: Optional[str] = Field(None, description="Emotion detected from speech")
    speech_emotion_confidence: Optional[float] = Field(None, description="Speech emotion confidence")


class SpeakerDiarizationModern(BaseAnnotation):
    """Speaker diarization results (modern Pydantic version)."""
    
    speaker_id: int = Field(..., description="Speaker identifier")
    start_time: float = Field(..., description="Speaker segment start time")
    end_time: float = Field(..., description="Speaker segment end time")
    speaker_label: Optional[str] = Field(None, description="Speaker label/name")
    
    # Speaker characteristics
    gender: Optional[str] = Field(None, description="Estimated gender")
    age_estimate: Optional[int] = Field(None, description="Estimated age")
    voice_characteristics: Optional[Dict[str, float]] = Field(None, description="Voice features")
    
    @property
    def duration(self) -> float:
        """Duration of speaker segment."""
        return self.end_time - self.start_time


class AudioClassification(BaseAnnotation):
    """Audio event classification."""
    
    event_type: str = Field(..., description="Type of audio event")
    event_confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Event confidence")
    
    # Common audio events
    AUDIO_EVENTS: ClassVar[List[str]] = [
        "speech", "laughter", "crying", "applause", "music",
        "toys", "television", "phone", "door", "footsteps",
        "coughing", "sneezing", "breathing", "silence"
    ]
    
    # Audio features
    spectral_features: Optional[Dict[str, float]] = Field(None, description="Spectral features")
    mfcc_features: Optional[List[float]] = Field(None, description="MFCC features")
    
    @property
    def is_human_vocalization(self) -> bool:
        """Check if event is human vocalization."""
        human_events = ["speech", "laughter", "crying", "coughing", "sneezing"]
        return self.event_type in human_events


# Legacy dataclass support
@dataclass
class SpeechSegment(AnnotationBase):
    """A speech segment with transcription (legacy dataclass support)."""
    start_time: float = 0.0  # Segment start time in seconds
    end_time: float = 0.0    # Segment end time in seconds
    text: str = ""          # Transcribed text
    language: Optional[str] = None  # Detected language
    speaker_id: Optional[str] = None  # Speaker identifier
    words: Optional[List[Dict[str, Any]]] = None  # Word-level timestamps
    
    def __post_init__(self):
        if not self.type:
            self.type = "speech_segment"
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text": self.text,
            "language": self.language,
            "speaker_id": self.speaker_id,
            "words": self.words
        })
        return base


@dataclass
class SpeakerDiarization(AnnotationBase):
    """Speaker identification and diarization results (legacy dataclass support)."""
    speakers: List[str] = None  # List of detected speaker IDs
    segments: List[Dict[str, Any]] = None  # Speaker segments with start/end times
    total_speech_time: float = 0.0  # Total time with speech
    speaker_change_points: List[float] = None  # Timestamps of speaker changes
    
    def __post_init__(self):
        if self.speakers is None:
            self.speakers = []
        if self.segments is None:
            self.segments = []
        if self.speaker_change_points is None:
            self.speaker_change_points = []
        if not self.type:
            self.type = "speaker_diarization"
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "speakers": self.speakers,
            "segments": self.segments,
            "total_speech_time": self.total_speech_time,
            "speaker_change_points": self.speaker_change_points
        })
        return base


@dataclass
class AudioFeatures(AnnotationBase):
    """Audio feature extraction results (legacy dataclass support)."""
    duration: float = 0.0
    sample_rate: int = 44100
    fundamental_frequency: Optional[List[float]] = None  # F0 over time
    spectral_features: Optional[Dict[str, List[float]]] = None  # MFCC, spectral centroid, etc.
    prosodic_features: Optional[Dict[str, float]] = None  # Pitch range, speaking rate, etc.
    voice_activity: Optional[List[Tuple[float, float]]] = None  # Voice activity detection segments
    laughter_segments: Optional[List[Dict[str, Any]]] = None  # Detected laughter
    
    def __post_init__(self):
        if not self.type:
            self.type = "audio_features"
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "fundamental_frequency": self.fundamental_frequency,
            "spectral_features": self.spectral_features,
            "prosodic_features": self.prosodic_features,
            "voice_activity": [list(segment) for segment in (self.voice_activity or [])],
            "laughter_segments": self.laughter_segments
        })
        return base
