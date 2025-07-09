"""
Base data schemas for VideoAnnotator pipeline outputs.
Defines common structures and validation for all annotation types.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from dataclasses import dataclass


class BaseAnnotation(BaseModel):
    """Base class for all annotation types using Pydantic."""
    
    video_id: str = Field(..., description="Unique identifier for the video")
    timestamp: float = Field(..., description="Timestamp in seconds")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    
    class Config:
        extra = "forbid"
        validate_assignment = True


@dataclass
class AnnotationBase:
    """Base class for all annotations (legacy dataclass support)."""
    type: str
    video_id: str
    timestamp: float  # Time in seconds
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert annotation to dictionary format."""
        return {
            "type": self.type,
            "video_id": self.video_id,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "metadata": self.metadata or {}
        }


@dataclass  
class VideoMetadata:
    """Video file metadata."""
    video_id: str
    filepath: str
    width: int
    height: int
    fps: float
    duration: float
    total_frames: int
    codec: Optional[str] = None
    created_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary format."""
        return {
            "video_id": self.video_id,
            "filepath": self.filepath,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "duration": self.duration,
            "total_frames": self.total_frames,
            "codec": self.codec,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None
        }


@dataclass
class BoundingBox:
    """Standard bounding box representation."""
    x: float  # Top-left x coordinate (normalized 0-1)
    y: float  # Top-left y coordinate (normalized 0-1) 
    width: float  # Width (normalized 0-1)
    height: float  # Height (normalized 0-1)
    
    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}
    
    @property
    def x2(self) -> float:
        """Right edge of bounding box."""
        return self.x + self.width
    
    @property
    def y2(self) -> float:
        """Bottom edge of bounding box."""
        return self.y + self.height
    
    @property
    def center(self) -> tuple[float, float]:
        """Center point of bounding box."""
        return (self.x + self.width / 2, self.y + self.height / 2)
    
    @property
    def area(self) -> float:
        """Area of bounding box."""
        return self.width * self.height


class Point2D(BaseModel):
    """2D point with confidence."""
    
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Point confidence")


class Point3D(BaseModel):
    """3D point with confidence."""
    
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    z: float = Field(..., description="Z coordinate")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Point confidence")


class EmotionScores(BaseModel):
    """Emotion classification scores."""
    
    anger: float = Field(default=0.0, ge=0.0, le=1.0)
    disgust: float = Field(default=0.0, ge=0.0, le=1.0)
    fear: float = Field(default=0.0, ge=0.0, le=1.0)
    happiness: float = Field(default=0.0, ge=0.0, le=1.0)
    sadness: float = Field(default=0.0, ge=0.0, le=1.0)
    surprise: float = Field(default=0.0, ge=0.0, le=1.0)
    neutral: float = Field(default=0.0, ge=0.0, le=1.0)
    
    @property
    def dominant_emotion(self) -> str:
        """Get the emotion with highest score."""
        emotions = {
            'anger': self.anger,
            'disgust': self.disgust,
            'fear': self.fear,
            'happiness': self.happiness,
            'sadness': self.sadness,
            'surprise': self.surprise,
            'neutral': self.neutral
        }
        return max(emotions, key=emotions.get)


@dataclass
class KeyPoint:
    """Standard keypoint representation."""
    x: float  # X coordinate (normalized 0-1)
    y: float  # Y coordinate (normalized 0-1)
    confidence: float  # Confidence score (0-1)
    visible: bool = True  # Whether the keypoint is visible
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y, 
            "confidence": self.confidence,
            "visible": self.visible
        }


class BasePipeline(ABC):
    """Abstract base class for all annotation pipelines."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_initialized = False
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the pipeline (load models, etc.)."""
        pass
    
    @abstractmethod
    def process(self, video_path: str) -> List[BaseAnnotation]:
        """Process a video and return annotations."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources."""
        pass
    
    def __enter__(self):
        """Context manager entry."""
        if not self.is_initialized:
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
