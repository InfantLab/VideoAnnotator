"""
Scene detection and classification schemas.
"""

from typing import Dict, Any, Optional, List, ClassVar
from dataclasses import dataclass
from pydantic import BaseModel, Field

from .base_schema import AnnotationBase, BaseAnnotation


class SceneAnnotation(BaseAnnotation):
    """Scene detection annotation (modern Pydantic version)."""
    
    scene_id: str = Field(..., description="Unique scene identifier")
    start_time: float = Field(..., description="Scene start time in seconds")
    end_time: float = Field(..., description="Scene end time in seconds")
    change_type: str = Field(default="cut", description="Type of scene change (cut, fade, etc.)")
    
    @property
    def duration(self) -> float:
        """Duration of the scene in seconds."""
        return self.end_time - self.start_time


class SceneClassification(BaseAnnotation):
    """Scene classification annotation."""
    
    scene_id: str = Field(..., description="Associated scene identifier")
    label: str = Field(..., description="Scene classification label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    categories: Dict[str, float] = Field(default_factory=dict, description="All category scores")
    
    # Common scene categories
    INDOOR_CATEGORIES: ClassVar[List[str]] = [
        "living_room", "bedroom", "kitchen", "bathroom", "office", 
        "clinic", "nursery", "classroom", "hospital", "restaurant"
    ]
    
    OUTDOOR_CATEGORIES: ClassVar[List[str]] = [
        "park", "garden", "street", "beach", "playground", "forest", 
        "yard", "balcony", "patio", "field"
    ]
    
    ACTIVITY_CATEGORIES: ClassVar[List[str]] = [
        "play", "eating", "sleeping", "reading", "exercise", 
        "conversation", "television", "music", "learning"
    ]


class AudioContext(BaseAnnotation):
    """Audio context for scene understanding."""
    
    scene_id: str = Field(..., description="Associated scene identifier")
    audio_tags: List[str] = Field(default_factory=list, description="Audio classification tags")
    speech_detected: bool = Field(default=False, description="Whether speech is detected")
    music_detected: bool = Field(default=False, description="Whether music is detected")
    noise_level: float = Field(default=0.0, ge=0.0, le=1.0, description="Background noise level")
    dominant_frequency: Optional[float] = Field(None, description="Dominant frequency in Hz")
    
    # Common audio tags
    AUDIO_TAGS = [
        "speech", "laughter", "crying", "music", "television", 
        "toys", "appliances", "outdoor", "traffic", "nature"
    ]


@dataclass
class SceneSegment(AnnotationBase):
    """A detected scene segment (legacy dataclass support)."""
    start_time: float  # Segment start time in seconds
    end_time: float    # Segment end time in seconds
    scene_id: str      # Unique scene identifier
    scene_type: Optional[str] = None  # Scene classification (e.g., "living_room", "outdoor")
    transition_type: Optional[str] = None  # Type of transition ("cut", "fade", etc.)
    
    def __post_init__(self):
        if not self.type:
            self.type = "scene_segment"
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "start_time": self.start_time,
            "end_time": self.end_time,
            "scene_id": self.scene_id,
            "scene_type": self.scene_type,
            "transition_type": self.transition_type
        })
        return base


@dataclass
class SceneAnnotation(AnnotationBase):
    """Complete scene analysis for a video segment."""
    segments: List[SceneSegment]
    total_scenes: int
    avg_scene_length: float
    scene_types: List[str]  # All detected scene types
    
    def __post_init__(self):
        if not self.type:
            self.type = "scene_analysis"
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "segments": [seg.to_dict() for seg in self.segments],
            "total_scenes": self.total_scenes,
            "avg_scene_length": self.avg_scene_length,
            "scene_types": self.scene_types
        })
        return base
