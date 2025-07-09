"""
Face detection, emotion recognition, and gaze estimation schemas.
Includes support for OpenFace 3.0, DeepFace, and MediaPipe.
"""

from typing import Dict, Any, Optional, List, Tuple, ClassVar
from dataclasses import dataclass
from pydantic import BaseModel, Field

from .base_schema import AnnotationBase, BoundingBox, KeyPoint, BaseAnnotation, Point2D, Point3D, EmotionScores


class FaceDetection(BaseAnnotation):
    """Face detection annotation (modern Pydantic version)."""
    
    face_id: int = Field(..., description="Unique face identifier")
    person_id: Optional[int] = Field(None, description="Associated person ID if tracked")
    bbox: Dict[str, float] = Field(..., description="Face bounding box")
    landmarks_2d: Optional[Dict[str, Point2D]] = Field(None, description="2D facial landmarks")
    landmarks_3d: Optional[Dict[str, Point3D]] = Field(None, description="3D facial landmarks")
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Face quality/clarity score")
    blur_score: Optional[float] = Field(None, description="Face blur score")
    illumination_score: Optional[float] = Field(None, description="Face illumination quality")
    
    # OpenFace 3.0 specific features
    head_pose: Optional[Dict[str, float]] = Field(None, description="Head pose angles (pitch, yaw, roll)")
    face_alignment_confidence: Optional[float] = Field(None, description="Face alignment confidence")
    
    @property
    def is_frontal(self) -> bool:
        """Check if face is roughly frontal."""
        if self.head_pose:
            yaw = abs(self.head_pose.get("yaw", 0))
            pitch = abs(self.head_pose.get("pitch", 0))
            return yaw < 30 and pitch < 30
        return True


class FaceEmotion(BaseAnnotation):
    """Facial emotion recognition results."""
    
    face_id: int = Field(..., description="Associated face identifier")
    person_id: Optional[int] = Field(None, description="Associated person ID")
    emotions: EmotionScores = Field(..., description="Emotion scores")
    arousal: Optional[float] = Field(None, ge=0.0, le=1.0, description="Arousal level")
    valence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Valence level")
    
    # Demographics
    age_estimate: Optional[int] = Field(None, ge=0, le=150, description="Estimated age")
    gender_estimate: Optional[str] = Field(None, description="Estimated gender")
    gender_confidence: Optional[float] = Field(None, description="Gender estimation confidence")
    
    # Model information
    model_used: str = Field(default="unknown", description="Model used for recognition")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")


class FaceGaze(BaseAnnotation):
    """Gaze direction and attention estimation (OpenFace 3.0 enhanced)."""
    
    face_id: int = Field(..., description="Associated face identifier")
    person_id: Optional[int] = Field(None, description="Associated person ID")
    
    # Gaze direction
    gaze_direction_3d: Optional[Point3D] = Field(None, description="3D gaze direction vector")
    gaze_angles: Optional[Dict[str, float]] = Field(None, description="Gaze angles (pitch, yaw)")
    
    # Eye-specific information
    left_eye_gaze: Optional[Point3D] = Field(None, description="Left eye gaze vector")
    right_eye_gaze: Optional[Point3D] = Field(None, description="Right eye gaze vector")
    eye_landmarks: Optional[Dict[str, List[Point2D]]] = Field(None, description="Eye landmarks")
    
    # Attention and focus
    attention_target: Optional[str] = Field(None, description="Estimated attention target")
    focus_confidence: Optional[float] = Field(None, description="Focus estimation confidence")
    
    # Blink detection
    left_eye_open: Optional[bool] = Field(None, description="Left eye open state")
    right_eye_open: Optional[bool] = Field(None, description="Right eye open state")
    blink_detected: Optional[bool] = Field(None, description="Blink detected")
    
    @property
    def is_looking_at_camera(self) -> bool:
        """Estimate if person is looking at camera."""
        if self.gaze_angles:
            pitch = abs(self.gaze_angles.get("pitch", 0))
            yaw = abs(self.gaze_angles.get("yaw", 0))
            return pitch < 15 and yaw < 15
        return False


class FaceActionUnits(BaseAnnotation):
    """Facial Action Units (OpenFace 3.0 specific)."""
    
    face_id: int = Field(..., description="Associated face identifier")
    person_id: Optional[int] = Field(None, description="Associated person ID")
    
    # Action Units (AU) with intensity scores
    aus: Dict[str, float] = Field(..., description="Action Unit intensities")
    
    # Common Action Units
    AU_NAMES: ClassVar[Dict[str, str]] = {
        "AU01": "Inner Brow Raiser",
        "AU02": "Outer Brow Raiser", 
        "AU04": "Brow Lowerer",
        "AU05": "Upper Lid Raiser",
        "AU06": "Cheek Raiser",
        "AU07": "Lid Tightener",
        "AU09": "Nose Wrinkler",
        "AU10": "Upper Lip Raiser",
        "AU12": "Lip Corner Puller",
        "AU14": "Dimpler",
        "AU15": "Lip Corner Depressor",
        "AU17": "Chin Raiser",
        "AU20": "Lip Stretcher",
        "AU23": "Lip Tightener",
        "AU25": "Lips Part",
        "AU26": "Jaw Drop",
        "AU28": "Lip Suck",
        "AU45": "Blink"
    }
    
    @property
    def active_aus(self) -> List[str]:
        """Get list of active action units (intensity > 0.5)."""
        return [au for au, intensity in self.aus.items() if intensity > 0.5]
    
    @property
    def smile_intensity(self) -> float:
        """Calculate smile intensity from relevant AUs."""
        smile_aus = ["AU06", "AU12"]  # Cheek raiser and lip corner puller
        return max(self.aus.get(au, 0.0) for au in smile_aus)


class FaceIdentity(BaseAnnotation):
    """Face recognition and identity annotation."""
    
    face_id: int = Field(..., description="Associated face identifier")
    person_id: Optional[int] = Field(None, description="Associated person ID")
    
    # Identity features
    identity_embedding: Optional[List[float]] = Field(None, description="Face embedding vector")
    identity_confidence: Optional[float] = Field(None, description="Identity match confidence")
    known_identity: Optional[str] = Field(None, description="Known identity name")
    
    # Recognition model info
    model_used: str = Field(default="unknown", description="Face recognition model used")
    embedding_dimension: Optional[int] = Field(None, description="Embedding vector dimension")
    
    @property
    def is_known(self) -> bool:
        """Check if face has known identity."""
        return self.known_identity is not None
    
# Legacy dataclass support
@dataclass
class FaceDetectionLegacy(AnnotationBase):
    """A detected face in a frame (legacy dataclass support)."""
    face_id: int = 0  # Unique face identifier
    person_id: Optional[int] = None  # Associated person ID if tracked
    bbox: Optional[BoundingBox] = None  # Face bounding box
    landmarks: Optional[List[KeyPoint]] = None  # Facial landmarks
    quality_score: Optional[float] = None  # Face quality/clarity score
    
    def __post_init__(self):
        if not self.type:
            self.type = "face_detection"
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "face_id": self.face_id,
            "person_id": self.person_id,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "landmarks": [lm.to_dict() for lm in self.landmarks] if self.landmarks else None,
            "quality_score": self.quality_score
        })
        return base


@dataclass
class FaceEmotionLegacy(AnnotationBase):
    """Facial emotion recognition results (legacy dataclass support)."""
    face_id: int = 0
    person_id: Optional[int] = None
    emotions: Optional[Dict[str, float]] = None  # Emotion probabilities (anger, disgust, fear, happy, sad, surprise, neutral)
    dominant_emotion: str = "neutral"  # The emotion with highest probability
    arousal: Optional[float] = None  # Arousal level (0-1)
    valence: Optional[float] = None  # Valence level (0-1)
    age_estimate: Optional[int] = None  # Estimated age
    gender_estimate: Optional[str] = None  # Estimated gender
    
    def __post_init__(self):
        if not self.type:
            self.type = "face_emotion"
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "face_id": self.face_id,
            "person_id": self.person_id,
            "emotions": self.emotions or {},
            "dominant_emotion": self.dominant_emotion,
            "arousal": self.arousal,
            "valence": self.valence,
            "age_estimate": self.age_estimate,
            "gender_estimate": self.gender_estimate
        })
        return base


@dataclass
class FaceGazeLegacy(AnnotationBase):
    """Gaze direction and attention estimation (legacy dataclass support)."""
    face_id: int = 0
    person_id: Optional[int] = None
    gaze_direction: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # 3D gaze vector (x, y, z)
    gaze_angles: Tuple[float, float] = (0.0, 0.0)  # Pitch and yaw angles in degrees
    eye_landmarks: Optional[Dict[str, List[KeyPoint]]] = None  # Left/right eye landmarks
    attention_target: Optional[str] = None  # What the person is looking at
    
    def __post_init__(self):
        if not self.type:
            self.type = "face_gaze"
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "face_id": self.face_id,
            "person_id": self.person_id,
            "gaze_direction": list(self.gaze_direction),
            "gaze_angles": list(self.gaze_angles),
            "eye_landmarks": {
                eye: [lm.to_dict() for lm in landmarks] 
                for eye, landmarks in (self.eye_landmarks or {}).items()
            },
            "attention_target": self.attention_target
        })
        return base
