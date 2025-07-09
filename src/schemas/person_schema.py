"""
Person detection, tracking, and pose estimation schemas.
"""

from typing import Dict, Any, Optional, List, ClassVar
from dataclasses import dataclass
from pydantic import BaseModel, Field

from .base_schema import AnnotationBase, BoundingBox, KeyPoint, BaseAnnotation, Point2D


class PersonDetection(BaseAnnotation):
    """Person detection annotation (modern Pydantic version)."""
    
    person_id: int = Field(..., description="Unique tracking identifier")
    bbox: Dict[str, float] = Field(..., description="Bounding box coordinates")
    tracking_confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Tracking confidence")
    detection_confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Detection confidence")
    
    # Optional attributes
    age_estimate: Optional[float] = Field(None, description="Estimated age")
    gender_estimate: Optional[str] = Field(None, description="Estimated gender")
    clothing_colors: Optional[List[str]] = Field(None, description="Dominant clothing colors")
    
    @property
    def center_point(self) -> tuple[float, float]:
        """Get center point of bounding box."""
        return (
            self.bbox["x"] + self.bbox["width"] / 2,
            self.bbox["y"] + self.bbox["height"] / 2
        )


class PoseKeypoints(BaseAnnotation):
    """YOLO11 pose keypoints annotation."""
    
    person_id: int = Field(..., description="Associated person identifier")
    keypoints: Dict[str, Point2D] = Field(..., description="Named keypoints")
    pose_type: str = Field(default="coco_17", description="Keypoint format")
    action_label: Optional[str] = Field(None, description="Detected action/gesture")
    
    # Standard COCO-17 keypoint names
    COCO_17_KEYPOINTS: ClassVar[List[str]] = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    @property
    def is_standing(self) -> bool:
        """Estimate if person is standing based on keypoints."""
        if "left_ankle" in self.keypoints and "right_ankle" in self.keypoints:
            ankle_y = (self.keypoints["left_ankle"].y + self.keypoints["right_ankle"].y) / 2
            if "nose" in self.keypoints:
                return self.keypoints["nose"].y < ankle_y - 0.3
        return False
    
    @property
    def pose_confidence(self) -> float:
        """Average confidence of all visible keypoints."""
        if not self.keypoints:
            return 0.0
        return sum(kp.confidence for kp in self.keypoints.values()) / len(self.keypoints)


class PersonTrajectory(BaseAnnotation):
    """Person tracking trajectory across frames."""
    
    person_id: int = Field(..., description="Unique tracking identifier")
    detections: List[PersonDetection] = Field(..., description="Detection history")
    first_seen: float = Field(..., description="First appearance timestamp")
    last_seen: float = Field(..., description="Last appearance timestamp")
    track_status: str = Field(default="active", description="Tracking status")
    
    @property
    def duration(self) -> float:
        """Total tracking duration."""
        return self.last_seen - self.first_seen
    
    @property
    def is_continuous(self) -> bool:
        """Check if tracking is continuous (no gaps)."""
        if len(self.detections) < 2:
            return True
        
        for i in range(1, len(self.detections)):
            gap = self.detections[i].timestamp - self.detections[i-1].timestamp
            if gap > 1.0:  # More than 1 second gap
                return False
        return True
    
    @property
    def movement_distance(self) -> float:
        """Calculate total movement distance."""
        if len(self.detections) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(self.detections)):
            prev_center = self.detections[i-1].center_point
            curr_center = self.detections[i].center_point
            
            distance = ((curr_center[0] - prev_center[0])**2 + 
                       (curr_center[1] - prev_center[1])**2)**0.5
            total_distance += distance
        
        return total_distance



# Legacy dataclass support
@dataclass
class PersonDetectionLegacy(AnnotationBase):
    """A detected person in a frame (legacy dataclass support)."""
    person_id: int = 0  # Unique identifier for tracking
    bbox: Optional[BoundingBox] = None  # Person bounding box
    pose_keypoints: Optional[List[KeyPoint]] = None  # COCO-style keypoints if available
    attributes: Optional[Dict[str, Any]] = None  # Additional attributes (age, clothing, etc.)
    
    def __post_init__(self):
        if not self.type:
            self.type = "person_detection"
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "person_id": self.person_id,
            "bbox": self.bbox.to_dict(),
            "pose_keypoints": [kp.to_dict() for kp in self.pose_keypoints] if self.pose_keypoints else None,
            "attributes": self.attributes
        })
        return base


@dataclass
class PersonTrackingLegacy(AnnotationBase):
    """Person tracking across multiple frames (legacy dataclass support)."""
    person_id: int = 0
    trajectory: Optional[List[PersonDetectionLegacy]] = None  # Detections across frames
    first_seen: float = 0.0  # First appearance timestamp
    last_seen: float = 0.0   # Last appearance timestamp
    total_duration: float = 0.0  # Total time person is visible
    
    def __post_init__(self):
        if not self.type:
            self.type = "person_tracking"
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "person_id": self.person_id,
            "trajectory": [det.to_dict() for det in self.trajectory] if self.trajectory else [],
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "total_duration": self.total_duration
        })
        return base


@dataclass
class PoseKeypointsLegacy(AnnotationBase):
    """Detailed pose keypoint annotation (legacy dataclass support)."""
    person_id: int = 0
    keypoints: Optional[Dict[str, KeyPoint]] = None  # Named keypoints (nose, left_eye, etc.)
    pose_type: str = "coco_17"  # Keypoint format (coco_17, coco_133, etc.)
    action_label: Optional[str] = None  # Detected action/gesture
    
    def __post_init__(self):
        if not self.type:
            self.type = "pose_keypoints"
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "person_id": self.person_id,
            "keypoints": {name: kp.to_dict() for name, kp in self.keypoints.items()} if self.keypoints else {},
            "pose_type": self.pose_type,
            "action_label": self.action_label
        })
        return base
