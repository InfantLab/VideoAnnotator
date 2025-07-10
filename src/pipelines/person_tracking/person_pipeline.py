"""
Person detection and tracking pipeline using YOLO11.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from pathlib import Path
from datetime import datetime
import json

from ..base_pipeline import BasePipeline
from ...schemas.person_schema import PersonDetection, PersonTrajectory, PoseKeypoints
from ...schemas.base_schema import BoundingBox, KeyPoint


class PersonTrackingPipeline(BasePipeline):
    """
    Person detection, pose estimation, and tracking using YOLO11.
    
    Supports unified detection, pose estimation, and tracking in one model.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "model": "yolo11n-pose.pt",  # YOLO11 pose model
            "conf_threshold": 0.4,
            "iou_threshold": 0.7,
            "track_mode": True,  # Enable tracking
            "tracker": "bytetrack",  # or "botsort"
            "pose_format": "coco_17",  # COCO 17 keypoints
            "min_keypoint_confidence": 0.3
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
    
    def process(
        self, 
        video_path: str, 
        start_time: float = 0.0, 
        end_time: Optional[float] = None,
        pps: float = 30.0,  # Default to 30 FPS for tracking
        output_dir: Optional[str] = None
    ) -> List[PersonDetection]:
        """Process video for person detection and tracking."""
        
        try:
            from ultralytics import YOLO
            import cv2
        except ImportError:
            raise ImportError("Ultralytics not installed. Run: pip install ultralytics")
        
        # Load YOLO11 model
        model = YOLO(self.config["model"])
        
        # Get video metadata
        video_metadata = self.get_video_metadata(video_path)
        
        # Process video
        detections = []
        tracks = {}  # Track ID -> list of detections
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame range
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps) if end_time else video_metadata.total_frames
        
        frame_skip = max(1, int(fps / pps)) if pps > 0 else 1
        
        try:
            frame_number = start_frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            while frame_number < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = frame_number / fps
                
                # Run YOLO11 with tracking
                if self.config["track_mode"]:
                    results = model.track(
                        frame,
                        conf=self.config["conf_threshold"],
                        iou=self.config["iou_threshold"],
                        tracker=f"{self.config['tracker']}.yaml",
                        persist=True
                    )
                else:
                    results = model(
                        frame,
                        conf=self.config["conf_threshold"],
                        iou=self.config["iou_threshold"]
                    )
                
                # Process results
                if results and len(results) > 0:
                    result = results[0]
                    
                    # Extract detections
                    if result.boxes is not None:
                        for i, box in enumerate(result.boxes):
                            # Get person ID (from tracking or sequential)
                            person_id = int(box.id.item()) if box.id is not None else i
                            
                            # Only process person class (class 0 in COCO)
                            if int(box.cls.item()) == 0:  # Person class
                                # Get bounding box (normalized)
                                bbox_xyxy = box.xyxyn[0].cpu().numpy()
                                bbox_dict = {
                                    "x": float(bbox_xyxy[0]),
                                    "y": float(bbox_xyxy[1]), 
                                    "width": float(bbox_xyxy[2] - bbox_xyxy[0]),
                                    "height": float(bbox_xyxy[3] - bbox_xyxy[1])
                                }
                                
                                # Get pose keypoints if available
                                pose_keypoints = None
                                # Note: Disabling pose keypoints temporarily for testing
                                # if result.keypoints is not None and i < len(result.keypoints.xyn):
                                #     pose_keypoints = self._extract_keypoints(
                                #         result.keypoints.xyn[i], 
                                #         result.keypoints.conf[i]
                                #     )
                                
                                # Create detection
                                detection = PersonDetection(
                                    video_id=video_metadata.video_id,
                                    timestamp=timestamp,
                                    person_id=person_id,
                                    bbox=bbox_dict,
                                    detection_confidence=float(box.conf.item()),
                                    metadata={
                                        "frame": frame_number,
                                        "model": self.config["model"]
                                    }
                                )
                                
                                detections.append(detection)
                                
                                # Track for trajectory building
                                if person_id not in tracks:
                                    tracks[person_id] = []
                                tracks[person_id].append(detection)
                
                # Skip frames based on PPS
                frame_number += frame_skip
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        finally:
            cap.release()
        
        # Save if output directory specified
        if output_dir:
            # Save individual detections
            detection_path = Path(output_dir) / f"{video_metadata.video_id}_person_detections.json"
            self.save_annotations(detections, str(detection_path))
            
            # Save tracking trajectories
            if self.config["track_mode"]:
                trajectories = self._build_trajectories(tracks, video_metadata.video_id)
                trajectory_path = Path(output_dir) / f"{video_metadata.video_id}_person_tracks.json"
                self.save_annotations(trajectories, str(trajectory_path))
        
        return detections
    
    def _extract_keypoints(self, keypoints_xyn, confidences) -> List[KeyPoint]:
        """Extract COCO keypoints from YOLO11 output."""
        # COCO 17 keypoint names
        coco_keypoints = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
        pose_keypoints = []
        for i, (kp, conf) in enumerate(zip(keypoints_xyn, confidences)):
            if i < len(coco_keypoints) and conf > self.config["min_keypoint_confidence"]:
                keypoint = KeyPoint(
                    x=float(kp[0]),
                    y=float(kp[1]),
                    confidence=float(conf),
                    visible=conf > self.config["min_keypoint_confidence"]
                )
                pose_keypoints.append(keypoint)
        
        return pose_keypoints
    
    def _build_trajectories(self, tracks: Dict[int, List[PersonDetection]], video_id: str) -> List[PersonTrajectory]:
        """Build trajectory objects from tracked detections."""
        trajectories = []
        
        for person_id, detections in tracks.items():
            if len(detections) > 1:  # Only include multi-frame tracks
                detections.sort(key=lambda d: d.timestamp)
                
                trajectory = PersonTrajectory(
                    video_id=video_id,
                    timestamp=detections[0].timestamp,
                    person_id=person_id,
                    detections=detections,
                    first_seen=detections[0].timestamp,
                    last_seen=detections[-1].timestamp
                )
                trajectories.append(trajectory)
        
        return trajectories
    
    def get_schema(self) -> Dict[str, Any]:
        """Return JSON schema for person detection annotations."""
        return {
            "type": "object",
            "properties": {
                "type": {"type": "string", "const": "person_detection"},
                "video_id": {"type": "string"},
                "timestamp": {"type": "number"},
                "person_id": {"type": "integer"},
                "bbox": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "width": {"type": "number"},
                        "height": {"type": "number"}
                    },
                    "required": ["x", "y", "width", "height"]
                },
                "pose_keypoints": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "confidence": {"type": "number"},
                            "visible": {"type": "boolean"}
                        }
                    }
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "metadata": {"type": "object"}
            },
            "required": ["type", "video_id", "timestamp", "person_id", "bbox"]
        }

    def initialize(self) -> None:
        """Initialize the pipeline (load models, etc.)."""
        self.logger.info("Initializing Person Tracking Pipeline")
        
        # Verify YOLO is available
        try:
            from ultralytics import YOLO
            self.logger.info(f"YOLO (Ultralytics) available")
            
            # Try to load the model to verify it works
            model_name = self.config.get("model", "yolo11n-pose.pt")
            self.logger.info(f"Using model: {model_name}")
            
            # Note: We don't actually load the model here to save memory
            # It will be loaded when needed in the process method
            
        except ImportError:
            self.logger.warning("YOLO (Ultralytics) not available - person tracking will be disabled")
        
        # Verify OpenCV availability
        try:
            import cv2
            self.logger.info(f"OpenCV version: {cv2.__version__}")
        except ImportError:
            self.logger.warning("OpenCV not available - video processing will be limited")
        
        self.is_initialized = True
        self.logger.info("Person Tracking Pipeline initialized successfully")
        
        # Set model information (will be updated when model is actually loaded)
        model_name = self.config.get("model", "yolo11n-pose.pt")
        self.set_model_info(model_name, None)

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.logger.info("Cleaning up Person Tracking Pipeline")
        # Nothing specific to clean up for this pipeline
        self.is_initialized = False

    def get_video_metadata(self, video_path: str):
        """Get video metadata - delegate to base class method."""
        from ...schemas.base_schema import VideoMetadata
        import cv2
        
        video_info = self.get_video_info(video_path)
        path = Path(video_path)
        
        return VideoMetadata(
            video_id=path.stem,
            filepath=str(path),
            duration=video_info["duration"],
            fps=video_info["fps"],
            width=video_info["width"],
            height=video_info["height"],
            total_frames=video_info["frame_count"]
        )

    def save_annotations(self, detections: List[PersonDetection], output_path: str, video_metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save person detections to JSON file with comprehensive metadata."""
        
        # Enhance video metadata with detection statistics
        if video_metadata is None:
            video_metadata = {}
            
        detection_stats = {
            "total_detections": len(detections),
            "unique_persons": len(set(d.person_id for d in detections)),
            "time_range": {
                "start": min(d.timestamp for d in detections) if detections else 0.0,
                "end": max(d.timestamp for d in detections) if detections else 0.0
            }
        }
        video_metadata.update(detection_stats)
        
        output_data = {
            "metadata": self.create_output_metadata(video_metadata),
            "pipeline": "person_tracking",
            "timestamp": datetime.now().isoformat(),
            "detections": [d.model_dump(mode='json') if hasattr(d, 'model_dump') else d.dict() for d in detections]
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
