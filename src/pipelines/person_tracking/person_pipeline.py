"""
Standards-only person tracking pipeline.

This pipeline works directly with COCO person/keypoint format annotations,
eliminating all custom schema dependencies.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from ..base_pipeline import BasePipeline
from ...exporters.native_formats import (
    create_coco_annotation,
    create_coco_keypoints_annotation,
    create_coco_image_entry,
    export_coco_json,
    validate_coco_json,
)

# Optional imports
try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class PersonTrackingPipeline(BasePipeline):
    """
    Standards-only person tracking pipeline using COCO person/keypoint format.

    Returns native COCO annotation dictionaries instead of custom schemas.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "model": "yolo11n-pose.pt",  # YOLO11 pose model
            "conf_threshold": 0.4,
            "iou_threshold": 0.7,
            "track_mode": True,  # Enable tracking
            "tracker": "bytetrack",  # or "botsort"
            "pose_format": "coco_17",  # COCO 17 keypoints
            "min_keypoint_confidence": 0.3,
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)

        self.logger = logging.getLogger(__name__)
        self.model = None

    def process(
        self,
        video_path: str,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        pps: float = 5,  # 5 predictions per second for person tracking
        output_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process video for person detection and tracking.

        Returns:
            List of COCO format annotation dictionaries with person detection and pose results.
        """

        # Get video metadata
        video_metadata = self._get_video_metadata(video_path)

        # Ensure pipeline is initialized
        if not self.is_initialized:
            self.initialize()

        # Process video
        annotations = []
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame processing parameters
        if end_time is None:
            end_time = total_frames / fps

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        frame_step = max(1, int(fps / pps))

        annotation_id = 1

        try:
            for frame_num in range(start_frame, min(end_frame, total_frames), frame_step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()

                if not ret:
                    continue

                timestamp = frame_num / fps

                # Detect and track persons
                frame_annotations = self._process_frame(
                    frame, timestamp, video_metadata["video_id"], frame_num
                )

                # Assign unique annotation IDs
                for ann in frame_annotations:
                    ann["id"] = annotation_id
                    annotation_id += 1

                annotations.extend(frame_annotations)

        finally:
            cap.release()

        # Save results if output directory specified
        if output_dir and annotations:
            self._save_coco_annotations(annotations, output_dir, video_metadata)

        self.logger.info(f"Person tracking complete: {len(annotations)} detections")
        return annotations

    def _get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract video metadata."""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        cap.release()

        return {
            "video_id": Path(video_path).stem,
            "filepath": video_path,
            "width": width,
            "height": height,
            "fps": fps,
            "duration": duration,
            "total_frames": total_frames,
        }

    def _initialize_model(self):
        """Initialize YOLO11 pose model."""
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics package required for person tracking")

        try:
            self.model = YOLO(self.config["model"])
            self.logger.info(f"Initialized YOLO model: {self.config['model']}")
        except Exception as e:
            self.logger.error(f"Failed to initialize YOLO model: {e}")
            raise

    def _process_frame(
        self, frame: np.ndarray, timestamp: float, video_id: str, frame_number: int
    ) -> List[Dict[str, Any]]:
        """Process a single frame for person detection and pose estimation."""

        height, width = frame.shape[:2]

        # Run YOLO inference
        if self.config["track_mode"]:
            results = self.model.track(
                frame,
                conf=self.config["conf_threshold"],
                iou=self.config["iou_threshold"],
                tracker=f"{self.config['tracker']}.yaml",
                persist=True,
            )
        else:
            results = self.model(
                frame, conf=self.config["conf_threshold"], iou=self.config["iou_threshold"]
            )

        annotations = []

        if results and len(results) > 0:
            result = results[0]

            # Process each detection
            if result.boxes is not None:
                boxes = result.boxes.cpu().numpy()
                
                # Handle keypoints properly
                keypoints_data = None
                if result.keypoints is not None:
                    # Extract the actual keypoint data from the Keypoints object
                    keypoints_data = result.keypoints.data.cpu().numpy()  # Shape: (N, 17, 3)

                for i, box in enumerate(boxes):
                    # Filter for person class (class 0 in COCO)
                    if int(box.cls[0]) == 0:  # Person class
                        # Get bounding box in COCO format [x, y, width, height]
                        x1, y1, x2, y2 = box.xyxy[0]
                        bbox_coco = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

                        # Get tracking ID if available
                        track_id = int(box.id[0]) if box.id is not None else i

                        # Check if keypoints are available for this detection
                        if keypoints_data is not None and i < len(keypoints_data):
                            # Create keypoints annotation
                            kp_data = keypoints_data[i]  # Shape: (17, 3) for COCO-17

                            # Convert to COCO keypoints format: [x1,y1,v1,x2,y2,v2,...]
                            coco_keypoints = []
                            visible_keypoints = 0

                            for kp in kp_data:
                                x, y, conf = float(kp[0]), float(kp[1]), float(kp[2])
                                
                                # Visibility: 0=not labeled, 1=labeled but not visible, 2=labeled and visible
                                visibility = (
                                    2 if conf > self.config["min_keypoint_confidence"] else 0
                                )
                                if visibility > 0:
                                    visible_keypoints += 1
                                coco_keypoints.extend([x, y, visibility])

                            # Create COCO keypoints annotation
                            annotation = create_coco_keypoints_annotation(
                                annotation_id=0,  # Will be set later
                                image_id=f"{video_id}_frame_{frame_number}",
                                category_id=1,  # Person category
                                keypoints=coco_keypoints,
                                bbox=bbox_coco,
                                num_keypoints=visible_keypoints,
                                score=float(box.conf[0]),
                                # VideoAnnotator extensions
                                track_id=track_id,
                                timestamp=timestamp,
                                frame_number=frame_number,
                            )
                        else:
                            # Create basic bounding box annotation
                            annotation = create_coco_annotation(
                                annotation_id=0,  # Will be set later
                                image_id=f"{video_id}_frame_{frame_number}",
                                category_id=1,  # Person category
                                bbox=bbox_coco,
                                score=float(box.conf[0]),
                                # VideoAnnotator extensions
                                track_id=track_id,
                                timestamp=timestamp,
                                frame_number=frame_number,
                            )

                        annotations.append(annotation)

        return annotations

    def _save_coco_annotations(
        self, annotations: List[Dict[str, Any]], output_dir: str, video_metadata: Dict[str, Any]
    ):
        """Save annotations in COCO format."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create COCO images list
        images = []
        image_ids = set()

        for ann in annotations:
            image_id = ann["image_id"]
            if image_id not in image_ids:
                image_ids.add(image_id)
                frame_number = ann.get("frame_number", 0)
                timestamp = ann.get("timestamp", 0.0)

                image = create_coco_image_entry(
                    image_id=image_id,
                    width=video_metadata["width"],
                    height=video_metadata["height"],
                    file_name=f"frame_{frame_number:06d}.jpg",
                    # VideoAnnotator extensions
                    video_id=video_metadata["video_id"],
                    frame_number=frame_number,
                    timestamp=timestamp,
                )
                images.append(image)

        # Export COCO JSON with keypoints
        categories = [
            {
                "id": 1,
                "name": "person",
                "supercategory": "person",
                "keypoints": [
                    "nose",
                    "left_eye",
                    "right_eye",
                    "left_ear",
                    "right_ear",
                    "left_shoulder",
                    "right_shoulder",
                    "left_elbow",
                    "right_elbow",
                    "left_wrist",
                    "right_wrist",
                    "left_hip",
                    "right_hip",
                    "left_knee",
                    "right_knee",
                    "left_ankle",
                    "right_ankle",
                ],
                "skeleton": [
                    [16, 14],
                    [14, 12],
                    [17, 15],
                    [15, 13],
                    [12, 13],
                    [6, 12],
                    [7, 13],
                    [6, 7],
                    [6, 8],
                    [7, 9],
                    [8, 10],
                    [9, 11],
                    [2, 3],
                    [1, 2],
                    [1, 3],
                    [2, 4],
                    [3, 5],
                    [4, 6],
                    [5, 7],
                ],
            }
        ]

        coco_path = output_path / f"{video_metadata['video_id']}_person_tracking.json"
        export_coco_json(annotations, images, str(coco_path), categories)

        # Validate COCO format
        validation_result = validate_coco_json(str(coco_path), "person_tracking")
        if validation_result.is_valid:
            self.logger.info(f"Person tracking COCO validation successful: {coco_path}")
        else:
            self.logger.warning(
                f"Person tracking COCO validation warnings: {', '.join(validation_result.warnings)}"
            )

    def cleanup(self):
        """Cleanup resources."""
        if self.model is not None:
            del self.model
            self.model = None

    def initialize(self) -> None:
        """Initialize the person tracking pipeline by loading the YOLO model."""
        if self.is_initialized:
            return

        self.logger.info("Initializing PersonTrackingPipeline...")

        try:
            if not YOLO_AVAILABLE:
                raise ImportError("Ultralytics YOLO not available. Install with: pip install ultralytics")

            self._initialize_model()
            self.set_model_info("yolo", self.config["model"])
            self.is_initialized = True
            self.logger.info("PersonTrackingPipeline initialization complete")

        except Exception as e:
            self.logger.error(f"Failed to initialize PersonTrackingPipeline: {e}")
            raise

    def get_schema(self) -> Dict[str, Any]:
        """Return the COCO schema for person tracking outputs."""
        return {
            "type": "array",
            "description": "Person tracking results in COCO annotation format",
            "items": {
                "type": "object",
                "description": "COCO annotation for person detection/tracking",
                "properties": {
                    "id": {"type": "integer", "description": "Unique annotation ID"},
                    "image_id": {"type": "integer", "description": "Image/frame ID"},
                    "category_id": {"type": "integer", "description": "COCO category ID (1 for person)"},
                    "bbox": {
                        "type": "array",
                        "description": "Bounding box [x, y, width, height]",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4
                    },
                    "area": {"type": "number", "description": "Bounding box area"},
                    "iscrowd": {"type": "integer", "description": "0 for individual objects"},
                    "keypoints": {
                        "type": "array",
                        "description": "COCO-17 keypoints [x1,y1,v1, x2,y2,v2, ...]",
                        "items": {"type": "number"}
                    },
                    "num_keypoints": {"type": "integer", "description": "Number of visible keypoints"},
                    "score": {"type": "number", "description": "Detection confidence"},
                    "track_id": {"type": ["integer", "null"], "description": "Tracking ID across frames"}
                },
                "required": ["id", "image_id", "category_id", "bbox", "area", "iscrowd"]
            }
        }
