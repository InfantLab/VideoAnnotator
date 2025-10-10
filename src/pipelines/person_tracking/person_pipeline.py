"""
Standards-only person tracking pipeline.

This pipeline works directly with COCO person/keypoint format annotations,
eliminating all custom schema dependencies.
"""

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from exporters.native_formats import (
    create_coco_annotation,
    create_coco_image_entry,
    create_coco_keypoints_annotation,
    export_coco_json,
    validate_coco_json,
)
from pipelines.base_pipeline import BasePipeline
from utils.automatic_labeling import infer_person_labels_from_tracks
from utils.model_loader import log_model_download
from utils.person_identity import PersonIdentityManager
from utils.size_based_person_analysis import run_size_based_analysis

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

    def __init__(self, config: dict[str, Any] | None = None):
        default_config = {
            "model": "models/yolo/yolo11n-pose.pt",  # YOLO11 pose model
            "conf_threshold": 0.4,
            "iou_threshold": 0.7,
            "track_mode": True,  # Enable tracking
            "tracker": "bytetrack",  # or "botsort"
            "pose_format": "coco_17",  # COCO 17 keypoints
            "min_keypoint_confidence": 0.3,
            # Size-based analysis (integrated simple analyzer)
            "size_analysis": {
                "enabled": True,
                "height_threshold": 0.4,
                "confidence": 0.7,
                "adult_label": "parent",
                "child_label": "infant",
            },
            # Person identity configuration
            "person_identity": {
                "enabled": True,
                "id_format": "semantic",  # "semantic" or "integer"
                "automatic_labeling": {
                    "enabled": True,
                    "confidence_threshold": 0.7,
                    "size_based": {
                        "enabled": True,
                        "height_threshold": 0.4,  # Normalized height threshold for child/adult
                        "confidence": 0.7,
                        "adult_label": "parent",
                        "child_label": "infant",
                    },
                    "position_based": {
                        "enabled": True,
                        "center_bias_threshold": 0.7,  # Threshold for center positioning
                        "confidence": 0.6,
                        "primary_label": "infant",  # Center subject is often the child
                        "secondary_label": "parent",
                    },
                    "temporal_consistency": {
                        "enabled": True,
                        "min_detections": 3,
                        "consistency_threshold": 0.6,
                    },
                },
            },
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)

        self.logger = logging.getLogger(__name__)
        self.model = None
        self.identity_manager = None  # Will be initialized per video

    def process(
        self,
        video_path: str,
        start_time: float = 0.0,
        end_time: float | None = None,
        pps: float = 5,  # 5 predictions per second for person tracking
        output_dir: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Process video for person detection and tracking with identity management.

        Returns:
            List of COCO format annotation dictionaries with person detection, pose results, and identity information.
        """

        # Get video metadata
        video_metadata = self._get_video_metadata(video_path)

        # Initialize PersonIdentityManager if person identity is enabled
        if self.config.get("person_identity", {}).get("enabled", True):
            id_format = self.config["person_identity"].get("id_format", "semantic")
            self.identity_manager = PersonIdentityManager(
                video_metadata["video_id"], id_format
            )
            self.logger.info(
                f"Initialized PersonIdentityManager for video: {video_metadata['video_id']}"
            )
        else:
            self.identity_manager = None

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
            for frame_num in range(
                start_frame, min(end_frame, total_frames), frame_step
            ):
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

        # Apply automatic person labeling if enabled
        if self.identity_manager is not None and annotations:
            auto_labeling_config = self.config.get("person_identity", {}).get(
                "automatic_labeling", {}
            )
            if auto_labeling_config.get("enabled", True):
                self.logger.info("Applying automatic person labeling...")

                # Get person tracks for labeling
                person_tracks = [ann for ann in annotations if ann.get("person_id")]

                # Apply automatic labeling
                automatic_labels = infer_person_labels_from_tracks(
                    person_tracks, annotations, auto_labeling_config
                )

                # Update identity manager with automatic labels
                for person_id, label_info in automatic_labels.items():
                    if label_info["confidence"] >= auto_labeling_config.get(
                        "confidence_threshold", 0.7
                    ):
                        self.identity_manager.set_person_label(
                            person_id,
                            label_info["label"],
                            label_info["confidence"],
                            label_info["method"],
                        )
                        self.logger.info(
                            f"Auto-labeled {person_id}: {label_info['label']} "
                            f"(confidence={label_info['confidence']:.2f})"
                        )

                # Update annotations with new labels
                for annotation in annotations:
                    person_id = annotation.get("person_id")
                    if person_id:
                        label_info = self.identity_manager.get_person_label(person_id)
                        if label_info:
                            annotation["person_label"] = label_info["label"]
                            annotation["label_confidence"] = label_info["confidence"]
                            annotation["labeling_method"] = label_info["method"]

        # Size-based analysis pass (independent of identity manager labels)
        size_cfg = self.config.get("size_analysis", {})
        if size_cfg.get("enabled", True) and annotations:
            try:
                # Run analyzer on all annotations that have person_id
                size_labels = run_size_based_analysis(
                    [a for a in annotations if a.get("person_id")],
                    height_threshold=size_cfg.get("height_threshold", 0.4),
                    confidence=size_cfg.get("confidence", 0.7),
                )
                # Merge results without overwriting existing higher-confidence labels
                for ann in annotations:
                    pid = ann.get("person_id")
                    if pid and pid in size_labels:
                        existing_conf = ann.get("label_confidence")
                        new_conf = size_labels[pid]["confidence"]
                        if (existing_conf is None) or (new_conf >= existing_conf):
                            ann["person_label"] = size_labels[pid]["label"]
                            ann["label_confidence"] = new_conf
                            ann["labeling_method"] = size_labels[pid]["method"]
                            ann["size_based_reasoning"] = size_labels[pid].get(
                                "reasoning"
                            )
                self.logger.info(
                    "Size-based person analysis applied (%d persons)", len(size_labels)
                )
            except Exception as e:
                self.logger.warning(f"Size-based analysis failed: {e}")

        # Save results if output directory specified
        if output_dir and annotations:
            self._save_coco_annotations(annotations, output_dir, video_metadata)

            # Save person tracks information if identity manager is enabled
            if self.identity_manager is not None:
                person_tracks_path = (
                    Path(output_dir)
                    / f"{video_metadata['video_id']}_person_tracks.json"
                )
                detection_summary = {
                    "total_detections": len(annotations),
                    "unique_persons": len(self.identity_manager.get_all_person_ids()),
                    "labeled_persons": len(
                        [
                            p
                            for p in self.identity_manager.get_all_person_ids()
                            if self.identity_manager.get_person_label(p)
                        ]
                    ),
                }
                self.identity_manager.save_person_tracks(
                    str(person_tracks_path), detection_summary
                )
                self.logger.info(f"Saved person tracks to: {person_tracks_path}")

        self.logger.info(f"Person tracking complete: {len(annotations)} detections")
        return annotations

    def _get_video_metadata(self, video_path: str) -> dict[str, Any]:
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
        """Initialize YOLO11 pose model with enhanced logging."""
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics package required for person tracking")

        try:
            # Load model with enhanced download logging
            self.model = log_model_download(
                "YOLO11 Pose Detection Model",
                self.config["model"],
                YOLO,
                self.config["model"],
            )
            # ASCII-safe success marker
            self.logger.info(f"[OK] YOLO model ready: {self.config['model']}")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize YOLO model: {e}")
            raise

    def _process_frame(
        self, frame: np.ndarray, timestamp: float, video_id: str, frame_number: int
    ) -> list[dict[str, Any]]:
        """Process a single frame for person detection and pose estimation."""

        height, width = frame.shape[:2]

        # Run YOLO inference with error recovery
        try:
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
                    frame,
                    conf=self.config["conf_threshold"],
                    iou=self.config["iou_threshold"],
                )
        except AttributeError as e:
            if "'Conv' object has no attribute 'bn'" in str(e):
                self.logger.warning(
                    "Model corruption detected, reinitializing YOLO model..."
                )
                try:
                    # Reinitialize the model
                    self._initialize_model()
                    # Retry the inference
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
                            frame,
                            conf=self.config["conf_threshold"],
                            iou=self.config["iou_threshold"],
                        )
                except Exception as retry_e:
                    self.logger.error(
                        f"Failed to recover from model corruption: {retry_e}"
                    )
                    return []
            else:
                raise

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
                    keypoints_data = (
                        result.keypoints.data.cpu().numpy()
                    )  # Shape: (N, 17, 3)

                for i, box in enumerate(boxes):
                    # Filter for person class (class 0 in COCO)
                    if int(box.cls[0]) == 0:  # Person class
                        # Get bounding box in COCO format [x, y, width, height]
                        x1, y1, x2, y2 = box.xyxy[0]
                        bbox_coco = [
                            float(x1),
                            float(y1),
                            float(x2 - x1),
                            float(y2 - y1),
                        ]

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
                                    2
                                    if conf > self.config["min_keypoint_confidence"]
                                    else 0
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

                        # Add person identity information if enabled
                        if self.identity_manager is not None:
                            person_id = self.identity_manager.register_track(track_id)
                            annotation["person_id"] = person_id

                            # Add person labeling fields if available
                            label_info = self.identity_manager.get_person_label(
                                person_id
                            )
                            if label_info:
                                annotation["person_label"] = label_info["label"]
                                annotation["label_confidence"] = label_info[
                                    "confidence"
                                ]
                                annotation["labeling_method"] = label_info["method"]

                        annotations.append(annotation)

        return annotations

    def _save_coco_annotations(
        self,
        annotations: list[dict[str, Any]],
        output_dir: str,
        video_metadata: dict[str, Any],
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
        self.is_initialized = False

    def initialize(self) -> None:
        """Initialize the person tracking pipeline by loading the YOLO model."""
        if self.is_initialized:
            return

        self.logger.info("Initializing PersonTrackingPipeline...")

        try:
            if not YOLO_AVAILABLE:
                raise ImportError(
                    "Ultralytics YOLO not available. Install with: pip install ultralytics"
                )

            self._initialize_model()
            self.set_model_info("yolo", self.config["model"])
            self.is_initialized = True
            self.logger.info("PersonTrackingPipeline initialization complete")

        except Exception as e:
            self.logger.error(f"Failed to initialize PersonTrackingPipeline: {e}")
            raise

    def get_schema(self) -> dict[str, Any]:
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
                    "category_id": {
                        "type": "integer",
                        "description": "COCO category ID (1 for person)",
                    },
                    "bbox": {
                        "type": "array",
                        "description": "Bounding box [x, y, width, height]",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "area": {"type": "number", "description": "Bounding box area"},
                    "iscrowd": {
                        "type": "integer",
                        "description": "0 for individual objects",
                    },
                    "keypoints": {
                        "type": "array",
                        "description": "COCO-17 keypoints [x1,y1,v1, x2,y2,v2, ...]",
                        "items": {"type": "number"},
                    },
                    "num_keypoints": {
                        "type": "integer",
                        "description": "Number of visible keypoints",
                    },
                    "score": {"type": "number", "description": "Detection confidence"},
                    "track_id": {
                        "type": ["integer", "null"],
                        "description": "Tracking ID across frames",
                    },
                    # Person identity extensions
                    "person_id": {
                        "type": ["string", "null"],
                        "description": "Consistent person identifier",
                    },
                    "person_label": {
                        "type": ["string", "null"],
                        "description": "Semantic person label",
                    },
                    "label_confidence": {
                        "type": ["number", "null"],
                        "description": "Person label confidence",
                    },
                    "labeling_method": {
                        "type": ["string", "null"],
                        "description": "How the label was assigned",
                    },
                    # VideoAnnotator extensions
                    "timestamp": {
                        "type": ["number", "null"],
                        "description": "Frame timestamp in seconds",
                    },
                    "frame_number": {
                        "type": ["integer", "null"],
                        "description": "Frame number",
                    },
                },
                "required": [
                    "id",
                    "image_id",
                    "category_id",
                    "bbox",
                    "area",
                    "iscrowd",
                ],
            },
        }
