"""
OpenFace 3.0 face analysis pipeline.

This pipeline integrates CMU's OpenFace 3.0 for comprehensive facial behavior analysis,
including facial landmarks, action units, head pose, and gaze estimation.
Uses COCO format for compatibility with the VideoAnnotator standards.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import time
import json

from ..base_pipeline import BasePipeline
from ...exporters.native_formats import (
    create_coco_annotation,
    create_coco_image_entry,
    export_coco_json,
    validate_coco_json,
)

# Optional import for OpenFace 3.0
try:
    import openface3
    OPENFACE3_AVAILABLE = True
except ImportError:
    OPENFACE3_AVAILABLE = False
    logging.warning("OpenFace 3.0 not available. Install from https://github.com/CMU-MultiComp-Lab/OpenFace-3.0")


class OpenFace3Pipeline(BasePipeline):
    """
    OpenFace 3.0 face analysis pipeline using COCO format.
    
    Provides comprehensive facial behavior analysis including:
    - 2D and 3D facial landmarks (68-point model)
    - Facial Action Units (AU) intensity and presence
    - Head pose estimation (rotation and translation)
    - Gaze direction and eye gaze
    - Basic emotion recognition
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "detection_confidence": 0.7,
            "landmark_model": "68_point",  # 68_point, 51_point
            "enable_3d_landmarks": True,
            "enable_action_units": True,
            "enable_head_pose": True,
            "enable_gaze": True,
            "enable_emotions": False,  # Experimental feature
            "batch_size": 1,
            "device": "auto",  # auto, cpu, cuda
            "model_path": None,  # Path to custom models
            "max_faces": 5,  # Maximum faces to track
            "track_faces": True,  # Enable face tracking across frames
        }
        
        # Merge with default config
        merged_config = default_config.copy()
        if config:
            merged_config.update(config)
            
        super().__init__(merged_config)
        
        # OpenFace 3.0 components
        self.face_detector = None
        self.landmark_detector = None
        self.au_analyzer = None
        self.head_pose_estimator = None
        self.gaze_estimator = None
        
        # Face tracking state
        self.face_tracker = None
        self.tracked_faces = {}
        self.next_face_id = 0
        
        # Performance metrics
        self.processing_times = []
        
        self.logger = logging.getLogger(__name__)

    def initialize(self) -> None:
        """Initialize OpenFace 3.0 components."""
        if not OPENFACE3_AVAILABLE:
            raise ImportError(
                "OpenFace 3.0 is not installed. Please install it from "
                "https://github.com/CMU-MultiComp-Lab/OpenFace-3.0"
            )
        
        try:
            # Determine device
            device = self.config["device"]
            if device == "auto":
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"
            
            self.logger.info(f"Initializing OpenFace 3.0 on device: {device}")
            
            # Initialize face detector
            self.face_detector = openface3.FaceDetector(
                confidence_threshold=self.config["detection_confidence"],
                device=device
            )
            
            # Initialize landmark detector
            model_type = self.config["landmark_model"]
            self.landmark_detector = openface3.LandmarkDetector(
                model_type=model_type,
                enable_3d=self.config["enable_3d_landmarks"],
                device=device
            )
            
            # Initialize action unit analyzer
            if self.config["enable_action_units"]:
                self.au_analyzer = openface3.ActionUnitAnalyzer(device=device)
            
            # Initialize head pose estimator
            if self.config["enable_head_pose"]:
                self.head_pose_estimator = openface3.HeadPoseEstimator(device=device)
            
            # Initialize gaze estimator
            if self.config["enable_gaze"]:
                self.gaze_estimator = openface3.GazeEstimator(device=device)
            
            # Initialize face tracker
            if self.config["track_faces"]:
                self.face_tracker = openface3.FaceTracker(
                    max_faces=self.config["max_faces"]
                )
            
            self._model_info = {
                "model_name": "OpenFace 3.0",
                "version": openface3.__version__,
                "device": device,
                "landmark_model": model_type,
                "features": {
                    "landmarks": True,
                    "3d_landmarks": self.config["enable_3d_landmarks"],
                    "action_units": self.config["enable_action_units"],
                    "head_pose": self.config["enable_head_pose"],
                    "gaze": self.config["enable_gaze"],
                    "face_tracking": self.config["track_faces"],
                }
            }
            
            self.is_initialized = True
            self.logger.info("OpenFace 3.0 pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenFace 3.0: {e}")
            raise

    def get_schema(self) -> Dict[str, Any]:
        """Get the output schema for OpenFace 3.0 face analysis annotations."""
        return {
            "type": "coco_annotation",
            "format_version": "1.0",
            "categories": [
                {
                    "id": 1,
                    "name": "face",
                    "supercategory": "person"
                }
            ],
            "annotation_schema": {
                "id": "integer",
                "image_id": "integer", 
                "category_id": "integer",
                "bbox": "array[4]",  # [x, y, width, height]
                "area": "float",
                "iscrowd": "integer",
                "keypoints": "array[196]",  # 98 landmarks * 2 coordinates (x, y)
                "num_keypoints": "integer",
                "confidence": "float",
                "attributes": {
                    "action_units": "object",  # AU intensities and presence
                    "head_pose": {
                        "rotation": "array[3]",  # [rx, ry, rz] in radians
                        "translation": "array[3]"  # [tx, ty, tz] in mm
                    },
                    "gaze": {
                        "direction": "array[3]",  # 3D gaze direction vector
                        "left_eye": "array[2]",   # 2D gaze point for left eye
                        "right_eye": "array[2]"   # 2D gaze point for right eye
                    },
                    "emotion": "string",
                    "landmark_3d": "array[294]"  # 98 landmarks * 3 coordinates (x, y, z)
                }
            }
        }

    def process(
        self,
        video_path: str,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        pps: float = 1.0,
        output_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process video with OpenFace 3.0 face analysis.
        
        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            end_time: End time in seconds (None = full video)
            pps: Predictions per second
            output_dir: Directory to save results
            
        Returns:
            List of COCO format annotations with face analysis data
        """
        if not self.is_initialized:
            self.initialize()
        
        self.logger.info(f"Processing video: {video_path}")
        start_processing_time = time.time()
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        
        # Calculate frame range
        start_frame = int(start_time * fps)
        if end_time is None:
            end_frame = total_frames
            end_time = video_duration
        else:
            end_frame = min(int(end_time * fps), total_frames)
        
        # Calculate frame sampling
        if pps <= 0:
            # Process once per segment
            frames_to_process = [start_frame]
        else:
            # Process at specified rate
            frame_interval = max(1, int(fps / pps))
            frames_to_process = list(range(start_frame, end_frame, frame_interval))
        
        self.logger.info(f"Processing {len(frames_to_process)} frames at {pps} PPS")
        
        # COCO dataset structure
        annotations = []
        images = []
        categories = self._get_face_categories()
        
        annotation_id = 1
        
        # Process frames
        for frame_idx, frame_num in enumerate(frames_to_process):
            try:
                # Seek to frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning(f"Could not read frame {frame_num}")
                    continue
                
                # Calculate timestamp
                timestamp = frame_num / fps
                
                # Create COCO image entry
                image_entry = create_coco_image_entry(
                    image_id=frame_idx + 1,
                    width=frame.shape[1],
                    height=frame.shape[0],
                    file_name=f"frame_{frame_num:06d}.jpg",
                    timestamp=timestamp
                )
                images.append(image_entry)
                
                # Process frame with OpenFace 3.0
                face_results = self._process_frame(frame, timestamp)
                
                # Convert to COCO annotations
                for face_result in face_results:
                    annotation = self._create_face_annotation(
                        annotation_id=annotation_id,
                        image_id=frame_idx + 1,
                        face_data=face_result,
                        timestamp=timestamp
                    )
                    annotations.append(annotation)
                    annotation_id += 1
                
            except Exception as e:
                self.logger.error(f"Error processing frame {frame_num}: {e}")
                continue
        
        cap.release()
        
        # Create COCO dataset
        coco_dataset = {
            "info": {
                "description": "OpenFace 3.0 Face Analysis",
                "version": "1.0",
                "year": 2025,
                "contributor": "VideoAnnotator",
                "date_created": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "images": images,
            "annotations": annotations,
            "categories": categories
        }
        
        # Save results if output directory specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            video_name = Path(video_path).stem
            output_file = output_path / f"{video_name}_openface3_analysis.json"
            
            export_coco_json(coco_dataset, str(output_file))
            
            # Validate COCO format
            if validate_coco_json(str(output_file)):
                self.logger.info(f"OpenFace 3.0 analysis saved and validated: {output_file}")
            
            # Save detailed results
            detailed_file = output_path / f"{video_name}_openface3_detailed.json"
            self._save_detailed_results(detailed_file, annotations)
        
        processing_time = time.time() - start_processing_time
        self.processing_times.append(processing_time)
        
        self.logger.info(
            f"OpenFace 3.0 analysis complete: {len(annotations)} faces detected "
            f"in {processing_time:.2f}s"
        )
        
        return [coco_dataset]

    def _process_frame(self, frame: np.ndarray, timestamp: float) -> List[Dict[str, Any]]:
        """Process a single frame with OpenFace 3.0."""
        face_results = []
        
        try:
            # Detect faces
            face_detections = self.face_detector.detect(frame)
            
            if len(face_detections) == 0:
                return face_results
            
            # Process each detected face
            for i, detection in enumerate(face_detections):
                face_data = {
                    "timestamp": timestamp,
                    "face_id": i,
                    "detection": detection,
                }
                
                # Extract face region
                x, y, w, h = detection["bbox"]
                face_roi = frame[y:y+h, x:x+w]
                
                # Get facial landmarks
                landmarks = self.landmark_detector.detect(face_roi)
                if landmarks is not None:
                    face_data["landmarks_2d"] = landmarks["landmarks_2d"]
                    if self.config["enable_3d_landmarks"] and "landmarks_3d" in landmarks:
                        face_data["landmarks_3d"] = landmarks["landmarks_3d"]
                
                # Get action units
                if self.au_analyzer and landmarks is not None:
                    aus = self.au_analyzer.analyze(face_roi, landmarks)
                    if aus is not None:
                        face_data["action_units"] = aus
                
                # Get head pose
                if self.head_pose_estimator and landmarks is not None:
                    head_pose = self.head_pose_estimator.estimate(landmarks)
                    if head_pose is not None:
                        face_data["head_pose"] = head_pose
                
                # Get gaze direction
                if self.gaze_estimator and landmarks is not None:
                    gaze = self.gaze_estimator.estimate(face_roi, landmarks)
                    if gaze is not None:
                        face_data["gaze"] = gaze
                
                # Face tracking
                if self.face_tracker:
                    track_id = self.face_tracker.track(detection)
                    face_data["track_id"] = track_id
                
                face_results.append(face_data)
                
        except Exception as e:
            self.logger.error(f"Error processing frame at {timestamp:.2f}s: {e}")
        
        return face_results

    def _create_face_annotation(
        self,
        annotation_id: int,
        image_id: int,
        face_data: Dict[str, Any],
        timestamp: float
    ) -> Dict[str, Any]:
        """Create COCO annotation from face analysis data."""
        detection = face_data["detection"]
        bbox = detection["bbox"]  # [x, y, w, h]
        
        # Create base annotation
        annotation = create_coco_annotation(
            annotation_id=annotation_id,
            image_id=image_id,
            category_id=1,  # Face category
            bbox=bbox,
            area=bbox[2] * bbox[3],
            segmentation=[],
            iscrowd=0
        )
        
        # Add OpenFace 3.0 specific data
        openface_data = {
            "confidence": detection.get("confidence", 0.0),
            "timestamp": timestamp,
        }
        
        # Add facial landmarks
        if "landmarks_2d" in face_data:
            landmarks_2d = face_data["landmarks_2d"]
            # Convert to COCO keypoints format: [x1, y1, v1, x2, y2, v2, ...]
            keypoints = []
            for point in landmarks_2d:
                keypoints.extend([point[0], point[1], 2])  # 2 = visible
            
            annotation["keypoints"] = keypoints
            annotation["num_keypoints"] = len(landmarks_2d)
            openface_data["landmarks_2d"] = landmarks_2d.tolist()
        
        if "landmarks_3d" in face_data:
            openface_data["landmarks_3d"] = face_data["landmarks_3d"].tolist()
        
        # Add action units
        if "action_units" in face_data:
            openface_data["action_units"] = face_data["action_units"]
        
        # Add head pose
        if "head_pose" in face_data:
            openface_data["head_pose"] = face_data["head_pose"]
        
        # Add gaze information
        if "gaze" in face_data:
            openface_data["gaze"] = face_data["gaze"]
        
        # Add tracking ID
        if "track_id" in face_data:
            openface_data["track_id"] = face_data["track_id"]
        
        # Store OpenFace data in custom field
        annotation["openface3"] = openface_data
        
        return annotation

    def _get_face_categories(self) -> List[Dict[str, Any]]:
        """Get COCO categories for face analysis."""
        return [
            {
                "id": 1,
                "name": "face",
                "supercategory": "person",
                "keypoints": [
                    f"landmark_{i}" for i in range(68)  # 68-point model
                ],
                "skeleton": []  # Face landmarks don't have skeleton connections
            }
        ]

    def _save_detailed_results(self, output_file: Path, annotations: List[Dict]) -> None:
        """Save detailed OpenFace 3.0 results with all extracted features."""
        detailed_results = {
            "metadata": {
                "pipeline": "OpenFace3Pipeline",
                "model_info": self._model_info,
                "config": self.config,
                "processing_stats": {
                    "total_faces": len(annotations),
                    "avg_processing_time": np.mean(self.processing_times) if self.processing_times else 0,
                }
            },
            "faces": []
        }
        
        for annotation in annotations:
            if "openface3" in annotation:
                face_entry = {
                    "annotation_id": annotation["id"],
                    "bbox": annotation["bbox"],
                    "timestamp": annotation["openface3"]["timestamp"],
                    "features": annotation["openface3"]
                }
                detailed_results["faces"].append(face_entry)
        
        with open(output_file, 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        self.logger.info(f"Detailed OpenFace 3.0 results saved: {output_file}")

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the OpenFace 3.0 pipeline."""
        info = {
            "name": "OpenFace3Pipeline",
            "version": "1.0.0",
            "description": "Facial behavior analysis using OpenFace 3.0",
            "capabilities": [
                "face_detection",
                "facial_landmarks",
                "action_units",
                "head_pose",
                "gaze_estimation",
                "face_tracking"
            ],
            "output_format": "COCO",
            "dependencies": {
                "openface3": OPENFACE3_AVAILABLE,
                "opencv": True,
                "numpy": True
            }
        }
        
        if self.is_initialized and self._model_info:
            info["model_info"] = self._model_info
        
        return info

    def cleanup(self) -> None:
        """Cleanup resources."""
        if hasattr(self, 'face_detector') and self.face_detector:
            del self.face_detector
        if hasattr(self, 'landmark_detector') and self.landmark_detector:
            del self.landmark_detector
        if hasattr(self, 'au_analyzer') and self.au_analyzer:
            del self.au_analyzer
        if hasattr(self, 'head_pose_estimator') and self.head_pose_estimator:
            del self.head_pose_estimator
        if hasattr(self, 'gaze_estimator') and self.gaze_estimator:
            del self.gaze_estimator
        
        self.logger.info("OpenFace 3.0 pipeline cleaned up")
