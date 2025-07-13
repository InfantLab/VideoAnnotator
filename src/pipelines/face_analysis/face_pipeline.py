"""
Standards-only face analysis pipeline.

This pipeline works directly with COCO format annotations, eliminating all custom schema dependencies.
Uses native FOSS libraries for all data representation and export.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from ..base_pipeline import BasePipeline
from ...exporters.native_formats import (
    create_coco_annotation,
    create_coco_image_entry,
    export_coco_json,
    validate_coco_json,
)

# Optional imports for enhanced face analysis
try:
    from deepface import DeepFace

    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class FaceAnalysisPipeline(BasePipeline):
    """
    Standards-only face analysis pipeline using COCO format.

    Returns native COCO annotation dictionaries instead of custom schemas.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "detection_backend": "opencv",  # opencv, mediapipe, deepface
            "emotion_backend": "deepface",  # deepface, disabled
            "confidence_threshold": 0.7,
            "min_face_size": 30,  # Minimum face size in pixels
            "scale_factor": 1.1,  # For OpenCV Haar cascades
            "min_neighbors": 5,  # For OpenCV Haar cascades
            "max_faces": 10,  # Maximum faces to detect per frame
        }
        # Merge with default config
        merged_config = default_config.copy()
        if config:
            merged_config.update(config)
        super().__init__(merged_config)
        self.face_cascade = None
        self.mp_face_detection = None
        self.mp_face_mesh = None
        self.logger = logging.getLogger(__name__)

    def initialize(self) -> None:
        """Initialize the face analysis backend."""
        self.logger.info(f"Initializing FaceAnalysisPipeline with backend: {self.config['detection_backend']}")
        self._initialize_backend()

    def get_schema(self) -> Dict[str, Any]:
        """Get the output schema for face analysis annotations."""
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
                "keypoints": "array[15]",  # Face landmarks (5 points: 2 eyes, nose, 2 mouth corners)
                "num_keypoints": "integer",
                "confidence": "float",
                "attributes": {
                    "emotion": "string",
                    "age": "integer", 
                    "gender": "string"
                }
            }
        }

    def process(
        self,
        video_path: str,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        pps: float = 5.0,  # 5 FPS for face analysis
        output_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process video for face analysis.

        Returns:
            List of COCO format annotation dictionaries with face detection results.
        """

        # Get video metadata
        video_metadata = self._get_video_metadata(video_path)

        # Initialize detection backend
        self._initialize_backend()

        # Process video frames
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
        frame_step = max(1, int(fps / pps))  # Process every Nth frame

        annotation_id = 1

        try:
            for frame_num in range(start_frame, min(end_frame, total_frames), frame_step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()

                if not ret:
                    continue

                timestamp = frame_num / fps
                height, width = frame.shape[:2]

                # Detect faces in frame
                face_annotations = self._detect_faces_in_frame(
                    frame, timestamp, video_metadata["video_id"], frame_num, width, height
                )

                # Assign unique annotation IDs
                for face_ann in face_annotations:
                    face_ann["id"] = annotation_id
                    annotation_id += 1

                annotations.extend(face_annotations)

        finally:
            cap.release()

        # Save results if output directory specified
        if output_dir and annotations:
            self._save_coco_annotations(annotations, output_dir, video_metadata)

        self.logger.info(f"Face analysis complete: {len(annotations)} detections")
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

    def _initialize_backend(self):
        """Initialize the selected detection backend."""
        backend = self.config["detection_backend"]

        if backend == "mediapipe" and MEDIAPIPE_AVAILABLE:
            self.backends["mediapipe"] = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=self.config["confidence_threshold"]
            )
            self.logger.info("Initialized MediaPipe face detection")

        elif backend == "opencv":
            # OpenCV Haar cascades are built-in
            self.logger.info("Using OpenCV face detection")

        elif backend == "deepface" and DEEPFACE_AVAILABLE:
            # DeepFace will be initialized on first use
            self.logger.info("Using DeepFace detection")

        else:
            self.logger.warning(f"Backend {backend} not available, falling back to OpenCV")
            self.config["detection_backend"] = "opencv"

    def _detect_faces_in_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        video_id: str,
        frame_number: int,
        width: int,
        height: int,
    ) -> List[Dict[str, Any]]:
        """Detect faces in a single frame and return COCO annotations."""

        backend = self.config["detection_backend"]

        if backend == "mediapipe" and "mediapipe" in self.backends:
            return self._detect_faces_mediapipe(
                frame, timestamp, video_id, frame_number, width, height
            )
        elif backend == "deepface" and DEEPFACE_AVAILABLE:
            return self._detect_faces_deepface(
                frame, timestamp, video_id, frame_number, width, height
            )
        else:
            return self._detect_faces_opencv(
                frame, timestamp, video_id, frame_number, width, height
            )

    def _detect_faces_opencv(
        self,
        frame: np.ndarray,
        timestamp: float,
        video_id: str,
        frame_number: int,
        width: int,
        height: int,
    ) -> List[Dict[str, Any]]:
        """Detect faces using OpenCV Haar cascades."""

        # Load cascade classifier
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rects = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.config["min_face_size"], self.config["min_face_size"]),
        )

        annotations = []
        for i, (x, y, w, h) in enumerate(face_rects):
            # Create COCO annotation
            annotation = create_coco_annotation(
                annotation_id=0,  # Will be set later
                image_id=f"{video_id}_frame_{frame_number}",
                category_id=100,  # Face category
                bbox=[float(x), float(y), float(w), float(h)],
                score=1.0,  # OpenCV doesn't provide confidence scores
                # VideoAnnotator extensions
                face_id=i,
                timestamp=timestamp,
                frame_number=frame_number,
                backend="opencv",
            )
            annotations.append(annotation)

        return annotations

    def _detect_faces_mediapipe(
        self,
        frame: np.ndarray,
        timestamp: float,
        video_id: str,
        frame_number: int,
        width: int,
        height: int,
    ) -> List[Dict[str, Any]]:
        """Detect faces using MediaPipe."""

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.backends["mediapipe"].process(rgb_frame)

        annotations = []
        if results.detections:
            for i, detection in enumerate(results.detections):
                bbox_norm = detection.location_data.relative_bounding_box

                # Convert normalized coordinates to pixels
                x = float(bbox_norm.x_center - bbox_norm.width / 2) * width
                y = float(bbox_norm.y_center - bbox_norm.height / 2) * height
                w = float(bbox_norm.width) * width
                h = float(bbox_norm.height) * height

                # Filter by minimum face size
                if min(w, h) >= self.config["min_face_size"]:
                    annotation = create_coco_annotation(
                        annotation_id=0,  # Will be set later
                        image_id=f"{video_id}_frame_{frame_number}",
                        category_id=100,  # Face category
                        bbox=[x, y, w, h],
                        score=float(detection.score[0]),
                        # VideoAnnotator extensions
                        face_id=i,
                        timestamp=timestamp,
                        frame_number=frame_number,
                        backend="mediapipe",
                    )
                    annotations.append(annotation)

        return annotations

    def _detect_faces_deepface(
        self,
        frame: np.ndarray,
        timestamp: float,
        video_id: str,
        frame_number: int,
        width: int,
        height: int,
    ) -> List[Dict[str, Any]]:
        """Detect faces using DeepFace."""

        try:
            # DeepFace expects RGB format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Use DeepFace.extract_faces for detection
            face_objs = DeepFace.extract_faces(
                rgb_frame,
                detector_backend="opencv",  # Use OpenCV backend for speed
                enforce_detection=False,
            )

            annotations = []
            for i, face_obj in enumerate(face_objs):
                if face_obj is not None:
                    # DeepFace doesn't provide bounding box coordinates directly
                    # This would need to be enhanced for production use
                    # For now, create a placeholder annotation
                    annotation = create_coco_annotation(
                        annotation_id=0,  # Will be set later
                        image_id=f"{video_id}_frame_{frame_number}",
                        category_id=100,  # Face category
                        bbox=[0.0, 0.0, 100.0, 100.0],  # Placeholder
                        score=1.0,
                        # VideoAnnotator extensions
                        face_id=i,
                        timestamp=timestamp,
                        frame_number=frame_number,
                        backend="deepface",
                    )
                    annotations.append(annotation)

            return annotations

        except Exception as e:
            self.logger.warning(f"DeepFace detection failed: {e}")
            return []

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

        # Export COCO JSON
        categories = [{"id": 100, "name": "face", "supercategory": "person"}]
        coco_path = output_path / f"{video_metadata['video_id']}_face_detections.json"

        export_coco_json(annotations, images, str(coco_path), categories)

        # Validate COCO format
        validation_result = validate_coco_json(str(coco_path), "face_analysis")
        if validation_result.is_valid:
            self.logger.info(f"Face detection COCO validation successful: {coco_path}")
        else:
            self.logger.warning(
                f"Face detection COCO validation warnings: {', '.join(validation_result.warnings)}"
            )

    def cleanup(self):
        """Cleanup resources."""
        if "mediapipe" in self.backends:
            self.backends["mediapipe"].close()
        self.backends.clear()
