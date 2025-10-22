"""Standards-only face analysis pipeline.

This pipeline works directly with COCO format annotations, eliminating
all custom schema dependencies. Uses native FOSS libraries for all data
representation and export.
"""

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .exporters.native_formats import (
    create_coco_annotation,
    create_coco_image_entry,
    export_coco_json,
    validate_coco_json,
)
from .pipelines.base_pipeline import BasePipeline
from .utils.person_identity import PersonIdentityManager
from .version import __version__

# Optional import for enhanced face analysis
try:
    from deepface import DeepFace

    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False


class FaceAnalysisPipeline(BasePipeline):
    """Standards-only face analysis pipeline using COCO format.

    Returns native COCO annotation dictionaries instead of custom
    schemas.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the FaceAnalysisPipeline with optional configuration.

        Args:
            config: Optional configuration dictionary to override defaults.
        """
        default_config = {
            "detection_backend": "deepface",  # opencv, deepface
            "emotion_backend": "deepface",  # deepface, disabled
            "confidence_threshold": 0.7,
            "min_face_size": 30,  # Minimum face size in pixels
            "scale_factor": 1.1,  # For OpenCV Haar cascades
            "min_neighbors": 5,  # For OpenCV Haar cascades
            "max_faces": 10,  # Maximum faces to detect per frame
            # Analysis features
            "detect_emotions": True,
            "detect_age": True,
            "detect_gender": True,
            # DeepFace settings
            "deepface": {
                "emotion_model": "VGG-Face",
                "age_gender_model": "VGG-Face",
                "detector_backend": "opencv",
                "enforce_detection": False,
            },
            # Person identity configuration
            "person_identity": {
                "enabled": True,
                "link_to_persons": True,
                "iou_threshold": 0.5,
                "require_person_id": False,  # Graceful fallback
            },
        }
        # Merge with default config
        merged_config = default_config.copy()
        if config:
            merged_config.update(config)
        super().__init__(merged_config)
        self.face_cascade = None
        self.identity_manager = None  # Will be initialized during processing
        self.logger = logging.getLogger(__name__)

    def initialize(self) -> None:
        """Initialize the face analysis backend."""
        self.logger.info(
            f"Initializing FaceAnalysisPipeline with backend: {self.config['detection_backend']}"
        )
        self._initialize_backend()
        self.is_initialized = True

    def get_schema(self) -> dict[str, Any]:
        """Get the output schema for face analysis annotations."""
        return {
            "type": "coco_annotation",
            "format_version": __version__,
            "categories": [{"id": 1, "name": "face", "supercategory": "person"}],
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
                # Person identity fields
                "person_id": {
                    "type": ["string", "null"],
                    "description": "Linked person identifier",
                },
                "person_label": {
                    "type": ["string", "null"],
                    "description": "Person semantic label",
                },
                "person_label_confidence": {
                    "type": ["number", "null"],
                    "description": "Person label confidence",
                },
                "person_labeling_method": {
                    "type": ["string", "null"],
                    "description": "How person was labeled",
                },
                # VideoAnnotator extensions
                "timestamp": {
                    "type": ["number", "null"],
                    "description": "Frame timestamp",
                },
                "frame_number": {
                    "type": ["integer", "null"],
                    "description": "Frame number",
                },
                # DeepFace analysis results
                "emotion": {
                    "type": ["string", "null"],
                    "description": "Dominant emotion detected",
                },
                "emotion_confidence": {
                    "type": ["number", "null"],
                    "description": "Confidence of emotion prediction",
                },
                "emotion_scores": {
                    "type": ["object", "null"],
                    "description": "All emotion scores",
                },
                "age": {"type": ["number", "null"], "description": "Predicted age"},
                "gender": {
                    "type": ["string", "null"],
                    "description": "Predicted gender",
                },
                "gender_confidence": {
                    "type": ["number", "null"],
                    "description": "Confidence of gender prediction",
                },
                "gender_scores": {
                    "type": ["object", "null"],
                    "description": "All gender scores",
                },
                "backend": {
                    "type": "string",
                    "description": "Face detection backend used",
                },
            },
        }

    def process(
        self,
        video_path: str,
        start_time: float = 0.0,
        end_time: float | None = None,
        pps: float = 5.0,  # 5 FPS for face analysis
        output_dir: str | None = None,
        person_tracks: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Process video for face analysis with person identity linking.

        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            end_time: End time in seconds (None for full video)
            pps: Frames per second to process
            output_dir: Output directory (optional)
            person_tracks: Optional person tracking data to associate faces with

        Returns:
            List of COCO format annotation dictionaries with face detection results and person identity information.
        """
        # Get video metadata
        video_metadata = self._get_video_metadata(video_path)

        # Initialize PersonIdentityManager if person linking is enabled
        person_config = self.config.get("person_identity", {})
        if person_config.get("enabled", True) and person_config.get(
            "link_to_persons", True
        ):
            if person_tracks:
                # Create identity manager from existing person tracks
                self.identity_manager = PersonIdentityManager.from_person_tracks(
                    person_tracks, video_metadata["video_id"]
                )
                self.logger.info(
                    f"Loaded PersonIdentityManager with {len(person_tracks)} person tracks"
                )
            else:
                # Try to load person tracks from default location
                person_tracks = self._load_person_tracks(
                    video_metadata["video_id"], output_dir
                )
                if person_tracks:
                    self.identity_manager = PersonIdentityManager.from_person_tracks(
                        person_tracks, video_metadata["video_id"]
                    )
                    self.logger.info(
                        f"Loaded PersonIdentityManager from file with {len(person_tracks)} person tracks"
                    )
                elif not person_config.get("require_person_id", False):
                    self.identity_manager = None
                    self.logger.info(
                        "No person tracks available, proceeding without person linking"
                    )
                else:
                    raise ValueError("Person tracks required but not available")
        else:
            self.identity_manager = None

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
            for frame_num in range(
                start_frame, min(end_frame, total_frames), frame_step
            ):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()

                if not ret:
                    continue

                timestamp = frame_num / fps
                height, width = frame.shape[:2]

                # Detect faces in frame
                face_annotations = self._detect_faces_in_frame(
                    frame,
                    timestamp,
                    video_metadata["video_id"],
                    frame_num,
                    width,
                    height,
                )

                # Link faces to persons if identity manager is available
                if self.identity_manager is not None and face_annotations:
                    # Get person annotations for this frame from person tracks
                    frame_person_annotations = self._get_frame_person_annotations(
                        timestamp, frame_num
                    )

                    # Link each face to a person using IoU matching
                    for face_ann in face_annotations:
                        person_id = self._link_face_to_person(
                            face_ann, frame_person_annotations
                        )
                        if person_id:
                            face_ann["person_id"] = person_id

                            # Add person label information if available
                            label_info = self.identity_manager.get_person_label(
                                person_id
                            )
                            if label_info:
                                face_ann["person_label"] = label_info["label"]
                                face_ann["person_label_confidence"] = label_info[
                                    "confidence"
                                ]
                                face_ann["person_labeling_method"] = label_info[
                                    "method"
                                ]

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

    def _load_person_tracks(
        self, video_id: str, output_dir: str | None = None
    ) -> list[dict[str, Any]] | None:
        """Load person tracking data from file."""
        person_data_paths = [
            Path(output_dir or ".") / f"person_tracking_{video_id}.json",
            Path("data") / f"person_tracking_{video_id}.json",
            Path("demo_results") / f"person_tracking_{video_id}.json",
        ]

        for person_data_path in person_data_paths:
            if person_data_path.exists():
                try:
                    with open(person_data_path) as f:
                        person_data = json.load(f)

                    # Extract annotations with person tracks
                    if "annotations" in person_data:
                        self.logger.info(
                            f"Loaded person tracks from {person_data_path}"
                        )
                        return person_data["annotations"]
                except Exception as e:
                    self.logger.warning(
                        f"Failed to load person data from {person_data_path}: {e}"
                    )

        return None

    def _get_frame_person_annotations(
        self, timestamp: float, frame_num: int
    ) -> list[dict[str, Any]]:
        """Get person annotations for a specific frame from the loaded identity manager.

        Returns an empty list when no identity manager is present. In a full
        implementation this would search loaded person tracks by timestamp or
        frame number.
        """
        if not self.identity_manager:
            return []

        # This would typically load from cached person tracks data
        # For now, return empty list as the identity manager doesn't store frame-specific data
        # In a full implementation, this would search loaded person tracks by timestamp/frame
        return []

    def _link_face_to_person(
        self, face_annotation: dict[str, Any], person_annotations: list[dict[str, Any]]
    ) -> str | None:
        """Link a face detection to a person using IoU matching via PersonIdentityManager.

        Returns the linked person identifier or None if linking is not
        possible.
        """
        if not self.identity_manager or not person_annotations:
            return None

        face_bbox = face_annotation.get("bbox", [])
        if len(face_bbox) != 4:
            return None

        # Use PersonIdentityManager's linking method
        return self.identity_manager.link_face_to_person(
            face_bbox,
            person_annotations,
            iou_threshold=self.config["person_identity"]["iou_threshold"],
        )

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

    def _initialize_backend(self):
        """Initialize the selected detection backend."""
        backend = self.config["detection_backend"]

        if backend == "opencv":
            # OpenCV Haar cascades are built-in
            self.logger.info("Using OpenCV face detection")

        elif backend == "deepface" and DEEPFACE_AVAILABLE:
            # DeepFace will be initialized on first use
            self.logger.info("Using DeepFace detection")

        else:
            self.logger.warning(
                f"Backend {backend} not available, falling back to OpenCV"
            )
            self.config["detection_backend"] = "opencv"

    def _detect_faces_in_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        video_id: str,
        frame_number: int,
        width: int,
        height: int,
    ) -> list[dict[str, Any]]:
        """Detect faces in a single frame and return COCO annotations."""
        backend = self.config["detection_backend"]

        if backend == "deepface" and DEEPFACE_AVAILABLE:
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
    ) -> list[dict[str, Any]]:
        """Detect faces using OpenCV Haar cascades."""
        # Load cascade classifier
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rects = face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.config.get("scale_factor", 1.1),
            minNeighbors=self.config.get("min_neighbors", 5),
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

    def _detect_faces_deepface(
        self,
        frame: np.ndarray,
        timestamp: float,
        video_id: str,
        frame_number: int,
        width: int,
        height: int,
    ) -> list[dict[str, Any]]:
        """Detect faces using DeepFace with comprehensive analysis."""
        try:
            # DeepFace expects RGB format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get DeepFace configuration
            deepface_config = self.config.get("deepface", {})
            detector_backend = deepface_config.get("detector_backend", "opencv")
            enforce_detection = deepface_config.get("enforce_detection", False)

            annotations = []

            # First, detect faces and get their regions
            try:
                face_objs = DeepFace.extract_faces(
                    rgb_frame,
                    detector_backend=detector_backend,
                    enforce_detection=enforce_detection,
                    align=True,
                    grayscale=False,
                )

                if not face_objs:
                    return []

                # Get face regions with coordinates using DeepFace.analyze
                analyze_results = DeepFace.analyze(
                    rgb_frame,
                    actions=["emotion", "age", "gender"]
                    if (
                        self.config.get("detect_emotions", True)
                        or self.config.get("detect_age", True)
                        or self.config.get("detect_gender", True)
                    )
                    else ["emotion"],  # At least one action is required
                    detector_backend=detector_backend,
                    enforce_detection=enforce_detection,
                )

                # Handle single face vs multiple faces
                if not isinstance(analyze_results, list):
                    analyze_results = [analyze_results]

                for i, analysis in enumerate(analyze_results):
                    # Extract face region coordinates
                    region = analysis.get("region", {})
                    x = region.get("x", 0)
                    y = region.get("y", 0)
                    w = region.get("w", 100)
                    h = region.get("h", 100)

                    # Skip faces that are too small
                    if (
                        w < self.config["min_face_size"]
                        or h < self.config["min_face_size"]
                    ):
                        continue

                    # Prepare analysis data
                    analysis_data = {}

                    # Emotion analysis
                    if (
                        self.config.get("detect_emotions", True)
                        and "emotion" in analysis
                    ):
                        emotions = analysis["emotion"]
                        # Get dominant emotion
                        dominant_emotion = analysis.get("dominant_emotion", "unknown")
                        emotion_confidence = (
                            emotions.get(dominant_emotion, 0.0) / 100.0
                            if dominant_emotion != "unknown"
                            else 0.0
                        )

                        analysis_data.update(
                            {
                                "emotion": dominant_emotion,
                                "emotion_confidence": float(emotion_confidence),
                                "emotion_scores": {
                                    k: float(v / 100.0) for k, v in emotions.items()
                                },
                            }
                        )

                    # Age analysis
                    if self.config.get("detect_age", True) and "age" in analysis:
                        analysis_data["age"] = float(analysis["age"])

                    # Gender analysis
                    if self.config.get("detect_gender", True) and "gender" in analysis:
                        gender_scores = analysis.get("gender", {})
                        dominant_gender = analysis.get("dominant_gender", "unknown")
                        gender_confidence = (
                            gender_scores.get(dominant_gender, 0.0) / 100.0
                            if dominant_gender != "unknown"
                            else 0.0
                        )

                        analysis_data.update(
                            {
                                "gender": dominant_gender,
                                "gender_confidence": float(gender_confidence),
                                "gender_scores": {
                                    k: float(v / 100.0)
                                    for k, v in gender_scores.items()
                                },
                            }
                        )

                    # Create COCO annotation with DeepFace analysis
                    annotation = create_coco_annotation(
                        annotation_id=0,  # Will be set later
                        image_id=f"{video_id}_frame_{frame_number}",
                        category_id=100,  # Face category
                        bbox=[float(x), float(y), float(w), float(h)],
                        score=1.0,
                        # VideoAnnotator extensions
                        face_id=i,
                        timestamp=timestamp,
                        frame_number=frame_number,
                        backend="deepface",
                        **analysis_data,  # Add all analysis results
                    )
                    annotations.append(annotation)

                    # Limit number of faces
                    if len(annotations) >= self.config["max_faces"]:
                        break

            except Exception as analysis_error:
                self.logger.warning(f"DeepFace analysis failed: {analysis_error}")
                # Fallback to simple detection only
                try:
                    face_objs = DeepFace.extract_faces(
                        rgb_frame,
                        detector_backend=detector_backend,
                        enforce_detection=False,
                        align=False,
                    )

                    for i, face_obj in enumerate(face_objs):
                        if face_obj is not None:
                            # Create basic annotation without analysis
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

                            if len(annotations) >= self.config["max_faces"]:
                                break

                except Exception as fallback_error:
                    self.logger.warning(
                        f"DeepFace fallback detection also failed: {fallback_error}"
                    )

            return annotations

        except Exception as e:
            self.logger.warning(f"DeepFace detection failed completely: {e}")
            return []

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
        # Reset attributes
        self.face_cascade = None
        self.identity_manager = None
        self.is_initialized = False
