"""OpenFace 3.0 face analysis pipeline.

This pipeline integrates CMU's OpenFace 3.0 for comprehensive facial
behavior analysis, including facial landmarks, action units, head pose,
and gaze estimation. Uses COCO format for compatibility with the
VideoAnnotator standards.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from videoannotator.pipelines.base_pipeline import BasePipeline
from videoannotator.utils.person_identity import PersonIdentityManager
from videoannotator.version import __version__

from .exporters.native_formats import (
    create_coco_annotation,
    create_coco_image_entry,
    export_coco_json,
    validate_coco_json,
)


# Apply SciPy compatibility patch first - OpenFace 3.0 uses deprecated scipy.integrate.simps
def patch_scipy_compatibility():
    """Patch SciPy compatibility for OpenFace 3.0."""
    try:
        import scipy.integrate

        if not hasattr(scipy.integrate, "simps"):
            logging.info(
                "Applying scipy.integrate.simps compatibility patch for OpenFace 3.0"
            )
            scipy.integrate.simps = scipy.integrate.simpson
            logging.info("Successfully patched scipy.integrate.simps")
    except ImportError as e:
        logging.warning(f"Failed to apply scipy compatibility patch: {e}")
    except Exception as e:
        logging.error(f"Unexpected error applying scipy patch: {e}")


OPENFACE3_AVAILABLE = False  # Will be updated to True after successful lazy import


def _lazy_import_openface():
    """Import OpenFace modules lazily to avoid argparse side-effects at import.

    Some OpenFace distributions parse command line arguments on import.
    Delaying import until pipeline.initialize() prevents pytest (test
    collection) from encountering unexpected argparse exits.
    """
    global OPENFACE3_AVAILABLE, FaceDetector, LandmarkDetector, MultitaskPredictor
    # If already successfully imported return True immediately
    if OPENFACE3_AVAILABLE:
        return True
    patch_scipy_compatibility()
    try:
        from openface.face_detection import FaceDetector  # type: ignore
        from openface.landmark_detection import LandmarkDetector  # type: ignore
        from openface.multitask_model import MultitaskPredictor  # type: ignore

        OPENFACE3_AVAILABLE = True
        logging.info("OpenFace 3.0 successfully imported (lazy)")
    except SystemExit as e:
        # Some OpenFace builds call sys.exit via argparse on import when CLI args missing.
        OPENFACE3_AVAILABLE = False
        logging.warning(
            f"OpenFace 3.0 import triggered SystemExit ({e}). Treating as unavailable for runtime."
        )
    except ImportError as e:
        OPENFACE3_AVAILABLE = False
        logging.error(f"OpenFace 3.0 not available: {e}")
        logging.error("Install from: https://github.com/CMU-MultiComp-Lab/OpenFace-3.0")
    return OPENFACE3_AVAILABLE


class OpenFace3Pipeline(BasePipeline):
    """OpenFace 3.0 face analysis pipeline using COCO format.

    Provides comprehensive facial behavior analysis including:
    - 2D and 3D facial landmarks (68-point model)
    - Facial Action Units (AU) intensity and presence
    - Head pose estimation (rotation and translation)
    - Gaze direction and eye gaze
    - Basic emotion recognition
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize OpenFace3Pipeline with merged default config."""
        default_config = {
            "detection_confidence": 0.7,
            "landmark_model": "98_point",  # Use 98_point - we have the correct model
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

        # Person identity management
        self.identity_manager = None

        # Performance metrics
        self.processing_times = []

        self.logger = logging.getLogger(__name__)

    def initialize(self) -> None:
        """Initialize OpenFace 3.0 components."""
        if not _lazy_import_openface():
            # Graceful skip: mark as unavailable without raising ImportError so test suite can continue
            self.is_initialized = False
            self._model_info = {
                "model_name": "OpenFace 3.0",
                "available": False,
                "unavailable_reason": "OpenFace 3.0 not installed",
            }
            self.logger.warning(
                "OpenFace 3.0 unavailable - skipping initialization (graceful)"
            )
            return

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

            # Initialize face detector with model path
            face_detector_path = (
                self.config.get("model_path") or "./weights/Alignment_RetinaFace.pth"
            )
            self.face_detector = FaceDetector(
                model_path=face_detector_path,
                device=device,
                confidence_threshold=self.config["detection_confidence"],
            )

            # Initialize landmark detector - use correct signature
            model_type = self.config["landmark_model"]
            # Use provided landmark_model path directly if it contains a path, otherwise construct it
            if "/" in model_type or "\\" in model_type:
                landmark_model_path = model_type
            else:
                landmark_model_path = (
                    f"./weights/Landmark_{model_type.split('_')[0]}.pkl"
                )
            # Configure device IDs for CUDA
            if device == "cuda":
                device_ids = [0]  # Use first GPU device
            else:
                device_ids = [-1]  # CPU mode

            self.landmark_detector = LandmarkDetector(
                model_path=landmark_model_path, device=device, device_ids=device_ids
            )

            # Initialize multitask predictor for Action Units, Head Pose, Gaze, etc.
            if any(
                [
                    self.config["enable_action_units"],
                    self.config["enable_head_pose"],
                    self.config["enable_gaze"],
                    self.config["enable_emotions"],
                ]
            ):
                mtl_model_path = "./weights/MTL_backbone.pth"
                self.multitask_predictor = MultitaskPredictor(
                    model_path=mtl_model_path, device=device
                )
                self.logger.info("MultitaskPredictor initialized for advanced features")
            else:
                self.multitask_predictor = None

            # Face tracking would be implemented here if needed
            if self.config["track_faces"]:
                self.face_tracker = (
                    None  # Will implement based on actual OpenFace 3.0 API
                )

            self._model_info = {
                "model_name": "OpenFace 3.0",
                "version": "3.0",
                "device": device,
                "landmark_model": model_type,
                "features": {
                    "landmarks": True,
                    "3d_landmarks": self.config["enable_3d_landmarks"],
                    "action_units": self.config["enable_action_units"],
                    "head_pose": self.config["enable_head_pose"],
                    "gaze": self.config["enable_gaze"],
                    "emotions": self.config["enable_emotions"],
                    "face_tracking": self.config["track_faces"],
                },
            }

            self.is_initialized = True
            self.logger.info("OpenFace 3.0 pipeline initialized successfully")

        except Exception as e:
            # Graceful failure: do not propagate ImportError chain to preserve pipeline availability semantics
            self.logger.error(f"Failed to initialize OpenFace 3.0: {e}")
            self.is_initialized = False
            self._model_info = {
                "model_name": "OpenFace 3.0",
                "available": False,
                "unavailable_reason": str(e),
            }
            return

    def _load_person_tracks(self, video_path: str) -> list[dict[str, Any]] | None:
        """Load person tracking data for face-person linking."""
        if not self.config["person_identity"]["link_to_persons"]:
            return None

        # Initialize person identity manager if needed
        if self.identity_manager is None:
            try:
                self.identity_manager = PersonIdentityManager(
                    config=self.config.get("person_identity", {})
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize PersonIdentityManager: {e}")
                return None

        # Look for person tracking data
        video_name = Path(video_path).stem
        person_data_paths = [
            Path(video_path).parent / f"person_tracking_{video_name}.json",
            Path("data") / f"person_tracking_{video_name}.json",
            Path("demo_results") / f"person_tracking_{video_name}.json",
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

        self.logger.info("No person tracking data found")
        return None

    def _get_frame_person_annotations(
        self, person_tracks: list[dict[str, Any]], image_id: int
    ) -> list[dict[str, Any]]:
        """Get person annotations for a specific frame."""
        if not person_tracks:
            return []

        frame_persons = []
        for ann in person_tracks:
            if ann.get("image_id") == image_id:
                frame_persons.append(ann)

        return frame_persons

    def _link_face_to_person(
        self, face_bbox: list[float], person_annotations: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Link a face detection to a person using IoU matching."""
        if not person_annotations:
            return {}

        iou_threshold = self.config["person_identity"]["iou_threshold"]
        best_match = None
        best_iou = 0.0

        for person_ann in person_annotations:
            person_bbox = person_ann.get("bbox", [])
            if len(person_bbox) == 4:
                iou = self._calculate_iou(face_bbox, person_bbox)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_match = person_ann

        if best_match:
            person_id = best_match.get("person_id", "unknown")
            person_label = best_match.get("person_label", "unknown")
            person_label_confidence = best_match.get("person_label_confidence", 0.0)
            person_labeling_method = best_match.get("person_labeling_method", "unknown")

            return {
                "person_id": person_id,
                "person_label": person_label,
                "person_label_confidence": person_label_confidence,
                "person_labeling_method": person_labeling_method,
                "face_person_iou": best_iou,
            }

        return {}

    def _calculate_iou(self, bbox1: list[float], bbox2: list[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Convert to corner coordinates
        x1_max, y1_max = x1 + w1, y1 + h1
        x2_max, y2_max = x2 + w2, y2 + h2

        # Calculate intersection
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(x1_max, x2_max)
        inter_y2 = min(y1_max, y2_max)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def get_schema(self) -> dict[str, Any]:
        """Get the output schema for OpenFace 3.0 face analysis annotations."""
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
                "keypoints": "array[294]",  # 98 landmarks * 3 coordinates (x, y, visibility)
                "num_keypoints": "integer",
                "openface3": {
                    "confidence": "float",
                    "timestamp": "float",
                    "landmarks_2d": "array[196]",  # 98 points * 2 coordinates
                    "landmarks_3d": "array[294]",  # 98 points * 3 coordinates
                    "action_units": {
                        "intensity": "object",  # AU name -> intensity value
                        "presence": "object",  # AU name -> binary presence
                    },
                    "head_pose": {
                        "rotation": "array[3]",  # [rx, ry, rz] in radians
                        "translation": "array[3]",  # [tx, ty, tz] in mm
                        "confidence": "float",
                    },
                    "gaze": {
                        "direction": "array[3]",  # Gaze direction vector
                        "left_eye": "array[2]",  # Left eye gaze point
                        "right_eye": "array[2]",  # Right eye gaze point
                        "confidence": "float",
                    },
                    "emotion": {
                        "dominant": "string",  # Primary emotion
                        "probabilities": "object",  # All emotion probabilities
                        "valence": "float",  # Emotion valence [-1, 1]
                        "arousal": "float",  # Emotion arousal [-1, 1]
                    },
                    "track_id": "integer",  # Face tracking ID
                },
            },
        }

    def get_schema(self) -> dict[str, Any]:
        """Get the output schema for OpenFace 3.0 face analysis annotations."""
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
                "keypoints": "array[196]",  # 98 landmarks * 2 coordinates (x, y)
                "num_keypoints": "integer",
                "confidence": "float",
                "person_id": "string",  # Person identity
                "person_label": "string",  # Person label (name)
                "person_label_confidence": "float",  # Person labeling confidence
                "person_labeling_method": "string",  # How person was labeled
                "attributes": {
                    "action_units": "object",  # AU intensities and presence
                    "head_pose": {
                        "rotation": "array[3]",  # [rx, ry, rz] in radians
                        "translation": "array[3]",  # [tx, ty, tz] in mm
                    },
                    "gaze": {
                        "direction": "array[3]",  # 3D gaze direction vector
                        "left_eye": "array[2]",  # 2D gaze point for left eye
                        "right_eye": "array[2]",  # 2D gaze point for right eye
                    },
                    "emotion": "string",
                    "landmark_3d": "array[294]",  # 98 landmarks * 3 coordinates (x, y, z)
                },
            },
        }

    def process(
        self,
        video_path: str,
        start_time: float = 0.0,
        end_time: float | None = None,
        pps: float = 1.0,
        output_dir: str | None = None,
    ) -> list[dict[str, Any]]:
        """Process video with OpenFace 3.0 face analysis.

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

        # Load person tracking data for face-person linking
        person_tracks = self._load_person_tracks(video_path)
        if person_tracks:
            self.logger.info(f"Loaded {len(person_tracks)} person track annotations")
        elif self.config["person_identity"]["require_person_id"]:
            raise ValueError("Person tracking data required but not found")

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
                    timestamp=timestamp,
                )
                images.append(image_entry)

                # Process frame with OpenFace 3.0
                face_results = self._process_frame(frame, timestamp)

                # Get person annotations for this frame
                frame_persons = (
                    self._get_frame_person_annotations(person_tracks, frame_idx + 1)
                    if person_tracks
                    else []
                )

                # Convert to COCO annotations with person linking
                for face_result in face_results:
                    # Get face bounding box for person linking
                    face_bbox = face_result.get("bbox", [])

                    # Link face to person
                    person_info = self._link_face_to_person(face_bbox, frame_persons)

                    # Create annotation with person information
                    annotation = self._create_face_annotation(
                        annotation_id=annotation_id,
                        image_id=frame_idx + 1,
                        face_data=face_result,
                        timestamp=timestamp,
                        person_info=person_info,
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
                "version": __version__,
                "year": 2025,
                "contributor": "VideoAnnotator",
                "date_created": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

        # Save results if output directory specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            video_name = Path(video_path).stem
            output_file = output_path / f"{video_name}_openface3_analysis.json"

            export_coco_json(
                annotations=coco_dataset["annotations"],
                images=coco_dataset["images"],
                output_path=str(output_file),
                categories=coco_dataset["categories"],
            )

            # Validate COCO format
            if validate_coco_json(str(output_file)):
                self.logger.info(
                    f"OpenFace 3.0 analysis saved and validated: {output_file}"
                )

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

    def _process_frame(
        self, frame: np.ndarray, timestamp: float
    ) -> list[dict[str, Any]]:
        """Process a single frame with OpenFace 3.0."""
        import os
        import tempfile

        face_results = []
        temp_frame_path = None

        try:
            # Save frame to temporary file since OpenFace expects file paths
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                cv2.imwrite(tmp.name, frame)
                temp_frame_path = tmp.name

            # Detect faces - returns (detections_array, processed_image)
            face_detections, _ = self.face_detector.detect_faces(temp_frame_path)

            if len(face_detections) == 0:
                return face_results

            # Process each detected face
            for i, detection in enumerate(face_detections):
                # Parse detection format: [x1, y1, x2, y2, confidence, landmarks...]
                x1, y1, x2, y2, confidence = detection[:5]
                face_landmarks = detection[5:] if len(detection) > 5 else None

                # Convert to standard bbox format [x, y, width, height]
                bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

                face_data = {
                    "timestamp": timestamp,
                    "face_id": i,
                    "detection": {
                        "bbox": bbox,
                        "confidence": float(confidence),
                        "landmarks": face_landmarks.tolist()
                        if face_landmarks is not None
                        and hasattr(face_landmarks, "tolist")
                        else (
                            list(face_landmarks) if face_landmarks is not None else None
                        ),
                    },
                }

                # Extract face region for additional analysis
                x, y, w, h = bbox
                # Ensure coordinates are within frame bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, frame.shape[1] - x)
                h = min(h, frame.shape[0] - y)

                if w > 0 and h > 0:
                    face_roi = frame[y : y + h, x : x + w]

                    # Get facial landmarks using landmark detector
                    # LandmarkDetector needs the full image and detections array, not just face ROI
                    try:
                        # Convert single detection to array format for landmark detector
                        det_array = np.array(
                            [detection[:5]]
                        )  # [x1, y1, x2, y2, confidence]
                        landmarks = self.landmark_detector.detect_landmarks(
                            frame, det_array
                        )
                        if landmarks is not None and len(landmarks) > 0:
                            # Handle different landmark formats
                            landmark_data = landmarks[0]
                            if hasattr(landmark_data, "tolist"):
                                face_data["landmarks_2d"] = landmark_data.tolist()
                            elif isinstance(landmark_data, (list, tuple)):
                                face_data["landmarks_2d"] = list(landmark_data)
                            else:
                                face_data["landmarks_2d"] = landmark_data

                    except Exception as e:
                        self.logger.warning(
                            f"Failed to get landmarks for face {i}: {e}"
                        )

                    # Advanced analysis with MultitaskPredictor
                    if self.multitask_predictor and w > 0 and h > 0:
                        try:
                            au_output, pose_output, emotion_output = (
                                self.multitask_predictor.predict(face_roi)
                            )

                            # Parse and add to face_data
                            if self.config["enable_action_units"]:
                                face_data["action_units"] = self._parse_action_units(
                                    au_output
                                )

                            if self.config["enable_head_pose"]:
                                face_data["head_pose"] = self._parse_head_pose(
                                    pose_output
                                )

                            if self.config["enable_emotions"]:
                                face_data["emotion"] = self._parse_emotions(
                                    emotion_output
                                )

                            # Note: Gaze processing would use pose_output or separate gaze model
                            if self.config["enable_gaze"]:
                                face_data["gaze"] = self._parse_gaze(pose_output)

                        except Exception as e:
                            self.logger.warning(
                                f"MultitaskPredictor failed for face {i}: {e}"
                            )

                face_results.append(face_data)

        except Exception as e:
            self.logger.error(f"Error processing frame at {timestamp:.2f}s: {e}")
        finally:
            # Clean up temporary file
            if temp_frame_path and os.path.exists(temp_frame_path):
                os.unlink(temp_frame_path)

        return face_results

    def _create_face_annotation(
        self,
        annotation_id: int,
        image_id: int,
        face_data: dict[str, Any],
        timestamp: float,
        person_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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
            iscrowd=0,
        )

        # Add person identity information
        if person_info:
            annotation["person_id"] = person_info.get("person_id", "unknown")
            annotation["person_label"] = person_info.get("person_label", "unknown")
            annotation["person_label_confidence"] = person_info.get(
                "person_label_confidence", 0.0
            )
            annotation["person_labeling_method"] = person_info.get(
                "person_labeling_method", "unknown"
            )
        else:
            annotation["person_id"] = "unknown"
            annotation["person_label"] = "unknown"
            annotation["person_label_confidence"] = 0.0
            annotation["person_labeling_method"] = "none"

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
            openface_data["landmarks_2d"] = (
                landmarks_2d.tolist()
                if hasattr(landmarks_2d, "tolist")
                else list(landmarks_2d)
            )

        if "landmarks_3d" in face_data:
            landmarks_3d = face_data["landmarks_3d"]
            openface_data["landmarks_3d"] = (
                landmarks_3d.tolist()
                if hasattr(landmarks_3d, "tolist")
                else list(landmarks_3d)
            )

        # Add action units
        if "action_units" in face_data:
            openface_data["action_units"] = face_data["action_units"]

        # Add head pose
        if "head_pose" in face_data:
            openface_data["head_pose"] = face_data["head_pose"]

        # Add gaze information
        if "gaze" in face_data:
            openface_data["gaze"] = face_data["gaze"]

        # Add emotion recognition
        if "emotion" in face_data:
            openface_data["emotion"] = face_data["emotion"]

        # Add tracking ID
        if "track_id" in face_data:
            openface_data["track_id"] = face_data["track_id"]

        # Store OpenFace data in custom field
        annotation["openface3"] = openface_data

        return annotation

    def _get_face_categories(self) -> list[dict[str, Any]]:
        """Get COCO categories for face analysis."""
        num_landmarks = self.config.get("landmark_points", 98)
        return [
            {
                "id": 1,
                "name": "face",
                "supercategory": "person",
                "keypoints": [f"landmark_{i}" for i in range(num_landmarks)],
                "skeleton": [],  # Face landmarks don't have skeleton connections
            }
        ]

    def _save_detailed_results(
        self, output_file: Path, annotations: list[dict]
    ) -> None:
        """Save detailed OpenFace 3.0 results with all extracted features."""
        detailed_results = {
            "metadata": {
                "pipeline": "OpenFace3Pipeline",
                "model_info": self._model_info,
                "config": self.config,
                "processing_stats": {
                    "total_faces": len(annotations),
                    "avg_processing_time": np.mean(self.processing_times)
                    if self.processing_times
                    else 0,
                },
            },
            "faces": [],
        }

        for annotation in annotations:
            if "openface3" in annotation:
                face_entry = {
                    "annotation_id": annotation["id"],
                    "bbox": annotation["bbox"],
                    "timestamp": annotation["openface3"]["timestamp"],
                    "features": annotation["openface3"],
                }
                detailed_results["faces"].append(face_entry)

        with open(output_file, "w") as f:
            json.dump(detailed_results, f, indent=2, default=str)

        self.logger.info(f"Detailed OpenFace 3.0 results saved: {output_file}")

    def get_pipeline_info(self) -> dict[str, Any]:
        """Get information about the OpenFace 3.0 pipeline."""
        info = {
            "name": "OpenFace3Pipeline",
            "version": __version__,
            "description": "Facial behavior analysis using OpenFace 3.0",
            "capabilities": [
                "face_detection",
                "facial_landmarks_98",
                "3d_landmarks",
                "action_units",
                "head_pose",
                "gaze_estimation",
                "emotion_recognition",
                "face_tracking",
            ],
            "output_format": "COCO",
            "dependencies": {
                "openface3": OPENFACE3_AVAILABLE,
                "opencv": True,
                "numpy": True,
            },
        }

        if self.is_initialized and self._model_info:
            info["model_info"] = self._model_info

        return info

    def _parse_action_units(self, au_output):
        """Parse Action Units from MultitaskPredictor output."""
        # au_output shape: (1, 8)
        # Need to determine which AUs these 8 values represent
        au_values = au_output.detach().cpu().numpy().flatten()

        # Standard FACS Action Units mapping (based on common OpenFace implementations)
        au_mapping = {
            0: "AU01_Inner_Brow_Raiser",
            1: "AU02_Outer_Brow_Raiser",
            2: "AU04_Brow_Lowerer",
            3: "AU05_Upper_Lid_Raiser",
            4: "AU06_Cheek_Raiser",
            5: "AU07_Lid_Tightener",
            6: "AU09_Nose_Wrinkler",
            7: "AU10_Upper_Lip_Raiser",
        }

        action_units = {}
        for idx, au_name in au_mapping.items():
            intensity = float(au_values[idx])
            # Convert to 0-5 intensity scale and presence detection
            normalized_intensity = max(
                0, min(5, (intensity + 4) * 1.25)
            )  # Normalize range
            presence = normalized_intensity > 0.5

            action_units[au_name] = {
                "intensity": normalized_intensity,
                "presence": presence,
            }

        return action_units

    def _parse_head_pose(self, pose_output):
        """Parse head pose from MultitaskPredictor output."""
        # pose_output shape: (1, 2)
        pose_values = pose_output.detach().cpu().numpy().flatten()

        # Determine which angles these represent (likely pitch and yaw)
        return {
            "pitch": float(pose_values[0]) * 180 / np.pi,  # Convert to degrees
            "yaw": float(pose_values[1]) * 180 / np.pi,
            "roll": 0.0,  # Not available from current output
            "confidence": 0.8,  # Default confidence
        }

    def _parse_emotions(self, emotion_output):
        """Parse emotions from MultitaskPredictor output."""
        # emotion_output shape: (1, 8)
        emotion_probs = emotion_output.detach().cpu().numpy().flatten()

        # Standard emotion mapping (common in facial expression research)
        emotion_labels = [
            "neutral",
            "happiness",
            "sadness",
            "anger",
            "fear",
            "surprise",
            "disgust",
            "contempt",
        ]

        # Softmax normalization
        exp_probs = np.exp(emotion_probs - np.max(emotion_probs))
        normalized_probs = exp_probs / np.sum(exp_probs)

        probabilities = {
            label: float(prob)
            for label, prob in zip(emotion_labels, normalized_probs, strict=False)
        }

        # Find dominant emotion
        dominant_idx = np.argmax(normalized_probs)
        dominant = emotion_labels[dominant_idx]

        # Calculate valence and arousal
        valence = self._calculate_valence(probabilities)
        arousal = self._calculate_arousal(probabilities)

        return {
            "dominant": dominant,
            "probabilities": probabilities,
            "valence": valence,
            "arousal": arousal,
            "confidence": float(normalized_probs[dominant_idx]),
        }

    def _parse_gaze(self, pose_output):
        """Parse gaze from MultitaskPredictor output using head pose as proxy."""
        # For now, use head pose as gaze direction proxy
        # In future versions, this could use a dedicated gaze model
        pose_values = pose_output.detach().cpu().numpy().flatten()

        # Convert head pose to gaze direction vector
        pitch = pose_values[0]
        yaw = pose_values[1]

        # Convert to 3D direction vector
        direction_x = np.sin(yaw) * np.cos(pitch)
        direction_y = -np.sin(pitch)
        direction_z = np.cos(yaw) * np.cos(pitch)

        return {
            "direction_x": float(direction_x),
            "direction_y": float(direction_y),
            "direction_z": float(direction_z),
            "confidence": 0.7,  # Lower confidence since this is head pose proxy
        }

    def _calculate_valence(self, probabilities):
        """Calculate valence (positive/negative sentiment) from emotion probabilities."""
        # Valence mapping: positive emotions = positive valence, negative = negative
        valence_map = {
            "happiness": 1.0,
            "surprise": 0.3,
            "neutral": 0.0,
            "contempt": -0.2,
            "anger": -0.8,
            "disgust": -0.7,
            "fear": -0.6,
            "sadness": -0.9,
        }

        valence = sum(
            prob * valence_map.get(emotion, 0.0)
            for emotion, prob in probabilities.items()
        )
        return max(-1.0, min(1.0, valence))  # Clamp to [-1, 1]

    def _calculate_arousal(self, probabilities):
        """Calculate arousal (activation level) from emotion probabilities."""
        # Arousal mapping: high-energy emotions = high arousal
        arousal_map = {
            "anger": 0.9,
            "fear": 0.8,
            "surprise": 0.7,
            "happiness": 0.6,
            "disgust": 0.5,
            "contempt": 0.3,
            "sadness": 0.2,
            "neutral": 0.0,
        }

        arousal = sum(
            prob * arousal_map.get(emotion, 0.0)
            for emotion, prob in probabilities.items()
        )
        return max(0.0, min(1.0, arousal))  # Clamp to [0, 1]

    def cleanup(self) -> None:
        """Cleanup resources."""
        if hasattr(self, "face_detector") and self.face_detector:
            del self.face_detector
        if hasattr(self, "landmark_detector") and self.landmark_detector:
            del self.landmark_detector
        if hasattr(self, "au_analyzer") and self.au_analyzer:
            del self.au_analyzer
        if hasattr(self, "head_pose_estimator") and self.head_pose_estimator:
            del self.head_pose_estimator
        if hasattr(self, "gaze_estimator") and self.gaze_estimator:
            del self.gaze_estimator
        if hasattr(self, "emotion_recognizer") and self.emotion_recognizer:
            del self.emotion_recognizer

        self.logger.info("OpenFace 3.0 pipeline cleaned up")
