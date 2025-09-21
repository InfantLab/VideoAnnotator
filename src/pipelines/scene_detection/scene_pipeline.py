"""
Standards-only scene detection and classification pipeline.

This pipeline uses PySceneDetect for shot boundary detection and CLIP for scene classification,
outputting native COCO format annotations with full standards compliance.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from pathlib import Path
import logging

from pipelines.base_pipeline import BasePipeline
from version import __version__
from exporters.native_formats import (
    create_coco_annotation,
    create_coco_image_entry,
    export_coco_json,
    validate_coco_json,
)

# Optional imports
try:
    from scenedetect import detect, ContentDetector

    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False

try:
    import clip
    import torch

    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


class SceneDetectionPipeline(BasePipeline):
    """
    Standards-only scene detection and classification pipeline.

    Uses PySceneDetect for shot boundary detection and CLIP for scene classification.
    Returns native COCO format annotations.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "threshold": 30.0,  # Scene detection threshold
            "min_scene_length": 2.0,  # Minimum scene length in seconds
            "scene_prompts": [
                "living room",
                "kitchen",
                "bedroom",
                "outdoor",
                "clinic",
                "nursery",
                "office",
                "playground",
            ],
            "clip_model": "ViT-B/32",
            "use_gpu": True,
            "keyframe_extraction": "middle",  # Extract keyframe from middle of scene
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)

        self.logger = logging.getLogger(__name__)
        self.clip_model = None
        self.clip_preprocess = None
        self.device = None

    def process(
        self,
        video_path: str,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        pps: float = 0.0,  # Not used for scene detection
        output_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Process video for scene detection and classification."""

        # Get video metadata
        video_metadata = self._get_video_metadata(video_path)

        # Step 1: Scene segmentation using PySceneDetect
        scene_segments = self._detect_scene_boundaries(video_path, start_time, end_time)

        # Step 2: Scene classification using CLIP (if available)
        if CLIP_AVAILABLE:
            classified_segments = self._classify_scenes(video_path, scene_segments)
        else:
            self.logger.warning("CLIP not available, skipping scene classification")
            classified_segments = scene_segments

        # Step 3: Create COCO-format annotations
        annotations = []

        for i, segment in enumerate(classified_segments):
            # Calculate middle frame for image_id
            mid_time = (segment["start"] + segment["end"]) / 2
            frame_number = int(mid_time * video_metadata["fps"])

            # Create COCO annotation for scene
            scene_annotation = create_coco_annotation(
                annotation_id=i + 1,
                image_id=f"{video_metadata['video_id']}_frame_{frame_number:06d}",
                category_id=1,  # Scene category
                bbox=[0, 0, video_metadata["width"], video_metadata["height"]],  # Full frame
                score=segment.get("confidence", 1.0),
                # VideoAnnotator extensions for scenes
                video_id=video_metadata["video_id"],
                timestamp=mid_time,
                start_time=segment["start"],
                end_time=segment["end"],
                duration=segment["end"] - segment["start"],
                scene_type=segment.get("classification", "unknown"),
                frame_start=int(segment["start"] * video_metadata["fps"]),
                frame_end=int(segment["end"] * video_metadata["fps"]),
                all_scores=segment.get("all_scores", {}),
            )
            annotations.append(scene_annotation)

        # Save results if output directory specified
        if output_dir and annotations:
            self._save_coco_annotations(annotations, output_dir, video_metadata)

        self.logger.info(f"Scene detection complete: {len(annotations)} scenes detected")
        return annotations

    def _get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract video metadata."""
        import cv2

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

    def _detect_scene_boundaries(
        self, video_path: str, start_time: float, end_time: Optional[float]
    ) -> List[Dict[str, float]]:
        """Detect scene boundaries using PySceneDetect."""

        if not SCENEDETECT_AVAILABLE:
            self.logger.warning(
                "PySceneDetect not available, creating single scene for entire video"
            )
            video_metadata = self._get_video_metadata(video_path)
            end_time = end_time or video_metadata["duration"]
            return [{"start": start_time, "end": end_time}]

        try:
            # Get video duration if end_time not specified
            if end_time is None:
                video_metadata = self._get_video_metadata(video_path)
                end_time = video_metadata["duration"]

            # Detect scenes
            scene_list = detect(
                video_path,
                ContentDetector(threshold=self.config["threshold"]),
                start_time=start_time,
                end_time=end_time,
            )

            # Convert to our format
            segments = []
            for scene in scene_list:
                start_sec = scene[0].get_seconds()
                end_sec = scene[1].get_seconds()

                # Filter by minimum scene length
                if end_sec - start_sec >= self.config["min_scene_length"]:
                    segments.append({"start": start_sec, "end": end_sec})

            # If no scenes detected, create a single scene for the entire video
            if not segments:
                self.logger.info(
                    f"No scene changes detected. Creating single scene ({start_time:.2f}s - {end_time:.2f}s)"
                )
                segments.append({"start": start_time, "end": end_time})

            self.logger.info(f"Detected {len(segments)} scene segments")
            return segments

        except Exception as e:
            self.logger.error(f"Scene detection failed: {e}")
            # Fallback to single scene
            return [{"start": start_time, "end": end_time or 0.0}]

    def _classify_scenes(
        self, video_path: str, segments: List[Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """Classify scenes using CLIP."""

        if not CLIP_AVAILABLE:
            return segments

        try:
            import cv2
            from PIL import Image

            # Initialize CLIP if not already loaded
            if self.clip_model is None:
                self._initialize_clip()

            # Prepare text prompts
            text_prompts = [f"a {prompt}" for prompt in self.config["scene_prompts"]]
            text = clip.tokenize(text_prompts).to(self.device)

            classified_segments = []
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)

            try:
                for segment in segments:
                    # Extract keyframe from middle of segment
                    mid_time = (segment["start"] + segment["end"]) / 2
                    frame_number = int(mid_time * fps)

                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    ret, frame = cap.read()

                    if ret:
                        # Convert BGR to RGB and create PIL Image
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(frame_rgb)

                        # Process with CLIP
                        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)

                        with torch.no_grad():
                            logits_per_image, logits_per_text = self.clip_model(image_input, text)
                            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

                        # Get best classification
                        best_idx = np.argmax(probs)
                        best_prob = probs[best_idx]
                        best_class = self.config["scene_prompts"][best_idx]

                        segment["classification"] = best_class
                        segment["confidence"] = float(best_prob)
                        segment["all_scores"] = {
                            prompt: float(prob)
                            for prompt, prob in zip(self.config["scene_prompts"], probs)
                        }
                    else:
                        # Default classification if frame extraction fails
                        segment["classification"] = "unknown"
                        segment["confidence"] = 0.0
                        segment["all_scores"] = {}

                    classified_segments.append(segment)

            finally:
                cap.release()

            self.logger.info(
                f"Scene classification complete for {len(classified_segments)} segments"
            )
            return classified_segments

        except Exception as e:
            self.logger.error(f"Scene classification failed: {e}")
            return segments

    def _initialize_clip(self):
        """Initialize CLIP model."""
        if not CLIP_AVAILABLE:
            raise ImportError("CLIP not available for scene classification")

        self.device = "cuda" if self.config["use_gpu"] and torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load(
            self.config["clip_model"], device=self.device
        )
        self.logger.info(f"CLIP model loaded: {self.config['clip_model']} on {self.device}")

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
                frame_number = ann.get("frame_start", 0)
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

        # Export COCO JSON with scene categories
        categories = [
            {
                "id": 1,
                "name": "scene",
                "supercategory": "video_segment",
                "description": "Video scene segment with temporal bounds",
            }
        ]

        coco_path = output_path / f"{video_metadata['video_id']}_scene_detection.json"
        export_coco_json(annotations, images, str(coco_path), categories)

        # Validate COCO format
        validation_result = validate_coco_json(str(coco_path), "scene_detection")
        if validation_result.is_valid:
            self.logger.info(f"Scene detection COCO validation successful: {coco_path}")
        else:
            self.logger.warning(
                f"Scene detection COCO validation warnings: {', '.join(validation_result.warnings)}"
            )

    def initialize(self) -> None:
        """Initialize the pipeline."""
        self.logger.info("Initializing Scene Detection Pipeline")

        # Check PySceneDetect availability
        if SCENEDETECT_AVAILABLE:
            import scenedetect

            self.logger.info(f"PySceneDetect available: {scenedetect.__version__}")
        else:
            self.logger.warning("PySceneDetect not available - scene detection will be limited")

        # Check CLIP availability
        if CLIP_AVAILABLE:
            self.logger.info("CLIP available for scene classification")
        else:
            self.logger.warning("CLIP not available - scene classification disabled")

        # Check OpenCV
        try:
            import cv2
            cv_version = getattr(cv2, "__version__", "unknown")
            self.logger.info(f"OpenCV available: {cv_version}")
        except ImportError:
            self.logger.warning("OpenCV not available - video processing will be limited")

        self.is_initialized = True
        self.logger.info("Scene Detection Pipeline initialized successfully")

    def cleanup(self) -> None:
        """Clean up pipeline resources."""
        if self.clip_model is not None:
            # Move to CPU to free GPU memory
            if hasattr(self.clip_model, "to") and self.device == "cuda":
                self.clip_model.to(torch.device("cpu"))

        self.clip_model = None
        self.clip_preprocess = None
        self.device = None
        self.is_initialized = False
        self.logger.info("Scene Detection Pipeline cleaned up")

    def get_schema(self) -> Dict[str, Any]:
        """Return JSON schema for scene annotations."""
        return {
            "type": "scene_detection",
            "description": "Scene detection and classification results",
            "properties": {
                "scenes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "scene_id": {"type": "string"},
                            "start_time": {"type": "number"},
                            "end_time": {"type": "number"},
                            "duration": {"type": "number"},
                            "scene_type": {"type": "string"},
                            "confidence": {"type": "number"},
                            "timestamp": {"type": "number"},
                            "frame_start": {"type": "integer"},
                            "frame_end": {"type": "integer"},
                        },
                    },
                    "description": "Detected scene segments with classifications",
                },
                "total_scenes": {
                    "type": "integer",
                    "description": "Total number of detected scenes",
                },
            },
        }

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the scene detection pipeline."""
        return {
            "name": "SceneDetectionPipeline",
            "version": __version__,
            "capabilities": {
                "scene_detection": SCENEDETECT_AVAILABLE,
                "scene_classification": CLIP_AVAILABLE,
            },
            "models": {
                "scene_detector": (
                    "PySceneDetect ContentDetector" if SCENEDETECT_AVAILABLE else None
                ),
                "scene_classifier": self.config["clip_model"] if CLIP_AVAILABLE else None,
            },
            "config": {
                "threshold": self.config["threshold"],
                "min_scene_length": self.config["min_scene_length"],
                "scene_prompts": self.config["scene_prompts"],
                "use_gpu": self.config["use_gpu"],
            },
            "requirements": {
                "scenedetect_available": SCENEDETECT_AVAILABLE,
                "clip_available": CLIP_AVAILABLE,
                "cuda_available": torch.cuda.is_available() if CLIP_AVAILABLE else False,
            },
        }
