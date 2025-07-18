import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoProcessor, AutoModel
from huggingface_hub import hf_hub_download  # new import

from ..base_pipeline import BasePipeline
from .face_pipeline import FaceAnalysisPipeline
from ...exporters.native_formats import create_coco_image_entry, export_coco_json

# List of emotion categories based on LAION taxonomy with correct file mappings
EMOTION_LABELS = {
    # Positive High-Energy
    "elation": "model_elation_best.pth",
    "amusement": "model_amusement_best.pth", 
    "pleasure_ecstasy": "model_pleasure_ecstasy_best.pth",
    "astonishment_surprise": "model_astonishment_surprise_best.pth",
    "hope_enthusiasm_optimism": "model_hope_enthusiasm_optimism_best.pth",
    "triumph": "model_triumph_best.pth",
    "awe": "model_awe_best.pth",
    "teasing": "model_teasing_best.pth",
    "interest": "model_interest_best.pth",
    # Positive Low-Energy
    "relief": "model_relief_best.pth",
    "contentment": "model_contentment_best.pth",
    "contemplation": "model_contemplation_best.pth",
    "pride": "model_pride_best.pth",
    "thankfulness_gratitude": "model_thankfulness_gratitude_best.pth",
    "affection": "model_affection_best.pth",
    # Negative High-Energy
    "anger": "model_anger_best.pth",
    "fear": "model_fear_best.pth",
    "distress": "model_distress_best.pth",
    "impatience_irritability": "model_impatience_and_irritability_best.pth",
    "disgust": "model_disgust_best.pth",
    "malevolence_malice": "model_malevolence_malice_best.pth",
    # Negative Low-Energy
    "helplessness": "model_helplessness_best.pth",
    "sadness": "model_sadness_best.pth",
    "emotional_numbness": "model_emotional_numbness_best.pth",
    "jealousy_envy": "model_jealousy_&_envy_best.pth",
    "embarrassment": "model_embarrassment_best.pth",
    "contempt": "model_contempt_best.pth",
    "shame": "model_shame_best.pth",
    "disappointment": "model_disappointment_best.pth",
    "doubt": "model_doubt_best.pth",
    "bitterness": "model_bitterness_best.pth",
    # Cognitive States
    "concentration": "model_concentration_best.pth",
    "confusion": "model_confusion_best.pth",
    # Physical States
    "fatigue_exhaustion": "model_fatigue_exhaustion_best.pth",
    "pain": "model_pain_best.pth",
    "sourness": "model_sourness_best.pth",
    "intoxication_altered_states": "model_intoxication_altered_states_of_consciousness_best.pth",
    # Longing & Lust
    "sexual_lust": "model_sexual_lust_best.pth",
    "longing": "model_longing_best.pth",
    "infatuation": "model_infatuation_best.pth",
    # Extra Dimensions
    "dominance": "model_dominance_best.pth",
    "arousal": "model_arousal_best.pth",
    "emotional_vulnerability": "model_emotional_vulnerability_best.pth"
}


class LAIONFacePipeline(BasePipeline):
    """
    LAION Empathic Insight Face Pipeline integrating SigLIP encoder and emotion classifiers.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            # Model configuration
            "model_size": "small",  # "small" or "large"
            "model_cache_dir": "./models/laion_face",

            # Processing configuration
            "confidence_threshold": 0.7,
            "min_face_size": 30,
            "max_faces": 10,
            "face_detection_backend": "opencv",  # Reuse existing detection backend

            # Output configuration
            "include_raw_scores": False,
            "include_normalized_scores": True,
            "top_k_emotions": 5,  # Return top K emotions per face

            # Performance configuration
            "batch_size": 32,
            "device": "auto",  # "cpu", "cuda", "auto"
        }
        merged_config = default_config.copy()
        if config:
            merged_config.update(config)
        super().__init__(merged_config)
        self.logger = logging.getLogger(__name__)
        # Initialize face detector with configured backend
        self.face_detector = FaceAnalysisPipeline({
            "detection_backend": self.config["face_detection_backend"],
            "confidence_threshold": self.config["confidence_threshold"],
            "min_face_size": self.config["min_face_size"],
            "max_faces": self.config["max_faces"],
            "scale_factor": 1.05,  # More aggressive detection
            "min_neighbors": 3,   # More sensitive detection
        })
        self.processor: Optional[AutoProcessor] = None
        self.model: Optional[AutoModel] = None
        self.device: Optional[torch.device] = None
        # Prepare classifier container
        self.classifiers: Dict[str, Any] = {}

    def initialize(self) -> None:
        """Initialize the SigLIP encoder and face detector backend."""
        self.logger.info(f"Initializing LAIONFacePipeline with model_size: {self.config['model_size']}")
        # Initialize underlying face detector
        self.face_detector.initialize()
        # Load SigLIP feature extractor and model
        model_name = "google/siglip2-so400m-patch16-384"
        self.processor = AutoProcessor.from_pretrained(
            model_name, cache_dir=self.config["model_cache_dir"]
        )
        self.model = AutoModel.from_pretrained(
            model_name, cache_dir=self.config["model_cache_dir"]
        )
        # Determine device
        if self.config["device"] == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config["device"])
        self.model.to(self.device)
        # Set model info metadata
        self.set_model_info(model_name, self.config.get("model_cache_dir"))
        self.is_initialized = True
        # Load emotion classifier models from cache
        self._load_classifiers()

    def get_schema(self) -> Dict[str, Any]:
        """Return the output schema, extending the COCO schema to include emotions and model info."""
        schema = self.face_detector.get_schema()
        # Extend attributes for emotion predictions
        schema["annotation_schema"]["attributes"] = {
            "emotions": "dict[str, {score: float, rank: int}]",
            "model_info": "dict",
        }
        return schema

    def process(
        self,
        video_path: str,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        pps: float = 1.0,
        output_dir: Optional[str] = None,
        detection_results: Optional[Dict[str, Any]] = None,
        person_tracks: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process video to detect faces, extract embeddings, run emotion classifiers, and return annotations.
        """
        # Ensure initialization
        if not self.is_initialized:
            self.initialize()

        # Open video capture and metadata
        video_path_obj = Path(video_path)
        cap = cv2.VideoCapture(str(video_path_obj))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Determine frame range
        if end_time is None:
            end_time = total_frames / fps
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        # Fix frame step calculation: if pps is 0, process all frames (step=1)
        frame_step = max(1, int(fps / pps)) if pps > 0 else 1

        annotations: List[Dict[str, Any]] = []
        images: List[Dict[str, Any]] = []
        annotation_id = 1
        # Helper IoU for matching to person tracks
        def iou(a, b):
            xa1, ya1, wa, ha = a
            xa2, ya2 = xa1+wa, ya1+ha
            xb1, yb1, wb, hb = b
            xb2, yb2 = xb1+wb, yb1+hb
            xi1, yi1 = max(xa1, xb1), max(ya1, yb1)
            xi2, yi2 = min(xa2, xb2), min(ya2, yb2)
            inter = max(0, xi2-xi1) * max(0, yi2-yi1)
            union = wa*ha + wb*hb - inter
            return inter/union if union>0 else 0

        # Process frames
        try:
            for frame_num in range(start_frame, min(end_frame, total_frames), frame_step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if not ret:
                    continue
                timestamp = frame_num / fps
                height, width = frame.shape[:2]
                # Get detections (external or via detector)
                image_id = f"{video_path_obj.stem}_frame_{frame_num}"
                if detection_results:
                    # Use provided detection results
                    if isinstance(detection_results, dict):
                        detections = [d for d in detection_results.get('annotations', [])
                                      if d.get('image_id')==image_id]
                    elif isinstance(detection_results, list):
                        # Handle case where detection_results is a list
                        detections = [d for d in detection_results
                                      if isinstance(d, dict) and d.get('image_id')==image_id]
                    else:
                        detections = []
                else:
                    # Use face detector to get detections
                    detections = self.face_detector._detect_faces_in_frame(
                        frame, timestamp, video_path_obj.stem, frame_num, width, height
                    )
                # Only proceed if faces found
                frame_annotations = []
                for det in detections:
                    # Handle both dict and other formats
                    if isinstance(det, dict):
                        bbox = det.get("bbox", [])
                    else:
                        # If det is not a dict, skip it
                        self.logger.warning(f"Unexpected detection format: {type(det)}")
                        continue
                        
                    if len(bbox) < 4:
                        self.logger.warning(f"Invalid bbox format: {bbox}")
                        continue
                        
                    x, y, w, h = map(int, bbox[:4])
                    
                    # Validate bbox is within frame bounds
                    if x < 0 or y < 0 or x + w > width or y + h > height:
                        self.logger.warning(f"Bbox out of bounds: {bbox} for frame {width}x{height}")
                        continue
                        
                    if w <= 0 or h <= 0:
                        self.logger.warning(f"Invalid bbox dimensions: {bbox}")
                        continue
                    
                    # Crop face from frame
                    face_crop = frame[y : y + h, x : x + w]
                    
                    if face_crop.size == 0:
                        self.logger.warning(f"Empty face crop for bbox: {bbox}")
                        continue
                    # Feature extraction: preprocess and move to device
                    try:
                        inputs = self.processor(images=face_crop, return_tensors="pt")
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    except Exception as e:
                        self.logger.warning(f"Failed to process face crop: {e}")
                        continue
                    
                    # Extract image embeddings: try CLIP-style get_image_features, fallback to encoder output
                    try:
                        with torch.no_grad():
                            try:
                                embedding = self.model.get_image_features(**inputs)
                            except Exception:
                                outputs = self.model(**inputs)
                                embedding = outputs.last_hidden_state[:, 0, :]
                    except Exception as e:
                        self.logger.warning(f"Failed to extract embeddings: {e}")
                        continue
                    
                    # Compute emotion scores via classifiers
                    raw_scores: Dict[str, float] = {}
                    for label, clf in self.classifiers.items():
                        try:
                            # classifier expects embedding tensor
                            with torch.no_grad():
                                logit = clf(embedding)  # raw model output (no sigmoid!)
                            # Store raw score directly (as in notebook)
                            raw_score = float(logit.item()) if torch.is_tensor(logit) else float(logit)
                            raw_scores[label] = raw_score
                        except Exception as e:
                            self.logger.warning(f"Failed to compute score for {label}: {e}")
                            # Set default score
                            raw_scores[label] = 0.0
                    
                    # Apply softmax across all emotions to get proper probability distribution
                    if raw_scores:
                        # Convert to tensor for softmax calculation
                        score_values = list(raw_scores.values())
                        score_tensor = torch.tensor(score_values, dtype=torch.float32)
                        softmax_scores = torch.softmax(score_tensor, dim=0)
                        
                        # Create emotion entries with proper softmax scores
                        emotion_items = []
                        for idx, (label, raw_score) in enumerate(raw_scores.items()):
                            softmax_score = float(softmax_scores[idx].item())
                            emotion_items.append((label, softmax_score, raw_score))
                        
                        # Sort by softmax scores and take top-k
                        emotion_items.sort(key=lambda x: x[1], reverse=True)
                        topk_items = emotion_items[: self.config.get("top_k_emotions", 5)]
                        emotions = {
                            label: {
                                "score": softmax_score, 
                                "rank": idx + 1,
                                "raw_score": raw_score
                            } 
                            for idx, (label, softmax_score, raw_score) in enumerate(topk_items)
                        }
                    else:
                        emotions = {}
                    
                    # Build annotation with emotions and model info
                    # Attach person_id via best IoU match if provided
                    person_id = None
                    if person_tracks:
                        for p in person_tracks:
                            if p.get('frame_number')==frame_num:
                                if iou(bbox, p.get('bbox', []))>0.5:
                                    person_id = p.get('track_id')
                                    break
                    
                    # Create a copy of the detection and add our attributes
                    annotation = det.copy() if isinstance(det, dict) else {}
                    annotation.update({
                        "bbox": bbox,
                        "timestamp": timestamp,
                        "frame_number": frame_num,
                        "image_id": image_id,
                        "attributes": {
                            "emotions": emotions,
                            "model_info": {
                                "model_size": self.config["model_size"],
                                "embedding_dim": embedding.shape[1]
                            },
                            **({"person_id": person_id} if person_id is not None else {})
                        },
                        "id": annotation_id
                    })
                    annotation_id += 1
                    frame_annotations.append(annotation)
                # record only frames with annotations
                if frame_annotations:
                    images.append(create_coco_image_entry(
                        image_id=image_id, width=width, height=height,
                        file_name=video_path_obj.name,
                        timestamp=timestamp, frame_number=frame_num
                    ))
                    annotations.extend(frame_annotations)
        finally:
            cap.release()

        # Export results if requested
        if output_dir and annotations:
            output_path = Path(output_dir) / f"{video_path_obj.stem}_laion_face_annotations.json"
            export_coco_json(annotations, images, str(output_path))

        self.logger.info(f"LAIONFacePipeline complete: {len(annotations)} annotations")
        return annotations

    def _load_classifiers(self) -> None:
        """Load pre-trained emotion classifier models for each label."""
        model_size = self.config.get("model_size", "small")
        cache_dir = Path(self.config.get("model_cache_dir", "./models/laion_face"))
        
        # Create directory if it doesn't exist
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Repository mapping
        repo_id = "laion/Empathic-Insight-Face-Small" if model_size == "small" else "laion/Empathic-Insight-Face-Large"
        
        for label, filename in EMOTION_LABELS.items():
            # Try to download the file if it doesn't exist locally
            local_path = cache_dir / filename
            
            if not local_path.exists():
                try:
                    self.logger.info(f"Downloading classifier for emotion: {label}")
                    downloaded_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        cache_dir=str(cache_dir.parent),  # Parent directory for HF cache structure
                        local_dir=str(cache_dir),  # Local directory to store file
                        local_dir_use_symlinks=False
                    )
                    self.logger.info(f"Downloaded {filename} to {downloaded_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to download classifier {label}: {e}")
                    continue
            
            # Load classifier if available
            if local_path.exists():
                try:
                    # Load state dict
                    state_dict = torch.load(local_path, map_location=self.device)
                    
                    # Create MLP classifier model with named layers to match state dict
                    embedding_dim = 1152  # SigLIP-so400m embedding dimension
                    
                    # Create a custom module with named layers matching the state dict
                    class EmotionClassifier(torch.nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.layers = torch.nn.ModuleDict({
                                '0': torch.nn.Linear(embedding_dim, 128),    # First layer: 1152 -> 128
                                '1': torch.nn.ReLU(),
                                '2': torch.nn.Dropout(0.1),
                                '3': torch.nn.Linear(128, 32),              # Second layer: 128 -> 32
                                '4': torch.nn.ReLU(),
                                '5': torch.nn.Dropout(0.1),
                                '6': torch.nn.Linear(32, 1)                 # Output layer: 32 -> 1
                            })
                        
                        def forward(self, x):
                            x = self.layers['0'](x)
                            x = self.layers['1'](x)
                            x = self.layers['2'](x)
                            x = self.layers['3'](x)
                            x = self.layers['4'](x)
                            x = self.layers['5'](x)
                            x = self.layers['6'](x)
                            return x
                    
                    classifier = EmotionClassifier()
                    
                    # Load the state dict into the model
                    classifier.load_state_dict(state_dict)
                    classifier.eval()
                    classifier.to(self.device)
                    
                    self.classifiers[label] = classifier
                    self.logger.info(f"Loaded classifier for emotion: {label}")
                except Exception as e:
                    self.logger.error(f"Error loading classifier {label}: {e}")
            else:
                self.logger.warning(f"Classifier file not found for emotion {label}: {local_path}")

    def cleanup(self) -> None:
        """Cleanup resources used by the pipeline."""
        if hasattr(self, 'face_detector'):
            self.face_detector.cleanup()
        self.model = None
        self.processor = None
        self.classifiers.clear()
        self.logger.info("Cleaned up LAIONFacePipeline resources.")
