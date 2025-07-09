"""
Scene detection and classification pipeline using PySceneDetect and CLIP.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from pathlib import Path

from ..base_pipeline import BasePipeline
from ...schemas.scene_schema import SceneSegment, SceneAnnotation


class SceneDetectionPipeline(BasePipeline):
    """
    Scene segmentation and classification pipeline.
    
    Uses PySceneDetect for shot boundary detection and CLIP for scene classification.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "threshold": 30.0,  # Scene detection threshold
            "min_scene_length": 2.0,  # Minimum scene length in seconds
            "scene_prompts": [
                "living room", "kitchen", "bedroom", "outdoor", 
                "clinic", "nursery", "office", "playground"
            ],
            "clip_model": "ViT-B/32"
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
    
    def process(
        self, 
        video_path: str, 
        start_time: float = 0.0, 
        end_time: Optional[float] = None,
        pps: float = 0.0,
        output_dir: Optional[str] = None
    ) -> List[SceneSegment]:
        """Process video for scene detection and classification."""
        
        # Step 1: Scene segmentation using PySceneDetect
        scene_segments = self._detect_scene_boundaries(video_path, start_time, end_time)
        
        # Step 2: Scene classification using CLIP
        classified_segments = self._classify_scenes(video_path, scene_segments)
        
        # Step 3: Create annotation objects
        annotations = []
        video_metadata = self.get_video_metadata(video_path)
        
        for i, segment in enumerate(classified_segments):
            scene_annotation = SceneSegment(
                type="scene_segment",
                video_id=video_metadata.video_id,
                timestamp=(segment['start'] + segment['end']) / 2,  # Midpoint
                start_time=segment['start'],
                end_time=segment['end'],
                scene_id=f"scene_{i:03d}",
                scene_type=segment.get('classification'),
                confidence=segment.get('confidence'),
                metadata={
                    "duration": segment['end'] - segment['start'],
                    "frame_start": int(segment['start'] * video_metadata.fps),
                    "frame_end": int(segment['end'] * video_metadata.fps)
                }
            )
            annotations.append(scene_annotation)
        
        # Save if output directory specified
        if output_dir:
            output_path = Path(output_dir) / f"{video_metadata.video_id}_scenes.json"
            self.save_annotations(annotations, str(output_path))
        
        return annotations
    
    def _detect_scene_boundaries(
        self, 
        video_path: str, 
        start_time: float, 
        end_time: Optional[float]
    ) -> List[Dict[str, float]]:
        """Detect scene boundaries using PySceneDetect."""
        try:
            from scenedetect import detect, ContentDetector, split_video_ffmpeg
            from scenedetect.video_splitter import split_video_ffmpeg
        except ImportError:
            raise ImportError("PySceneDetect not installed. Run: pip install scenedetect")
        
        # Detect scenes
        scene_list = detect(
            video_path, 
            ContentDetector(threshold=self.config["threshold"]),
            start_time=start_time,
            end_time=end_time
        )
        
        # Convert to our format
        segments = []
        for scene in scene_list:
            start_sec = scene[0].get_seconds()
            end_sec = scene[1].get_seconds()
            
            # Filter by minimum scene length
            if end_sec - start_sec >= self.config["min_scene_length"]:
                segments.append({
                    "start": start_sec,
                    "end": end_sec
                })
        
        return segments
    
    def _classify_scenes(
        self, 
        video_path: str, 
        segments: List[Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """Classify scenes using CLIP."""
        try:
            import clip
            import torch
            from PIL import Image
            import cv2
        except ImportError:
            print("CLIP not available for scene classification")
            return segments
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(self.config["clip_model"], device=device)
        
        # Prepare text prompts
        text_prompts = [f"a {prompt}" for prompt in self.config["scene_prompts"]]
        text = clip.tokenize(text_prompts).to(device)
        
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
                    image_input = preprocess(image).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        logits_per_image, logits_per_text = model(image_input, text)
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
                
                classified_segments.append(segment)
        
        finally:
            cap.release()
        
        return classified_segments
    
    def get_schema(self) -> Dict[str, Any]:
        """Return JSON schema for scene annotations."""
        return {
            "type": "object",
            "properties": {
                "type": {"type": "string", "const": "scene_segment"},
                "video_id": {"type": "string"},
                "timestamp": {"type": "number"},
                "start_time": {"type": "number"},
                "end_time": {"type": "number"},
                "scene_id": {"type": "string"},
                "scene_type": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "metadata": {"type": "object"}
            },
            "required": ["type", "video_id", "timestamp", "start_time", "end_time", "scene_id"]
        }
