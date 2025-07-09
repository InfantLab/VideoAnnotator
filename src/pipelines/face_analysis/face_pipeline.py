"""
Face analysis pipeline supporting OpenFace 3.0, DeepFace, and MediaPipe.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from pathlib import Path

from ..base_pipeline import BasePipeline
from ...schemas.face_schema import FaceDetection, FaceEmotion, FaceGaze
from ...schemas.base_schema import BoundingBox, KeyPoint


class FaceAnalysisPipeline(BasePipeline):
    """
    Comprehensive face analysis pipeline supporting multiple backends:
    - OpenFace 3.0 for facial landmarks, gaze estimation, and facial action units
    - DeepFace for emotion recognition and demographic estimation
    - MediaPipe for real-time face detection and landmarks
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "detection_backend": "opencv",  # opencv, ssd, dlib, mtcnn, retinaface, mediapipe
            "emotion_backend": "deepface",  # deepface, openface
            "gaze_backend": "openface",     # openface, mediapipe
            "landmark_backend": "openface", # openface, mediapipe, dlib
            "conf_threshold": 0.5,
            "min_face_size": 50,  # Minimum face size in pixels
            "skip_frames": 0,     # Process every N frames (0 = all frames)
            "openface_model_dir": "./models/openface",  # Path to OpenFace models
            "enable_emotion": True,
            "enable_gaze": True,
            "enable_landmarks": True,
            "enable_demographics": True
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
    
    def process(
        self, 
        video_path: str, 
        start_time: float = 0.0, 
        end_time: Optional[float] = None,
        pps: float = 10.0,  # 10 FPS for face analysis
        output_dir: Optional[str] = None
    ) -> List[FaceDetection]:
        """Process video for comprehensive face analysis."""
        
        # Get video metadata
        video_metadata = self.get_video_metadata(video_path)
        
        # Initialize backends
        self._initialize_backends()
        
        # Process video
        face_detections = []
        emotion_results = []
        gaze_results = []
        
        import cv2
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
                
                # Face detection
                faces = self._detect_faces(frame, timestamp, video_metadata.video_id, frame_number)
                face_detections.extend(faces)
                
                # Process each detected face
                for face in faces:
                    # Emotion analysis
                    if self.config["enable_emotion"]:
                        emotion = self._analyze_emotion(frame, face, timestamp, video_metadata.video_id)
                        if emotion:
                            emotion_results.append(emotion)
                    
                    # Gaze estimation
                    if self.config["enable_gaze"]:
                        gaze = self._estimate_gaze(frame, face, timestamp, video_metadata.video_id)
                        if gaze:
                            gaze_results.append(gaze)
                
                # Skip frames based on PPS
                frame_number += frame_skip
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        finally:
            cap.release()
        
        # Save results if output directory specified
        if output_dir:
            detection_path = Path(output_dir) / f"{video_metadata.video_id}_face_detections.json"
            self.save_annotations(face_detections, str(detection_path))
            
            if emotion_results:
                emotion_path = Path(output_dir) / f"{video_metadata.video_id}_face_emotions.json"
                self.save_annotations(emotion_results, str(emotion_path))
            
            if gaze_results:
                gaze_path = Path(output_dir) / f"{video_metadata.video_id}_face_gaze.json"
                self.save_annotations(gaze_results, str(gaze_path))
        
        return face_detections
    
    def _initialize_backends(self):
        """Initialize the selected backends."""
        self.backends = {}
        
        # Initialize OpenFace 3.0 if needed
        if any(backend == "openface" for backend in [
            self.config["emotion_backend"],
            self.config["gaze_backend"], 
            self.config["landmark_backend"]
        ]):
            self.backends["openface"] = self._init_openface()
        
        # Initialize DeepFace if needed
        if self.config["emotion_backend"] == "deepface":
            self.backends["deepface"] = self._init_deepface()
        
        # Initialize MediaPipe if needed
        if any(backend == "mediapipe" for backend in [
            self.config["detection_backend"],
            self.config["gaze_backend"],
            self.config["landmark_backend"]
        ]):
            self.backends["mediapipe"] = self._init_mediapipe()
    
    def _init_openface(self) -> Optional[Any]:
        """Initialize OpenFace 3.0."""
        try:
            # Note: This is a placeholder for OpenFace 3.0 integration
            # Actual implementation would depend on OpenFace 3.0 Python bindings
            import sys
            import os
            
            # Add OpenFace to path (example)
            openface_path = self.config["openface_model_dir"]
            if os.path.exists(openface_path):
                sys.path.append(openface_path)
                
            # Import OpenFace modules (example - actual API may differ)
            # from openface import FaceAnalyzer
            # return FaceAnalyzer(model_dir=openface_path)
            
            print("OpenFace 3.0 backend initialized")
            return {"initialized": True}  # Placeholder
            
        except ImportError:
            print("Warning: OpenFace 3.0 not available")
            return None
    
    def _init_deepface(self) -> Optional[Any]:
        """Initialize DeepFace backend."""
        try:
            from deepface import DeepFace
            return DeepFace
        except ImportError:
            print("Warning: DeepFace not available")
            return None
    
    def _init_mediapipe(self) -> Optional[Any]:
        """Initialize MediaPipe backend."""
        try:
            import mediapipe as mp
            mp_face_detection = mp.solutions.face_detection
            mp_face_mesh = mp.solutions.face_mesh
            
            return {
                "face_detection": mp_face_detection.FaceDetection(
                    model_selection=1, 
                    min_detection_confidence=self.config["conf_threshold"]
                ),
                "face_mesh": mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=10,
                    refine_landmarks=True,
                    min_detection_confidence=self.config["conf_threshold"]
                )
            }
        except ImportError:
            print("Warning: MediaPipe not available")
            return None
    
    def _detect_faces(self, frame: np.ndarray, timestamp: float, video_id: str, frame_number: int) -> List[FaceDetection]:
        """Detect faces in frame using selected backend."""
        faces = []
        height, width = frame.shape[:2]
        
        if self.config["detection_backend"] == "mediapipe" and "mediapipe" in self.backends:
            faces = self._detect_faces_mediapipe(frame, timestamp, video_id, frame_number, width, height)
        else:
            # Fall back to OpenCV cascade or other backends
            faces = self._detect_faces_opencv(frame, timestamp, video_id, frame_number, width, height)
        
        return faces
    
    def _detect_faces_mediapipe(self, frame: np.ndarray, timestamp: float, video_id: str, 
                               frame_number: int, width: int, height: int) -> List[FaceDetection]:
        """Detect faces using MediaPipe."""
        faces = []
        mp_backend = self.backends["mediapipe"]
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_backend["face_detection"].process(rgb_frame)
        
        if results.detections:
            for i, detection in enumerate(results.detections):
                bbox_norm = detection.location_data.relative_bounding_box
                
                # Filter by minimum face size
                face_width_px = bbox_norm.width * width
                face_height_px = bbox_norm.height * height
                
                if min(face_width_px, face_height_px) >= self.config["min_face_size"]:
                    bbox = BoundingBox(
                        x=bbox_norm.x_center - bbox_norm.width/2,
                        y=bbox_norm.y_center - bbox_norm.height/2,
                        width=bbox_norm.width,
                        height=bbox_norm.height
                    )
                    
                    # Extract landmarks if enabled
                    landmarks = None
                    if self.config["enable_landmarks"]:
                        landmarks = self._extract_mediapipe_landmarks(rgb_frame, bbox)
                    
                    face_detection = FaceDetection(
                        type="face_detection",
                        video_id=video_id,
                        timestamp=timestamp,
                        face_id=i,
                        person_id=None,  # Would need person tracking integration
                        bbox=bbox,
                        landmarks=landmarks,
                        confidence=detection.score[0],
                        quality_score=detection.score[0],
                        metadata={"frame": frame_number, "backend": "mediapipe"}
                    )
                    faces.append(face_detection)
        
        return faces
    
    def _detect_faces_opencv(self, frame: np.ndarray, timestamp: float, video_id: str,
                            frame_number: int, width: int, height: int) -> List[FaceDetection]:
        """Detect faces using OpenCV cascade."""
        import cv2
        
        # Load cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rects = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(self.config["min_face_size"], self.config["min_face_size"])
        )
        
        faces = []
        for i, (x, y, w, h) in enumerate(face_rects):
            # Normalize coordinates
            bbox = BoundingBox(
                x=x / width,
                y=y / height,
                width=w / width,
                height=h / height
            )
            
            face_detection = FaceDetection(
                type="face_detection",
                video_id=video_id,
                timestamp=timestamp,
                face_id=i,
                person_id=None,
                bbox=bbox,
                landmarks=None,
                confidence=0.8,  # OpenCV doesn't provide confidence
                quality_score=None,
                metadata={"frame": frame_number, "backend": "opencv"}
            )
            faces.append(face_detection)
        
        return faces
    
    def _extract_mediapipe_landmarks(self, rgb_frame: np.ndarray, bbox: BoundingBox) -> List[KeyPoint]:
        """Extract facial landmarks using MediaPipe."""
        if "mediapipe" not in self.backends:
            return []
        
        mp_backend = self.backends["mediapipe"]
        results = mp_backend["face_mesh"].process(rgb_frame)
        
        landmarks = []
        if results.multi_face_landmarks:
            # Take first face landmarks (could be improved with bbox matching)
            face_landmarks = results.multi_face_landmarks[0]
            
            # Convert key landmarks to our format
            key_landmark_indices = [33, 263, 1, 61, 291, 199]  # Eyes, nose, mouth corners
            
            for idx in key_landmark_indices:
                if idx < len(face_landmarks.landmark):
                    lm = face_landmarks.landmark[idx]
                    landmarks.append(KeyPoint(
                        x=lm.x,
                        y=lm.y,
                        confidence=0.9,  # MediaPipe doesn't provide per-landmark confidence
                        visible=True
                    ))
        
        return landmarks
    
    def _analyze_emotion(self, frame: np.ndarray, face: FaceDetection, 
                        timestamp: float, video_id: str) -> Optional[FaceEmotion]:
        """Analyze facial emotion."""
        if self.config["emotion_backend"] == "deepface" and "deepface" in self.backends:
            return self._analyze_emotion_deepface(frame, face, timestamp, video_id)
        elif self.config["emotion_backend"] == "openface" and "openface" in self.backends:
            return self._analyze_emotion_openface(frame, face, timestamp, video_id)
        return None
    
    def _analyze_emotion_deepface(self, frame: np.ndarray, face: FaceDetection,
                                 timestamp: float, video_id: str) -> Optional[FaceEmotion]:
        """Analyze emotion using DeepFace."""
        try:
            from deepface import DeepFace
            
            # Extract face region
            height, width = frame.shape[:2]
            x1 = int(face.bbox.x * width)
            y1 = int(face.bbox.y * height)
            x2 = int((face.bbox.x + face.bbox.width) * width)
            y2 = int((face.bbox.y + face.bbox.height) * height)
            
            face_img = frame[y1:y2, x1:x2]
            
            if face_img.size > 0:
                # Analyze with DeepFace
                result = DeepFace.analyze(
                    face_img, 
                    actions=['emotion', 'age', 'gender'],
                    enforce_detection=False
                )
                
                if isinstance(result, list):
                    result = result[0]
                
                emotions = result.get('emotion', {})
                dominant_emotion = result.get('dominant_emotion', 'neutral')
                age = result.get('age', 0)
                gender = result.get('dominant_gender', 'unknown')
                
                return FaceEmotion(
                    type="face_emotion",
                    video_id=video_id,
                    timestamp=timestamp,
                    face_id=face.face_id,
                    person_id=face.person_id,
                    emotions=emotions,
                    dominant_emotion=dominant_emotion,
                    age_estimate=int(age),
                    gender_estimate=gender,
                    confidence=emotions.get(dominant_emotion, 0) / 100.0,
                    metadata={"backend": "deepface"}
                )
        except Exception as e:
            print(f"Error in DeepFace emotion analysis: {e}")
        
        return None
    
    def _analyze_emotion_openface(self, frame: np.ndarray, face: FaceDetection,
                                 timestamp: float, video_id: str) -> Optional[FaceEmotion]:
        """Analyze emotion using OpenFace 3.0."""
        # Placeholder for OpenFace 3.0 emotion analysis
        # Actual implementation would use OpenFace 3.0 API
        return None
    
    def _estimate_gaze(self, frame: np.ndarray, face: FaceDetection,
                      timestamp: float, video_id: str) -> Optional[FaceGaze]:
        """Estimate gaze direction."""
        if self.config["gaze_backend"] == "openface" and "openface" in self.backends:
            return self._estimate_gaze_openface(frame, face, timestamp, video_id)
        return None
    
    def _estimate_gaze_openface(self, frame: np.ndarray, face: FaceDetection,
                               timestamp: float, video_id: str) -> Optional[FaceGaze]:
        """Estimate gaze using OpenFace 3.0."""
        # Placeholder for OpenFace 3.0 gaze estimation
        # Actual implementation would use OpenFace 3.0 gaze estimation API
        return None
    
    def get_schema(self) -> Dict[str, Any]:
        """Return JSON schema for face detection annotations."""
        return {
            "type": "object",
            "properties": {
                "type": {"type": "string", "const": "face_detection"},
                "video_id": {"type": "string"},
                "timestamp": {"type": "number"},
                "face_id": {"type": "integer"},
                "person_id": {"type": ["integer", "null"]},
                "bbox": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "width": {"type": "number"},
                        "height": {"type": "number"}
                    }
                },
                "landmarks": {
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
                "confidence": {"type": "number"},
                "quality_score": {"type": ["number", "null"]},
                "metadata": {"type": "object"}
            },
            "required": ["type", "video_id", "timestamp", "face_id", "bbox"]
        }
