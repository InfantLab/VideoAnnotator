# VideoAnnotator Pipeline Configuration Examples

## Scene Detection Configuration

```yaml
scene_detection:
  # PySceneDetect settings
  threshold: 30.0                    # Scene change detection threshold
  min_scene_length: 2.0             # Minimum scene length in seconds
  
  # CLIP classification settings
  model: "ViT-B/32"                 # CLIP model to use
  scene_labels:                     # Scene classification categories
    - "living room"
    - "bedroom" 
    - "kitchen"
    - "bathroom"
    - "nursery"
    - "clinic"
    - "office"
    - "outdoor"
    - "playground"
    - "car"
    - "restaurant"
    - "hospital"
  
  # Processing settings
  extract_keyframes: true           # Save keyframes for each scene
  keyframe_format: "jpg"           # Format for saved keyframes
  audio_analysis: false            # Analyze audio context (future)
```

## Person Tracking Configuration

```yaml
person_tracking:
  # YOLO11 model settings
  model_name: "yolo11n-pose.pt"     # YOLO11 pose model
  confidence_threshold: 0.5         # Detection confidence threshold
  iou_threshold: 0.5               # IoU threshold for NMS
  
  # Tracking settings
  track_mode: true                 # Enable multi-object tracking
  tracker_type: "bytetrack"        # Tracker: bytetrack, botsort
  persist_tracks: true             # Persist tracks across frames
  
  # Pose estimation settings
  pose_format: "coco_17"           # Keypoint format
  min_keypoint_confidence: 0.3     # Minimum keypoint confidence
  
  # Processing settings
  max_persons: 10                  # Maximum persons to track
  min_track_length: 5              # Minimum track length (frames)
  
  # Output settings
  save_trajectories: true          # Save complete trajectories
  trajectory_smoothing: true       # Apply trajectory smoothing
```

## Face Analysis Configuration

```yaml
face_analysis:
  # Backend selection
  backend: "mediapipe"             # Backend: openface, mediapipe, deepface
  
  # Detection settings
  face_confidence_threshold: 0.7   # Face detection confidence
  max_faces: 10                   # Maximum faces per frame
  
  # Analysis features
  detect_emotions: true           # Emotion recognition
  detect_gaze: true              # Gaze estimation (OpenFace only)
  detect_action_units: false     # Action Units (OpenFace only)
  detect_identity: false         # Face recognition
  
  # MediaPipe settings
  mediapipe:
    model_selection: 1            # 0: short-range, 1: full-range
    refine_landmarks: true        # Use refined face mesh
    
  # OpenFace 3.0 settings (when available)
  openface:
    model_path: "models/openface/"
    enable_3d_landmarks: true
    enable_head_pose: true
    enable_aus: true
    
  # DeepFace settings
  deepface:
    emotion_model: "VGG-Face"     # Emotion recognition model
    age_gender_model: "VGG-Face"  # Age/gender model
    detector_backend: "opencv"    # Face detector
    
  # Processing settings
  face_crop_padding: 20           # Padding around face crops
  emotion_batch_size: 8           # Batch size for emotion analysis
```

## Audio Processing Configuration

```yaml
audio_processing:
  # Speech recognition
  speech_recognition:
    model: "openai/whisper-base"   # Whisper model
    language: "auto"              # Language detection
    
  # Speaker diarization  
  speaker_diarization:
    model: "pyannote/speaker-diarization"
    min_speakers: 1
    max_speakers: 10
    
  # Audio classification
  audio_classification:
    model: "yamnet"               # Audio event classifier
    confidence_threshold: 0.5
    
  # Feature extraction
  features:
    sample_rate: 16000           # Audio sample rate
    hop_length: 512              # STFT hop length
    n_mels: 128                  # Mel spectrogram bins
    
  # Processing settings
  segment_length: 30.0           # Audio segment length (seconds)
  overlap: 0.5                  # Segment overlap ratio
```

## Global Configuration

```yaml
global:
  # Processing settings
  device: "auto"                 # Device: auto, cpu, cuda, mps
  batch_size: 8                  # Default batch size
  num_workers: 4                 # Number of worker processes
  
  # Output settings
  output_format: "json"          # Output format: json, csv, pkl
  save_intermediate: false       # Save intermediate results
  
  # Logging
  log_level: "INFO"              # Logging level
  log_file: "videoannotator.log" # Log file path
  
  # Performance
  memory_limit: "8GB"            # Memory limit
  gpu_memory_fraction: 0.8       # GPU memory fraction to use
  
  # Quality settings
  min_face_size: 50              # Minimum face size (pixels)
  min_person_size: 100           # Minimum person size (pixels)
  blur_threshold: 100            # Blur detection threshold
```

## Usage Examples

### Command Line Usage

```bash
# Process single video with all pipelines
python -m videoannotator.cli process \
  --video path/to/video.mp4 \
  --config config/full_pipeline.yaml \
  --output output/annotations/

# Process with specific pipelines only
python -m videoannotator.cli process \
  --video path/to/video.mp4 \
  --pipelines scene_detection,person_tracking \
  --config config/custom.yaml

# Batch processing
python -m videoannotator.cli batch \
  --input_dir videos/ \
  --output_dir outputs/ \
  --config config/batch.yaml
```

### Python API Usage

```python
from videoannotator import VideoAnnotator

# Initialize with config
annotator = VideoAnnotator(config_path="config/full_pipeline.yaml")

# Process single video
results = annotator.process_video(
    video_path="path/to/video.mp4",
    pipelines=["scene_detection", "person_tracking", "face_analysis"]
)

# Save results
annotator.save_results(results, "output/annotations.json")
```

### Pipeline-specific Usage

```python
from videoannotator.pipelines import ScenePipeline

# Configure and run scene detection
config = {
    "threshold": 25.0,
    "scene_labels": ["indoor", "outdoor", "office"]
}

with ScenePipeline(config) as pipeline:
    annotations = pipeline.process("video.mp4")
```

## Configuration Validation

The configuration files are validated against JSON schemas to ensure correctness:

- Required fields are enforced
- Value ranges are checked (e.g., confidence thresholds 0-1)
- Model paths are validated
- Hardware compatibility is checked

## Environment-specific Configurations

Create different configs for different environments:

- `config/development.yaml` - For development/testing
- `config/production.yaml` - For production deployment  
- `config/gpu.yaml` - For GPU-accelerated processing
- `config/cpu_only.yaml` - For CPU-only processing
