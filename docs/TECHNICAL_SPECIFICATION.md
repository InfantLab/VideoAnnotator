# VideoAnnotator Technical Specification

VideoAnnotator is a modern, modular video annotation system where each processor analyzes video segments and emits structured JSON annotations. Key annotation pipelines include person detection and tracking, pose estimation, object recognition, scene analysis, face detection & emotion recognition, speech transcription, and speaker/voice diarization.

The system replicates and extends capabilities found in commercial APIs (like Google's Video Intelligence API) using open-source models. Each pipeline accepts a video segment (start_time, end_time) and a "predictions per second" (PPS) parameter. If PPS ≥ FPS, output is per frame; PPS=1 yields one annotation per second; PPS=0 yields one result for the entire segment. All outputs are in standardized JSON format for maximum compatibility.

## Architecture Overview

### Scene Segmentation
Videos are segmented into fixed-length time slices (configurable, default 10 seconds) or using shot boundaries. Each segment is processed independently by the annotation pipelines. This allows parallel processing and avoids excessive memory usage.

### Person Detection & Tracking
Uses an object-detection model (YOLO11) to detect humans in each frame segment, then applies a tracker (ByteTrack or DeepSORT) to assign consistent IDs across frames. Outputs each person's bounding box, ID, and confidence at the requested PPS. Optionally infers roles by comparing box sizes.

### Pose Estimation & Tracking
Applies a pose estimation model (YOLO11-pose) to get 2D skeleton keypoints (joints) of each person. Outputs [x,y] coordinates and confidence for each keypoint. Pose tracking is maintained by tracking the corresponding person ID. Pose JSON data includes keypoints with (x,y) and confidence, following COCO keypoint format.

### Object Recognition
Runs a general object detector (YOLO11 trained on COCO) on each segment or frame. Outputs classes and bounding boxes for recognized objects (toys, phones, chairs, etc). Records each object label, its bounding box, timestamp/frame, and confidence.

### Scene/Shot Analysis
Applies shot-change detection and scene classification. Detects shot boundaries and classifies the overall environment (indoor/outdoor, room type) using scene classifiers. Records segments with start/end times where scene changes occur.

### Face Detection & Emotion Recognition
Detects faces in each frame using modern face detectors, then runs emotion/attribute classifiers (DeepFace or similar) on each face. Stores each face's bounding box, identity (if clustering or known IDs are used), and predicted emotion.

### Speech-to-Text (ASR)
Extracts the audio track and runs a speech recognition model (OpenAI's Whisper) to transcribe spoken words. Outputs time-aligned transcripts with start/end times, text, and confidence scores.

### Speaker Diarization
Performs speaker diarization using pyannote-audio to segment the audio by speaker turns. Aligns each speaker segment with detected persons when possible. Emits annotations indicating speaker segments with timestamps.

### Audio Events
Optionally runs audio-event detectors for specific events like laughter or crying. Each detected event is annotated with its time range and event type.

## Data Format and Storage

All outputs use JSON to maximize interoperability. Each annotation record includes the video ID, segment/frame timestamp, pipeline name, and data fields. 

Example person-detection JSON:
```json
{
  "type": "person_detection",
  "video_id": "example_video",
  "timestamp": 12.34,
  "bbox": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
  "person_id": 1,
  "confidence": 0.87,
  "metadata": {
    "model": "yolo11n-pose.pt",
    "frame": 370
  }
}
```

All JSONs are collected in standardized folders organized by video ID and pipeline. The system follows existing formats (compatible with CVAT/Datumaro JSON) to ensure compatibility with manual annotation tools.

## Project Structure

```
src/
├── pipelines/           # Core annotation pipelines
│   ├── base_pipeline.py # Common interface
│   ├── scene_detection/ # Scene analysis and object detection  
│   ├── person_tracking/ # Multi-person tracking and pose
│   ├── face_analysis/   # Face detection and emotion
│   └── audio_processing/# Speech recognition and diarization
├── schemas/            # JSON data schemas
└── version.py          # Versioning and metadata

data/                   # Input videos and auxiliary data
output/                 # JSON annotation outputs
configs/                # YAML configuration files
tests/                  # Comprehensive test suite
docs/                   # Documentation
```

## Dependencies and Environment

Key libraries:
- **PyTorch** - Deep learning framework
- **Ultralytics (YOLO11)** - Object detection and pose estimation
- **OpenAI Whisper** - Speech recognition
- **DeepFace** - Face analysis and emotion recognition
- **OpenCV** - Computer vision operations
- **PyAnnote** - Speaker diarization
- **Pydantic** - Data validation and schemas

GPU support is enabled for all heavy models to ensure efficient processing.

## Output and Integration

Each pipeline logs its output to JSON in the output/ folder. Utilities are provided to merge or aggregate JSONs (for example, combining per-frame person bboxes into track segments). Because all annotations are JSON, they can be easily:

- Loaded by ML training code
- Imported into annotation tools (CVAT, LabelStudio)
- Used by front-ends for overlay/caption display
- Processed by analysis pipelines

All annotation records include segment boundaries (start/end times) or frame indices. When PPS=0, one JSON record spans the entire segment. When PPS>0, records are emitted at the specified rate with individual timestamps.

## Configuration and Extensibility

All components are documented and configurable through YAML files. The modular design allows for:

- Easy addition of new pipelines
- Flexible configuration of existing pipelines
- GPU/CPU optimization based on available hardware
- Batch processing capabilities
- Custom output formats and schemas

The system leverages modern MLOps practices with comprehensive versioning, metadata tracking, and reproducible outputs.
