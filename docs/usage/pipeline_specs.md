# VideoAnnotator Pipeline Specifications

> 📖 **Navigation**: [Getting Started](GETTING_STARTED.md) | [Demo Commands](demo_commands.md) | [Installation Guide](../installation/INSTALLATION.md) | [Main Documentation](../README.md)

This document outlines the modern, standards-based pipeline architecture for VideoAnnotator. Our pipelines are designed for research flexibility, production scalability, and seamless integration with annotation tools.

## 🧠 Overview

VideoAnnotator provides a **modular, JSON-centric annotation system** for extracting rich behavioral data from videos. The system supports both research analysis and production deployment with standardized outputs.

## 🎯 Design Goals

- **Modern Architecture**: Clean, maintainable pipeline modules using YOLO11, open-source models
- **Simplified Outputs**: JSON arrays with specification-compliant schemas
- **Research Flexibility**: Configurable processing parameters and extensible outputs  
- **Tool Integration**: Direct export to CVAT, LabelStudio, ELAN for manual annotation
- **Production Ready**: Efficient processing with GPU acceleration and batch support

---

## 📦 Current Project Structure

```plaintext
VideoAnnotator/
├── src/
│   ├── pipelines/
│   │   ├── audio_processing/      # Speech recognition & diarization
│   │   ├── face_analysis/         # Face detection & emotion analysis  
│   │   ├── person_tracking/       # Person detection & tracking
│   │   └── scene_detection/       # Scene boundary detection
│   ├── schemas/                   # Simplified JSON schemas
│   ├── exporters/                 # CVAT, LabelStudio, ELAN exporters
│   └── utils/                     # Shared utilities
├── tests/                         # Comprehensive test suite (94% success)
├── configs/                       # Pipeline configuration files
├── examples/                      # Usage examples and demos
└── docs/                          # Documentation and specifications
```

## 🧩 Pipeline Modules

Each pipeline follows a standardized interface pattern optimized for research and production use:

```python
class ModernPipeline(BasePipeline):
    """Standard pipeline interface for all VideoAnnotator pipelines."""
    
    def process(self, video_path: str, **kwargs) -> List[dict]:
        """Process video and return JSON-compatible annotations."""
        pass
    
    def export_to_file(self, annotations: List[dict], output_path: str) -> None:
        """Save annotations to standardized JSON format."""
        pass
```

### 1. Person Detection & Tracking

**Technology**: YOLO11 with ByteTrack tracking  
**Purpose**: Detect and track people across video frames  
**Output**: Normalized bounding boxes with persistent person IDs

```bash
# Modern API Usage (v1.2.0)
uv run videoannotator job submit video.mp4 --pipelines person

# Legacy Direct Usage (for development)
# from src.pipelines.person_tracking import PersonPipeline
# pipeline = PersonPipeline()
# annotations = pipeline.process("video.mp4")
```

**Output Format**:
```json
[
  {
    "type": "person_bbox",
    "video_id": "vid123",
    "t": 12.34,
    "bbox": [0.2, 0.3, 0.4, 0.5],
    "person_id": 1,
    "confidence": 0.87
  }
]
```

### 2. Face Analysis

**Technology**: YOLO11-face + OpenCV for emotion detection  
**Purpose**: Detect faces and analyze emotions  
**Output**: Face bounding boxes with emotion predictions

```python
# Usage  
from src.pipelines.face_analysis import FacePipeline

pipeline = FacePipeline()
annotations = pipeline.process("video.mp4")
```

**Output Format**:
```json
[
  {
    "type": "facial_emotion",
    "video_id": "vid123", 
    "t": 12.34,
    "person_id": 1,
    "face_id": "face_001",
    "bbox": [0.45, 0.23, 0.12, 0.18],
    "emotion": "happy",
    "confidence": 0.91
  }
]
```

### 3. Scene Detection

**Technology**: PySceneDetect with CLIP for scene classification  
**Purpose**: Detect scene boundaries and classify environments  
**Output**: Scene segments with transition information

```python
# Usage
from src.pipelines.scene_detection import ScenePipeline

pipeline = ScenePipeline()
annotations = pipeline.process("video.mp4")
```

**Output Format**:
```json
[
  {
    "type": "scene_boundary",
    "video_id": "vid123",
    "t": 45.6,
    "scene_id": "scene_003", 
    "boundary_type": "cut",
    "confidence": 0.95
  }
]
```

### 4. Audio Processing

**Technology**: Whisper (OpenAI) + pyannote.audio for diarization  
**Purpose**: Speech recognition and speaker identification  
**Output**: Transcripts with speaker diarization

```python
# Usage
from src.pipelines.audio_processing import AudioPipeline

pipeline = AudioPipeline() 
annotations = pipeline.process("video.mp4")
## 📋 Configuration & Usage

### Pipeline Configuration

All pipelines support flexible configuration via YAML files:

```yaml
# configs/default.yaml
person_tracking:
  model: "yolo11n.pt"
  confidence_threshold: 0.5
  tracking_method: "bytetrack"

face_analysis:
  model: "yolo11n-face.pt" 
  emotion_model: "opencv_dnn"
  min_face_size: 30

audio_processing:
  whisper_model: "base"
  language: "auto"
  diarization_enabled: true

scene_detection:
  threshold: 30.0
  min_scene_length: 1.0
```

### Command Line Interface

```bash
# Process single video with all pipelines
python -m videoannotator process video.mp4

# Process specific pipeline
python -m videoannotator process video.mp4 --pipeline person_tracking

# Custom config
python -m videoannotator process video.mp4 --config configs/high_performance.yaml

# Batch processing
python -m videoannotator batch videos/ --output results/
```

### Python API

```python
from videoannotator import VideoAnnotator

# Initialize with default config
annotator = VideoAnnotator()

# Process all pipelines
results = annotator.process("video.mp4")

# Process specific pipelines
results = annotator.process("video.mp4", pipelines=["person_tracking", "face_analysis"])

# Custom configuration
annotator = VideoAnnotator(config="configs/lightweight.yaml")
results = annotator.process("video.mp4")
```

## � Integration & Export

### Annotation Tool Export

```python
from videoannotator.exporters import CVATExporter, LabelStudioExporter

# Export to CVAT
cvat_exporter = CVATExporter()
cvat_exporter.export(annotations, "output.json")

# Export to LabelStudio
ls_exporter = LabelStudioExporter()
ls_exporter.export(annotations, "labelstudio_tasks.json")
```

### Data Analysis Integration

```python
import pandas as pd
from videoannotator.utils import annotations_to_dataframe

# Convert to pandas for analysis
df = annotations_to_dataframe(annotations)

# Filter by type
person_detections = df[df['type'] == 'person_bbox']
speech_segments = df[df['type'] == 'transcript']

# Temporal analysis
temporal_density = df.groupby('t').size()
```

## � Performance & Scalability

### Hardware Requirements

- **CPU**: Multi-core processor (8+ cores recommended)
- **Memory**: 16GB+ RAM for large videos
- **GPU**: CUDA-compatible GPU for acceleration (optional but recommended)
- **Storage**: SSD recommended for video I/O

### Processing Performance

| Pipeline | Video Duration | Processing Time | GPU Acceleration |
|----------|----------------|-----------------|------------------|
| Person Tracking | 1 minute | ~30 seconds | 3x speedup |
| Face Analysis | 1 minute | ~45 seconds | 2x speedup |
| Audio Processing | 1 minute | ~15 seconds | CPU-optimized |
| Scene Detection | 1 minute | ~10 seconds | CPU-optimized |

### Batch Processing

```python
# Efficient batch processing
from videoannotator import BatchProcessor

processor = BatchProcessor(
    config="configs/high_performance.yaml",
    num_workers=4,
    gpu_acceleration=True
)

# Process entire directory
results = processor.process_directory("videos/", output_dir="results/")
```

---

## � Future Roadmap

- **Multi-modal Fusion**: Combine audio and visual cues for enhanced accuracy
- **Real-time Processing**: Support for live video streams
- **Cloud Deployment**: Scalable processing on cloud platforms
- **Custom Model Integration**: Easy integration of domain-specific models
- **Interactive Annotation**: Web-based annotation interface with AI assistance

pyannote-audio

deepface

opencv-python

ffmpeg, librosa, torchaudio

## 📌 Summary
This video annotation framework is designed to be:

Modular, maintainable and extensible

Fully JSON-based and compatible with downstream ML and human-in-the-loop tasks

Based on modern open-source libraries and accelerates research on parent–child interaction at scale

## 📝 Conclusion
This specification provides a high-level overview of the architecture and data formats for the Auto-PCI video annotation pipeline. Each module can be developed independently, allowing for easy updates and integration of new models or features as they become available.