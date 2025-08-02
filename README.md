# VideoAnnotator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-94%25%20passing-brightgreen.svg)](tests/)

A **modern, research-focused toolkit** for comprehensive video analysis of human interactions. Built with simplified schemas, standards-based pipelines, and seamless annotation tool integration.

## ✨ Key Features

### 🏗️ **Modern Architecture**
- **YOLO11-powered** person tracking and scene detection
- **Open-source models** for face analysis and audio processing  
- **Simplified JSON schemas** for maximum interoperability
- **Test-driven development** with 94% success rate

### 🎯 **Research Ready**
- **Annotation tool integration**: Direct export to CVAT, LabelStudio, ELAN
- **Flexible data formats**: String/integer IDs, extensible schemas
- **Batch processing**: Efficient multi-video workflows
- **Reproducible outputs**: Version-controlled processing

### 🚀 **Production Scalable**
- **GPU acceleration** for compute-intensive pipelines
- **Configurable processing** via YAML configs
- **Docker support** for containerized deployment
- **Cross-platform** Windows/macOS/Linux compatibility

## Quick Start

```bash
# Clone and setup
git clone https://github.com/InfantLab/VideoAnnotator.git
cd VideoAnnotator
pip install -r requirements.txt

# Process a video
python -m videoannotator process video.mp4

# View results
ls output/video/  # JSON files ready for analysis
```

📖 **[Full Documentation](docs/)** | 🧪 **[Examples](examples/)** | 🔧 **[Installation Guide](docs/INSTALLATION.md)**
python main.py --input video.mp4 --config configs/high_performance.yaml

# Batch process multiple videos
python main.py --input videos/ --batch --parallel 4

## 🧩 Pipeline Architecture

VideoAnnotator provides four core pipelines, each optimized for specific analysis tasks:

### 🎬 **Scene Detection**
- **Technology**: PySceneDetect + CLIP classification
- **Purpose**: Boundary detection and environment classification  
- **Output**: Scene segments with transition metadata

### 👥 **Person Tracking** 
- **Technology**: YOLO11 + ByteTrack
- **Purpose**: Multi-person detection and tracking
- **Output**: Normalized bounding boxes with persistent IDs

### 😊 **Face Analysis**
- **Technology**: Multiple backends available:
  - **OpenFace 3.0** (recommended): Comprehensive facial behavior analysis
  - **LAION Face**: CLIP-based face analysis and emotion detection
  - **OpenCV**: Basic face detection with emotion analysis
- **Purpose**: Face detection, landmark extraction, emotion recognition, action units, head pose, gaze estimation
- **Output**: COCO format with facial landmarks, emotions, and behavioral features

### 🎤 **Audio Processing**
- **Technology**: Whisper + pyannote.audio
- **Purpose**: Speech recognition and speaker diarization
- **Output**: Transcripts with speaker identification

## 🎭 **OpenFace 3.0 Integration**

VideoAnnotator now supports **OpenFace 3.0** for comprehensive facial behavior analysis:

### Features
- **68-point facial landmarks** (2D and 3D)
- **Facial Action Units (AUs)** intensity and presence detection
- **Head pose estimation** (rotation and translation)
- **Gaze direction** and eye tracking
- **Face tracking** across video frames
- **COCO format output** for annotation tool compatibility

### Quick Setup
```bash
# 1. Install OpenFace 3.0 dependencies
python scripts/test_openface3.py

# 2. Process video with OpenFace 3.0
python main.py --config configs/openface3.yaml --video_path video.mp4

# 3. Results include comprehensive facial analysis
# - Facial landmarks in COCO keypoints format
# - Action unit intensities
# - Head pose angles
# - Gaze direction vectors
```

📖 **[Full OpenFace 3.0 Installation Guide](docs/OPENFACE3_INSTALLATION.md)**

## 📊 **Output Formats**

All pipelines generate **simple JSON arrays** compatible with annotation tools:

```json
[
  {
    "type": "person_bbox",
    "video_id": "example",
    "t": 12.34,
    "person_id": 1,
    "bbox": [0.2, 0.3, 0.4, 0.5],
    "confidence": 0.87
  }
]
```

**✅ Key Benefits:**
- **Tool Integration**: Direct import to CVAT, LabelStudio, ELAN
- **Research Friendly**: Simple formats for analysis and visualization
- **Extensible**: Models can add custom fields seamlessly

## 🚀 **Usage Examples**

### Python API
```python
from videoannotator import VideoAnnotator

# Process all pipelines
annotator = VideoAnnotator()
results = annotator.process("video.mp4")

# Specific pipelines only  
results = annotator.process("video.mp4", pipelines=["person_tracking"])

# Custom configuration
annotator = VideoAnnotator(config="configs/high_performance.yaml")
results = annotator.process("video.mp4")
```

### Command Line
```bash
# Single video processing
python -m videoannotator process video.mp4

# Batch processing
python -m videoannotator batch videos/ --output results/

# Specific pipeline
python -m videoannotator process video.mp4 --pipeline face_analysis

# Custom config
python -m videoannotator process video.mp4 --config configs/lightweight.yaml
```

### Export to Annotation Tools
```python
from videoannotator.exporters import CVATExporter, LabelStudioExporter

# Export to CVAT
CVATExporter().export(annotations, "cvat_project.json")

# Export to LabelStudio  
LabelStudioExporter().export(annotations, "labelstudio_tasks.json")
```

## 📁 **Project Structure**

```
VideoAnnotator/
├── src/
│   ├── pipelines/           # Core analysis pipelines
│   ├── schemas/             # JSON schemas & validation  
│   ├── exporters/           # Annotation tool exporters
│   └── utils/               # Shared utilities
├── tests/                   # Comprehensive test suite (94% success)
├── configs/                 # Pipeline configurations
├── examples/                # Usage examples and demos
├── docs/                    # Documentation
└── requirements.txt         # Dependencies
```

## 📚 **Documentation**

| Document | Description |
|----------|-------------|
| **[Installation Guide](docs/INSTALLATION.md)** | Setup and dependencies |
| **[Pipeline Specs](docs/Pipeline%20Specs.md)** | Technical pipeline details |
| **[Output Formats](docs/OUTPUT_FORMATS.md)** | JSON schema documentation |
| **[Testing Standards](docs/TESTING_STANDARDS.md)** | Test framework and practices |
| **[Configuration Guide](configs/README.md)** | YAML configuration options |

## 🧪 **Quality Assurance**

VideoAnnotator maintains high code quality through comprehensive testing:

```bash
# Run full test suite (94% success rate)
python -m pytest tests/ -v

# Test specific pipelines
python -m pytest tests/test_face_pipeline_modern.py -v

# Performance benchmarks
python -m pytest tests/ -m performance -v

# Test coverage analysis
python -m pytest tests/ --cov=src --cov-report=html
```

**📊 Test Results:**
- ✅ **67/71 tests passing** (94% success rate)
- ✅ **Zero code duplication** after rationalization
- ✅ **Modern test patterns** across all pipelines
- ✅ **Performance benchmarks** for optimization

## 🤝 **Contributing**

1. **Follow Standards**: Use existing [Testing Standards](docs/TESTING_STANDARDS.md)
2. **Add Tests**: Integrate into existing test files in `tests/`
3. **Update Docs**: Keep documentation current with changes
4. **Quality Check**: Ensure test suite maintains 90%+ success rate

## 📄 **License & Acknowledgments**

**License**: MIT - see [LICENSE](LICENSE) for details

**Acknowledgments**: Built on the shoulders of giants including YOLO, Whisper, PyTorch, and the open-source ML community. Special thanks to research communities advancing computer vision and audio processing.
