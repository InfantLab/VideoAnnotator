# VideoAnnotator

[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi&logoColor=white)](http://localhost:18011/docs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-package%20manager-FF4B4B?logo=uv&logoColor=white)](https://github.com/astral-sh/uv)
[![Docker](https://img.shields.io/badge/Docker-GPU%20Ready-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/)
[![Tests](https://img.shields.io/badge/tests-720%20passing%20(94.4%25)-success.svg)](tests/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/InfantLab/VideoAnnotator)

**Automated video analysis toolkit for human interaction research** - Extract comprehensive behavioral annotations from videos using AI pipelines, with an intuitive web interface for visualization and analysis.

## 🎯 What is VideoAnnotator?

VideoAnnotator automatically analyzes videos of human interactions and extracts rich behavioral data including:

- **👥 Person tracking** - Multi-person detection and pose estimation with persistent IDs
- **😊 Facial analysis** - Emotions, expressions, gaze direction, and action units
- **🎬 Scene detection** - Environment classification and temporal segmentation
- **🎤 Audio analysis** - Speech recognition, speaker identification, and emotion detection

**Perfect for researchers studying parent-child interactions, social behavior, developmental psychology, and human-computer interaction.**

## 🖥️ Complete Solution: Processing + Visualization

VideoAnnotator provides both **automated processing** and **interactive visualization**:

### 📹 **VideoAnnotator** (this repository)

**AI-powered video processing pipeline**

- Processes videos to extract behavioral annotations
- REST API for integration with research workflows
- Supports batch processing and custom configurations
- Outputs standardized JSON data

### 🌐 **[Video Annotation Viewer](https://github.com/InfantLab/video-annotation-viewer)**

**Interactive web-based visualization tool**

- Load and visualize VideoAnnotator results
- Synchronized video playback with annotation overlays
- Timeline scrubbing with pose, face, and audio data
- Export tools for further analysis

**Complete workflow**: `Your Videos → [VideoAnnotator Processing] → Annotation Data → [Video Annotation Viewer] → Interactive Analysis`

## 🚀 Get Started in 60 Seconds

### 1. Quick Setup

```bash
# Install modern Python package manager
curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/Mac
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Clone and install
git clone https://github.com/InfantLab/VideoAnnotator.git
cd VideoAnnotator
uv sync  # Fast dependency installation (30 seconds)
```

### 2. Start Processing Videos

```bash
# Start the API server
uv run python api_server.py
# Note the API key printed on first startup - you'll need it below

# Process your first video (in another terminal)
curl -X POST "http://localhost:18011/api/v1/jobs/" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "video=@your_video.mp4" \
  -F "selected_pipelines=person,face,scene,audio"

# Check results at http://localhost:18011/docs
```

### 3. Visualize Results

```bash
# Install the companion web viewer
git clone https://github.com/InfantLab/video-annotation-viewer.git
cd video-annotation-viewer
npm install
npm run dev

Note: Ensure Node and NPM are installed. On macOS with Homebrew:
brew install node

# Open http://localhost:3000 and load your VideoAnnotator results
```

**🎉 That's it!** You now have both automated video processing and interactive visualization.

## 🧠 AI Pipelines & Capabilities

Authoritative pipeline metadata (names, tasks, modalities, capabilities) is generated from the registry:

- Pipeline specification table: `docs/pipelines_spec.md` (auto-generated; do not edit by hand)
- Emotion output format spec: `docs/specs/emotion_output_format.md`

Additional Specs:

- Output Naming Conventions: `docs/specs/output_naming_conventions.md` (stable patterns for downstream tooling)
- Emotion Validator Utility: `src/validation/emotion_validator.py` (programmatic validation of `.emotion.json` files)
- CLI Validation: `videoannotator validate-emotion path/to/file.emotion.json` returns non-zero exit on failure
  Client tools (e.g. the Video Annotation Viewer) should rely on those sources or the `/api/v1/pipelines` endpoint rather than hard-coding pipeline assumptions.

### **Person Tracking Pipeline**

- **Technology**: YOLO11 + ByteTrack multi-object tracking
- **Outputs**: Bounding boxes, pose keypoints, persistent person IDs
- **Use cases**: Movement analysis, social interaction tracking, activity recognition

### **Face Analysis Pipeline**

- **Technology**: OpenFace 3.0, LAION Face, OpenCV backends
- **Outputs**: 68-point landmarks, emotions, action units, gaze direction, head pose
- **Use cases**: Emotional analysis, attention tracking, facial expression studies

### **Scene Detection Pipeline**

- **Technology**: PySceneDetect + CLIP environment classification
- **Outputs**: Scene boundaries, environment labels, temporal segmentation
- **Use cases**: Context analysis, setting classification, behavioral context

### **Audio Processing Pipeline**

- **Technology**: OpenAI Whisper + pyannote speaker diarization
- **Outputs**: Speech transcripts, speaker identification, voice emotions
- **Use cases**: Conversation analysis, language development, vocal behavior

## 💡 Why VideoAnnotator?

### **🎯 Built for Researchers**

- **No coding required** - Web interface and REST API
- **Standardized outputs** - JSON formats compatible with analysis tools
- **Reproducible results** - Version-controlled processing pipelines
- **Batch processing** - Handle multiple videos efficiently

### **🔬 Research-Grade Accuracy**

- **State-of-the-art models** - YOLO11, OpenFace 3.0, Whisper
- **Validated pipelines** - Tested on developmental psychology datasets
- **Comprehensive metrics** - Confidence scores, validation tools
- **Flexible configuration** - Adjust parameters for your research needs

### **⚡ Production Ready**

- **Fast processing** - GPU acceleration, optimized pipelines
- **Scalable architecture** - Docker containers, API-first design
- **Cross-platform** - Windows, macOS, Linux support
- **Enterprise features** - Authentication, logging, monitoring

### **🔒 Privacy & Data Protection**

- **100% Local Processing** - All analysis runs on your hardware, no cloud dependencies
- **No Data Transmission** - Videos and results never leave your infrastructure
- **GDPR Compliant** - Full control over sensitive research data
- **Foundation Model Free** - No external API calls to commercial AI services
- **Research Ethics Ready** - Designed for studies requiring strict data confidentiality

## 📊 Example Output

VideoAnnotator generates rich, structured data like this:

```json
{
  "person_tracking": [
    {
      "timestamp": 12.34,
      "person_id": 1,
      "bbox": [0.2, 0.3, 0.4, 0.5],
      "pose_keypoints": [...],
      "confidence": 0.87
    }
  ],
  "face_analysis": [
    {
      "timestamp": 12.34,
      "person_id": 1,
      "emotion": "happy",
      "confidence": 0.91,
      "facial_landmarks": [...],
      "gaze_direction": [0.1, -0.2]
    }
  ],
  "scene_detection": [
    {
      "start_time": 0.0,
      "end_time": 45.6,
      "scene_type": "living_room",
      "confidence": 0.95
    }
  ],
  "audio_analysis": [
    {
      "start_time": 1.2,
      "end_time": 3.8,
      "speaker": "adult",
      "transcript": "Look at this toy!",
      "emotion": "excited"
    }
  ]
}
```

## 🔗 Integration & Export

### **Direct Integration**

- **Python**: Import JSON data into pandas, matplotlib, seaborn
- **R**: Load data with jsonlite, analyze with tidyverse
- **MATLAB**: Process JSON with built-in functions

### **Annotation Tools**

- **CVAT**: Computer Vision Annotation Tool integration
- **LabelStudio**: Machine learning annotation platform
- **ELAN**: Linguistic annotation software compatibility

### **Analysis Platforms**

- **Video Annotation Viewer**: Interactive web-based analysis (recommended)
- **Custom dashboards**: Build with our REST API
- **Jupyter notebooks**: Examples included in repository

## 🛠️ Installation & Usage

### **Method 1: Direct Installation (Recommended)**

```bash
# Modern Python environment
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/InfantLab/VideoAnnotator.git
cd VideoAnnotator
uv sync

# Start processing
uv run python api_server.py
```

### **Method 2: Docker (Production)**

```bash
# CPU version (lightweight)
docker build -f Dockerfile.cpu -t videoannotator:cpu .
docker run -p 18011:8000 videoannotator:cpu

# GPU version (faster processing)
docker build -f Dockerfile.gpu -t videoannotator:gpu .
docker run -p 18011:8000 --gpus all videoannotator:gpu

# Development version (pre-cached models)
docker build -f Dockerfile.dev -t videoannotator:dev .
docker run -p 18011:8000 --gpus all videoannotator:dev
```

### **Method 3: Research Platform Integration**

```python
# Python API for custom workflows
from videoannotator import VideoAnnotator

annotator = VideoAnnotator()
results = annotator.process("video.mp4", pipelines=["person", "face"])

# Analyze results
import pandas as pd
df = pd.DataFrame(results['person_tracking'])
print(f"Detected {df['person_id'].nunique()} unique people")
```

## 📚 Documentation & Resources

| Resource                                                                 | Description                            |
| ------------------------------------------------------------------------ | -------------------------------------- |
| **[📖 Interactive Docs](https://deepwiki.com/InfantLab/VideoAnnotator)** | Complete documentation with examples   |
| **[🎮 Live API Testing](http://localhost:18011/docs)**                   | Interactive API when server is running |
| **[🚀 Getting Started Guide](docs/usage/GETTING_STARTED.md)**            | Step-by-step setup and first video     |
| **[🔧 Installation Guide](docs/installation/INSTALLATION.md)**           | Detailed installation instructions     |
| **[⚙️ Pipeline Specifications](docs/usage/pipeline_specs.md)**           | Technical pipeline documentation       |
| **[🎯 Demo Commands](docs/usage/demo_commands.md)**                      | Example commands and workflows         |

## 👥 Research Applications

### **Developmental Psychology**

- **Parent-child interaction** studies with synchronized behavioral coding
- **Social development** research with multi-person tracking
- **Language acquisition** studies with audio-visual alignment

### **Clinical Research**

- **Autism spectrum** behavioral analysis with facial expression tracking
- **Therapy session** analysis with emotion and engagement metrics
- **Developmental assessment** with standardized behavioral measures

### **Human-Computer Interaction**

- **User experience** research with attention and emotion tracking
- **Interface evaluation** with gaze direction and facial feedback
- **Accessibility** studies with comprehensive behavioral data

## 🏗️ Architecture & Performance

### **Modern Technology Stack**

- **FastAPI** - High-performance REST API with automatic documentation
- **YOLO11** - State-of-the-art object detection and pose estimation
- **OpenFace 3.0** - Comprehensive facial behavior analysis
- **Whisper** - Robust speech recognition and transcription
- **PyTorch** - GPU-accelerated machine learning inference

### **Performance Characteristics**

- **Processing speed**: ~2-4x real-time with GPU acceleration
- **Memory usage**: 4-8GB RAM for typical videos
- **Storage**: ~100MB output per hour of video
- **Accuracy**: 90%+ for person detection, 85%+ for emotion recognition

### **Scalability**

- **Batch processing**: Handle multiple videos simultaneously
- **Container deployment**: Docker support for cloud platforms
- **Distributed processing**: API-first design for microservices
- **Resource optimization**: CPU and GPU variants available

## 🤝 Contributing & Community

### **Getting Involved**

- **🐛 Report issues**: [GitHub Issues](https://github.com/InfantLab/VideoAnnotator/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/InfantLab/VideoAnnotator/discussions)
- **📧 Contact**: Caspar Addyman at infantologist@gmail.com
- **🔬 Collaborations**: Open to research partnerships

### **Development**

- **Code quality**: 83% test coverage, modern Python practices
- **Documentation**: Comprehensive guides and API documentation
- **CI/CD**: Automated testing and deployment pipelines
- **Standards**: Following research software engineering best practices

## 📄 Citation & License

### **Citation**

If you use VideoAnnotator in your research, please cite:

```
Addyman, C. (2025). VideoAnnotator: Automated video analysis toolkit for human interaction research.
Zenodo. https://Zenodo. doi.org/10.5281/zenodo.16961751
```

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16961751.svg)](https://doi.org/10.5281/zenodo.16961751)

### **License**

MIT License - Full terms in [LICENSE](LICENSE)

### **Funding & Support**

- **The Global Parenting Initiative** (Funded by The LEGO Foundation)

## 🙏 Acknowledgments

### **Research Team**

- **Caspar Addyman** (infantologist@gmail.com) - Lead Developer & Research Director

### **Open Source Dependencies**

Built with and grateful to:

- **[YOLO & Ultralytics](https://ultralytics.com/)** - Object detection and tracking
- **[OpenFace 3.0](https://github.com/CMU-MultiComp-Lab/OpenFace-3.0)** - Facial behavior analysis
- **[OpenAI Whisper](https://github.com/openai/whisper)** - Speech recognition
- **[FastAPI](https://github.com/tiangolo/fastapi)** - Modern web framework
- **[PyTorch](https://pytorch.org/)** - Machine learning infrastructure

### **Development Tools & AI Assistance**

Development was greatly helped by:

- **[Visual Studio Code](https://code.visualstudio.com/)** - Primary development environment
- **[GitHub Copilot](https://github.com/features/copilot)** - AI pair programming assistance
- **[Claude Code](https://claude.ai/code)** - Architecture design and documentation
- **GPT-4 & Claude Models** - Code generation and debugging help

_This project demonstrates how AI-assisted development can accelerate research software creation while maintaining code quality and comprehensive testing._

---

**🎥 Ready to start analyzing videos?** Follow the [60-second setup](#-get-started-in-60-seconds) above!
