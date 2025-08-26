# Getting Started with VideoAnnotator v1.2.0

Welcome to VideoAnnotator v1.2.0! This guide will help you get up and running with our modern production-ready API system and integrated background processing.

## What's New in v1.2.0 🎉

- **🚀 Production-Ready API Server** - Complete REST API with integrated background job processing
- **⚡ Integrated Processing** - No separate worker processes needed - everything runs in one server
- **🛠️ Modern CLI Interface** - Complete command-line tools for server management and job control
- **🔧 Enhanced Pipeline Support** - All pipelines (scene, person, face, audio) working through API
- **📊 Real-time Job Status** - Live job tracking and detailed results retrieval

## Prerequisites

- **Python 3.12+** (required)
- **uv** package manager (fast, modern dependency management)  
- **Git** (for version control)
- Optional: **CUDA-compatible GPU** for faster processing

## Quick Installation

### 1. Install uv Package Manager

```bash
# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone and Setup

```bash
git clone https://github.com/InfantLab/VideoAnnotator.git
cd VideoAnnotator

# Install all dependencies (fast!)
uv sync

# Install development dependencies  
uv sync --extra dev
```

### 3. Verify Installation

```bash
# Test the installation
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test CLI interface
uv run videoannotator --help
```

## Basic Usage

### 🚀 Start the API Server

VideoAnnotator v1.2.0 runs everything through one integrated API server with built-in background processing:

```bash
# Start the API server (recommended)
uv run videoannotator server --host 0.0.0.0 --port 8000

# View interactive API documentation at http://localhost:8000/docs
# Server includes integrated background job processing - no separate worker needed!
```

### 📹 Process Videos via CLI

The modern CLI makes video processing simple:

```bash
# Submit a video processing job
uv run videoannotator job submit video.mp4 --pipelines "scene,person,face"

# Check job status (returns job ID from submit command)
uv run videoannotator job status <job_id>

# Get detailed results
uv run videoannotator job results <job_id>

# List all jobs
uv run videoannotator job list --status completed
```

### 🔧 Other CLI Commands

```bash
# List available pipelines
uv run videoannotator pipelines --detailed

# Show system information and database status
uv run videoannotator info

# Validate configuration files
uv run videoannotator config --validate config.yaml

# Backup database
uv run videoannotator backup backup.db
```

### 🌐 Processing Videos via HTTP API

Direct API access for integration with other systems:

```bash
# Submit a video processing job
curl -X POST "http://localhost:8000/api/v1/jobs/" \
  -F "video=@video.mp4" \
  -F "selected_pipelines=scene,person,face"

# Check job status
curl "http://localhost:8000/api/v1/jobs/{job_id}"

# Get detailed results with pipeline outputs
curl "http://localhost:8000/api/v1/jobs/{job_id}/results"

# Download specific pipeline result files
curl "http://localhost:8000/api/v1/jobs/{job_id}/results/files/scene_detection" -O
```

### Using the Python API

```python
from src.pipelines.scene_detection.scene_pipeline import SceneDetectionPipeline
from src.pipelines.person_tracking.person_pipeline import PersonTrackingPipeline

# Scene detection
scene_config = {
    "threshold": 30.0,
    "min_scene_length": 1.0,
    "enabled": True
}

pipeline = SceneDetectionPipeline(scene_config)
pipeline.initialize()

results = pipeline.process(
    video_path="path/to/video.mp4",
    start_time=0.0,
    end_time=30.0,  # Process first 30 seconds
    output_dir="output/"
)

pipeline.cleanup()
```

### Configuration

VideoAnnotator uses YAML configuration files for flexible setup:

```yaml
# config.yaml
scene_detection:
  threshold: 30.0
  min_scene_length: 1.0
  enabled: true

person_tracking:
  model: "yolo11n-pose.pt"
  conf_threshold: 0.4
  iou_threshold: 0.7
  track_mode: true
```

## Understanding the Output

VideoAnnotator generates structured JSON files with comprehensive metadata:

```json
{
  "metadata": {
    "videoannotator": {
      "version": "1.1.0",
      "git": {"commit_hash": "359d693e..."}
    },
    "pipeline": {"name": "SceneDetectionPipeline"},
    "model": {"model_name": "PySceneDetect + CLIP"}
  },
  "annotations": [
    {
      "scene_id": "scene_001",
      "start_time": 0.0,
      "end_time": 10.0,
      "scene_type": "living_room"
    }
  ]
}
```

## Available Pipelines (All Working Through API!)

| Pipeline | Description | Output | Status |
|----------|-------------|--------|--------|
| **scene_detection** | Scene boundary detection + CLIP environment classification | `*_scene_detection.json` | ✅ Ready |
| **person_tracking** | YOLO11 + ByteTrack multi-person pose tracking | `*_person_tracking.json` | ✅ Ready |
| **face_analysis** | OpenFace 3.0 + LAION facial behavior analysis | `*_laion_face_annotations.json` | ✅ Ready |
| **audio_processing** | Whisper speech recognition + pyannote diarization | `*_speech_recognition.vtt` | ✅ Ready |

All pipelines are fully integrated with the API server and process through the background job system!

## Next Steps

- 📖 Read the [Full Installation Guide](../installation/INSTALLATION.md) for detailed setup
- 🔧 Explore [Pipeline Specifications](pipeline_specs.md) for detailed pipeline documentation
- 📊 Learn about [Demo Commands](demo_commands.md) for complete usage examples
- 🧪 Check out [Testing Overview](../testing/testing_overview.md) for QA information
- 🗺️ See the [v1.2.0 Roadmap](../development/roadmap_v1.2.0.md) for development progress
- 🎯 Try the [v1.2.1 Examples Update](../development/EXAMPLES_CLI_UPDATE_CHECKLIST.md) for latest CLI patterns

## Common Issues

### GPU Not Detected
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### FFmpeg Not Found
```bash
# Install FFmpeg
# Ubuntu/Debian:
sudo apt install ffmpeg

# macOS:
brew install ffmpeg

# Windows: Download from https://ffmpeg.org/
```

### Model Download Issues
Models are downloaded automatically on first use. Ensure you have:
- Stable internet connection
- Sufficient disk space (~2GB for all models)
- Proper permissions for the models directory

## Getting Help

- 📖 **Documentation**: Check the `docs/` folder
- 🐛 **Issues**: Report bugs on GitHub Issues
- 💬 **Discussions**: Join GitHub Discussions for questions
- 📧 **Contact**: Email the development team

## Performance Tips

1. **Use GPU**: Install CUDA-compatible PyTorch for 10x speedup
2. **Batch Processing**: Process multiple videos together
3. **Optimize Parameters**: Reduce PPS for faster processing
4. **Memory Management**: Process shorter segments for large videos

Happy annotating! 🎥✨
