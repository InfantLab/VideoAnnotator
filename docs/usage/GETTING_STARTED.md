# Getting Started with VideoAnnotator v1.2.0

This guide will help you get up and running with VideoAnnotator's modern uv-based workflow.

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
uv run python -m src.cli --help
```

## Basic Usage

### API Server Mode (Recommended)

```bash
# Start the API server
uv run python -m src.cli server --host 0.0.0.0 --port 8000

# Alternative: Start server directly
uv run python api_server.py

# View API documentation at http://localhost:8000/docs
```

### Processing Videos via API

Once the server is running, submit processing jobs via:

```bash
# Submit a video processing job
curl -X POST "http://localhost:8000/api/v1/jobs" \
  -H "Content-Type: application/json" \
  -d '{"video_path": "video.mp4", "pipelines": ["scene", "person"]}'

# Check job status
curl "http://localhost:8000/api/v1/jobs/{job_id}/status"
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

## Available Pipelines

| Pipeline | Description | Output |
|----------|-------------|--------|
| **Scene Detection** | Detects scene boundaries and classifies environments | `{video}_scenes.json` |
| **Person Tracking** | Tracks people across frames with pose estimation | `{video}_person_detections.json` |
| **Face Analysis** | Detects faces and analyzes emotions | `{video}_faces.json` |
| **Audio Processing** | Speech recognition and speaker diarization | `{video}_speech.json` |

## Next Steps

- 📖 Read the [Full Installation Guide](INSTALLATION.md) for detailed setup
- 🔧 Explore [Configuration Options](../configs/README.md)
- 📊 Learn about [Output Formats](OUTPUT_FORMATS.md)
- 🧪 Check out [Testing Standards](TESTING_STANDARDS.md)
- 🗺️ See the [Development Roadmap](ROADMAP.md)

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
