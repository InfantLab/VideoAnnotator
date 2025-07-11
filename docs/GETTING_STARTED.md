# Getting Started with VideoAnnotator

This guide will help you get up and running with VideoAnnotator quickly.

## Prerequisites

- Python 3.8 or higher
- FFmpeg (for audio/video processing)
- Git (for version control)
- Optional: CUDA-compatible GPU for faster processing

## Quick Installation

### 1. Clone the Repository

```bash
git clone https://github.com/InfantLab/VideoAnnotator.git
cd VideoAnnotator
```

### 2. Create Virtual Environment

```bash
# Using conda (recommended)
conda env create -f environment.yml
conda activate VideoAnnotator

# Or using pip
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python test_demo.py
```

This will run a quick test on sample videos to verify all pipelines are working correctly.

## Basic Usage

### Process a Single Video

```bash
# Basic processing with default settings
python -m src.pipelines.scene_detection.scene_pipeline --input video.mp4

# Process with person tracking
python -m src.pipelines.person_tracking.person_pipeline --input video.mp4

# Custom output directory
python -m src.pipelines.scene_detection.scene_pipeline --input video.mp4 --output results/
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
      "version": "1.0.0",
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
