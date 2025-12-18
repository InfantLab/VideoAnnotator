# Getting Started with VideoAnnotator

This guide helps you get up and running with VideoAnnotator using Docker (recommended) or a local `uv` install.

## Prerequisites

- **Python 3.12+** (required)
- **uv** package manager (fast, modern dependency management)
- **Git** (for version control)
- Optional: **CUDA-compatible GPU** for faster processing

## Quick Installation

## Recommended: Run in Docker

For the most reliable setup (consistent dependencies, easier GPU support), run VideoAnnotator in Docker:

```bash
# CPU (default)
docker compose up --build

# GPU (requires NVIDIA Container Toolkit)
docker compose --profile gpu up --build videoannotator-gpu
```

Open http://localhost:18011/docs for the interactive API documentation.

To initialize the database and create an admin API key explicitly:

```bash
docker compose exec videoannotator setupdb --admin-email you@example.com --admin-username you

# If you launched the GPU service instead:
docker compose exec videoannotator-gpu setupdb --admin-email you@example.com --admin-username you
```

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

# Initialize the database and create an admin API key (idempotent)
uv run videoannotator setup-db --admin-email you@example.com --admin-username you
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

VideoAnnotator runs everything through one integrated API server with built-in background processing:

```bash
# Quick start (simplest - server is default)
uv run videoannotator

# Or explicitly specify the server command with options
uv run videoannotator server --host 0.0.0.0 --port 18011

# For custom client development or testing (allows all CORS origins)
uv run videoannotator --dev

# View interactive API documentation at http://localhost:18011/docs
# Server includes integrated background job processing - no separate worker needed!
```

**CORS Note**: The official web client (video-annotation-viewer on port 19011) is automatically allowed. For custom clients, use `--dev` mode or set `CORS_ORIGINS` environment variable.

### 📹 Process Videos via CLI

The modern CLI makes video processing simple:

```bash
# Submit a video processing job
uv run videoannotator job submit video.mp4 --pipelines "scene,person,face"

# Check job status (returns job ID from submit command)
uv run videoannotator job status <job_id>

# Get detailed results
uv run videoannotator job results <job_id>

# Download all annotations (ZIP)
uv run videoannotator job download-annotations <job_id>

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
# Use the API key printed by `setup-db` (or run `videoannotator generate-token`)
export API_KEY="va_api_your_key_here"

# Submit a video processing job
curl -X POST "http://localhost:18011/api/v1/jobs/" \
  -H "Authorization: Bearer $API_KEY" \
  -F "video=@video.mp4" \
  -F "selected_pipelines=scene,person,face"

# Check job status
curl -H "Authorization: Bearer $API_KEY" \
  "http://localhost:18011/api/v1/jobs/{job_id}"

# Get detailed results with pipeline outputs
curl -H "Authorization: Bearer $API_KEY" \
  "http://localhost:18011/api/v1/jobs/{job_id}/results"

# Download specific pipeline result files
curl -H "Authorization: Bearer $API_KEY" \
  "http://localhost:18011/api/v1/jobs/{job_id}/results/files/scene_detection" -O
```

### Using the Python API

```python
from videoannotator.pipelines.scene_detection.scene_pipeline import SceneDetectionPipeline
from videoannotator.pipelines.person_tracking.person_pipeline import PersonTrackingPipeline

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
      "version": "1.4.1",
      "git": { "commit_hash": "359d693e..." }
    },
    "pipeline": { "name": "SceneDetectionPipeline" },
    "model": { "model_name": "PySceneDetect + CLIP" }
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

| Pipeline             | Description                                                | Output                          | Status   |
| -------------------- | ---------------------------------------------------------- | ------------------------------- | -------- |
| **scene_detection**  | Scene boundary detection + CLIP environment classification | `*_scene_detection.json`        | ✅ Ready |
| **person_tracking**  | YOLO11 + ByteTrack multi-person pose tracking              | `*_person_tracking.json`        | ✅ Ready |
| **face_analysis**    | OpenFace 3.0 + LAION facial behavior analysis              | `*_laion_face_annotations.json` | ✅ Ready |
| **audio_processing** | Whisper speech recognition + pyannote diarization          | `*_speech_recognition.vtt`      | ✅ Ready |

All pipelines are fully integrated with the API server and process through the background job system!

## Next Steps

- Read the [Full Installation Guide](../installation/INSTALLATION.md) for detailed setup
- Explore [Pipeline Specifications](pipeline_specs.md) for detailed pipeline documentation
- Learn about [Demo Commands](demo_commands.md) for complete usage examples
- Check out [Testing Overview](../testing/testing_overview.md) for QA information
- See [Examples CLI Update Plan](../development/EXAMPLES_CLI_UPDATE_CHECKLIST.md) for updated CLI patterns

## Common Issues

### GPU Not Detected

```bash
# Check CUDA availability
uv run python -c "import torch; print(torch.cuda.is_available())"

# In Docker:
docker compose exec videoannotator uv run python -c "import torch; print(torch.cuda.is_available())"
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
