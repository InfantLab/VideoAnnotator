# Getting Started with VideoAnnotator

This guide helps you get up and running with VideoAnnotator: with the start-up program
(recommended), with Docker Compose (labs and servers), or with a local `uv` install.

## Recommended: Start VideoAnnotator

With [Docker Desktop](https://www.docker.com/products/docker-desktop/) or
[Podman Desktop](https://podman-desktop.io/) installed and started once:

```bash
# Linux and macOS: install the start-up program
curl -LsSf https://github.com/InfantLab/VideoAnnotator/releases/latest/download/install.sh | sh

# then
videoannotator-start
```

On Windows, install it from PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://github.com/InfantLab/VideoAnnotator/releases/latest/download/install.ps1 | iex"
```

and double-click **Start VideoAnnotator** on your desktop.

It asks which folder your videos are in, confirms that VideoAnnotator can read it but never change
it, and opens the viewer in your browser, connected. **Videos → Folders** shows the folder you
shared (see [Using the viewer](#using-the-viewer)). Results go to `VideoAnnotator` in your home
folder. Next time it asks nothing.
`videoannotator-start share` shares another folder; the
[installation guide](../installation/INSTALLATION.md#start-videoannotator-researchers) has the rest,
and [troubleshooting](../installation/troubleshooting.md#the-start-up-program) explains each message.

## Prerequisites

- **Python 3.12 or 3.13** (required; 3.13 recommended)
- **uv** package manager (fast, modern dependency management)
- **Git** (for version control)
- Optional: **CUDA-compatible GPU** for faster processing

## Quick Installation

## Labs and servers: Docker Compose

To run VideoAnnotator from a checkout with Docker Compose:

```bash
# CPU: your videos in ~/Studies, results in ~/VideoAnnotator
VIDEOS_DIR=~/Studies RESULTS_DIR=~/VideoAnnotator docker compose --profile prod up --build videoannotator-prod

# GPU (requires NVIDIA Container Toolkit)
VIDEOS_DIR=~/Studies RESULTS_DIR=~/VideoAnnotator docker compose --profile gpu up --build videoannotator-gpu
```

Your video folder is mounted read-only and the server is reachable from this computer only; see
[Your videos and results under Docker](../installation/INSTALLATION.md#your-videos-and-results-under-docker).

Open http://localhost:18011/docs for the interactive API documentation, or
http://localhost:18011/viewer to review annotated output in the bundled
[Video Annotation Viewer](../../viewer/) — no separate
install required (disable with `VIDEOANNOTATOR_ENABLE_VIEWER=false` if you don't want it).

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

# Development tools come with `uv sync` by default (dependency group `dev`)

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
# Review annotated output at http://localhost:18011/viewer
# Server includes integrated background job processing - no separate worker needed!
```

**One-command start (Linux, macOS, WSL)**: `scripts/start_server.sh` syncs the environment, sets
up the database and an admin API key (asking for an admin email the first time), starts the
server, and prints the viewer login link once the server is up. It is safe to re-run. Pass
`--background` to keep the server running after the terminal closes, and `--help` for the other
options (port, extras to install, non-interactive use). To shorten the commands, add
`source /path/to/VideoAnnotator/scripts/shell_aliases.sh` to your `~/.bashrc` or `~/.zshrc`: then
`va-start` runs the script and `va` runs `uv run videoannotator`, from any folder. The dev
container sets these up for you, and also puts `va` and `videoannotator` on the PATH as commands,
so they work in scripts and non-interactive shells too.

**Viewer**: `/viewer` serves a bundled build of Video Annotation Viewer, pre-configured to talk to
this server (same-origin, no setup needed). It also runs on its own — see
[`viewer/README.md`](../../viewer/README.md) if you want the standalone client
(e.g. for reviewing output from other tools, or a different VideoAnnotator instance).

**CORS Note**: The official standalone web client (video-annotation-viewer on port 19011) is
automatically allowed. For custom clients, use `--dev` mode or set `CORS_ORIGINS` environment variable.

## Using the viewer

Open http://127.0.0.1:18011/viewer/. The top bar has **Home · Videos · Ask AI · Pipelines**, the
runs going on, **Help** and **Settings**. Once the server has results, the viewer opens on Videos;
on a new server it opens on Home.

### Three ways to start

Home offers three:

- **Ask AI about a video**: ask a vision model a question about one video, try it on a few
  moments, then run it on the whole video ([Ask AI](ask_ai.md)).
- **Run pipelines on videos**: find speech, faces, people, scenes and more in a folder of videos,
  and get one results table ([Videos and projects](videos_and_projects.md)).
- **Open files from this computer**: look at results you already have as files, with their video,
  without a server.

### Your first run

1. **Videos → Folders**: pick the folder your videos are in. Every video in it is listed.
2. **Run…**: tick what to find (Faces, Speech, …) and keep the settings as they are.
3. **Start run on N videos**. You can close the tab: the run carries on on the server, and the top
   bar shows its progress.
4. When it finishes, a message offers the **Results table**; the run's header says
   **N videos → results table**, with ELAN and a methods paragraph beside it.

Videos are read where they are; nothing is copied. Results go to **`~/VideoAnnotator`**: one folder
per run, named after the run and its date, with one folder per video and a `run.json` describing
the run. See [Choosing videos](../installation/INSTALLATION.md#choosing-videos-on-your-own-computer)
and [Where results go](../installation/INSTALLATION.md#where-results-go) to change either folder.

Click a video to open its [Video page](video_page.md): the video with every result as a track.
Group videos into **projects** for a study whenever you like; see
[Videos and projects](videos_and_projects.md#projects).

### Demos

Home and Help open demo results (a short clip with every pipeline's output) in the Video page,
with no server needed.

### Words

The viewer uses a few words throughout:

| Word | Means |
|---|---|
| video | one recording, wherever the server keeps it |
| folder | where videos are: a shared folder, or Uploaded videos |
| run | one go of pipelines over some videos |
| project | a set of videos grouped for a study; a video can be in several |
| workflow | a saved set of pipelines and settings, reused when you ask |
| track | one row of results on the Video page |
| results table | one table of every pipeline's results, for your statistics |

The command line and the API keep their own words: a run of one video is a *job*, a run of several
a *batch*, a project a *dataset*, and a workflow a *preset*.

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
  "http://localhost:18011/api/v1/jobs/{job_id}/results/files/scene_detection" -OJ
```

### Using the Python API

```python
from videoannotator.pipelines.scene_detection.scene_pipeline import (
    SceneDetectionPipeline,
)
from videoannotator.pipelines.person_tracking.person_pipeline import (
    PersonTrackingPipeline,
)

# Scene detection
scene_config = {"threshold": 30.0, "min_scene_length": 1.0, "enabled": True}

pipeline = SceneDetectionPipeline(scene_config)
pipeline.initialize()

results = pipeline.process(
    video_path="path/to/video.mp4",
    start_time=0.0,
    end_time=30.0,  # Process first 30 seconds
    output_dir="output/",
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
  model: "yolo26n-pose.pt"
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
| **face_analysis**    | DeepFace face detection and emotion                        | `*_face_detections.json`       | ✅ Ready |
| **audio_processing** | Whisper speech recognition + pyannote diarization          | `*_speech_recognition.vtt`      | ✅ Ready |

All pipelines are fully integrated with the API server and process through the background job system!

## Next Steps

- Read the [Full Installation Guide](../installation/INSTALLATION.md) for detailed setup
- Explore [Pipeline Specifications](pipeline_specs.md) for detailed pipeline documentation
- Learn about [Demo Commands](demo_commands.md) for complete usage examples
- Check out [Testing Overview](../testing/testing_overview.md) for QA information

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
