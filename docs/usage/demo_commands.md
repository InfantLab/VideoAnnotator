# VideoAnnotator Demo Commands & Usage

> 📖 **Navigation**: [Getting Started](GETTING_STARTED.md) | [Accessing Results](accessing_results.md) | [Pipeline Specs](pipeline_specs.md) | [Installation Guide](../installation/INSTALLATION.md) | [Main Documentation](../README.md)

This guide demonstrates how to use VideoAnnotator with its API server and CLI interface for video processing.

## 🚀 Quick Start Commands

If you are running via Docker Compose (recommended), you can run commands inside the container like this:

```bash
# CPU service
docker compose exec videoannotator va --help

# GPU service (if you launched it)
docker compose exec videoannotator-gpu va --help
```

### Start API Server

```bash
# Start the VideoAnnotator API server
uv run videoannotator server --port 18011

# Server will be available at http://localhost:18011
# Interactive documentation at http://localhost:18011/docs
```

### Process Single Video via CLI

```bash
# Submit a video processing job through CLI
uv run videoannotator job submit video.mp4 --pipelines scene,person,face

# Check job status
uv run videoannotator job status <job_id>

# Get detailed results
uv run videoannotator job results <job_id>

# Download annotations (ZIP)
uv run videoannotator job download-annotations <job_id> --output ./results/
```

### Process Video via HTTP API

```bash
# Set your API key (from server startup output)
export API_KEY="va_api_your_key_here"

# Submit job via HTTP POST
curl -X POST "http://localhost:18011/api/v1/jobs/" \
  -H "Authorization: Bearer $API_KEY" \
  -F "video=@video.mp4" \
  -F "selected_pipelines=scene,person,face"

# Check status
curl -H "Authorization: Bearer $API_KEY" \
  "http://localhost:18011/api/v1/jobs/{job_id}"

# Download artifacts (ZIP)
curl -H "Authorization: Bearer $API_KEY" \
  "http://localhost:18011/api/v1/jobs/{job_id}/artifacts" \
  --output artifacts.zip
```

## 🛠️ CLI Management Commands

### System Information

```bash
# Show system status and database info
uv run videoannotator info

# List available pipelines
uv run videoannotator pipelines --detailed

# Validate configuration files
uv run videoannotator config --validate configs/default.yaml
```

### Job Management

```bash
# List all jobs
uv run videoannotator job list

# List completed jobs only
uv run videoannotator job list --status completed

# Get job results with details
uv run videoannotator job results <job_id>
```

## Job queueing

VideoAnnotator processes jobs in a simple FIFO queue:

- Jobs are dequeued in increasing `created_at` order (oldest pending job runs first).
- API responses may include `queue_position` for `pending` jobs.
  - `queue_position` is **1-based** among pending jobs.
  - `queue_position` is `null` (or absent) for non-pending jobs.
  - `queue_position` can change over time as other jobs complete.

## 🚀 Modern API-First Architecture
### Key Features:

- **Integrated Background Processing** - No separate worker processes needed
- **Real-time Job Status** - Live job tracking and progress updates
- **Complete Pipeline Integration** - All pipelines working through API
- **Modern CLI Interface** - Comprehensive command-line tools
- **Production Ready** - Designed for research and production workflows

## 📋 Complete API Reference

### Available Pipelines

- **scene_detection** - Scene boundary detection with CLIP classification
- **person_tracking** - YOLO11 + ByteTrack multi-person pose tracking
- **face_analysis** - OpenFace 3.0 + LAION facial behavior analysis
- **audio_processing** - Whisper speech recognition + pyannote diarization

### Pipeline Combinations

```bash
# Run all pipelines
uv run videoannotator job submit video.mp4 --pipelines scene,person,face,audio

# Scene + person analysis
uv run videoannotator job submit video.mp4 --pipelines scene,person

# Face analysis only
uv run videoannotator job submit video.mp4 --pipelines face
```

## 🔄 Working with Job Results

### Get Job Results

```bash
# Get summary of job results
uv run videoannotator job results <job_id>

# API endpoint for results
curl -H "Authorization: Bearer $API_KEY" \
  "http://localhost:18011/api/v1/jobs/{job_id}/results"

# Download specific pipeline result file
curl -H "Authorization: Bearer $API_KEY" \
  "http://localhost:18011/api/v1/jobs/{job_id}/results/files/scene_detection" -O
```

### Configuration Options

```bash
# Use custom configuration
uv run videoannotator job submit video.mp4 --config configs/high_performance.yaml

# Validate config before use
uv run videoannotator config --validate configs/high_performance.yaml

# View default configuration
uv run videoannotator config --show-default
```

### System Management

```bash
# Show version and system info
uv run videoannotator version
uv run videoannotator info

# Backup database
uv run videoannotator backup backup_$(date +%Y%m%d).db

# Server management
uv run videoannotator server --host 0.0.0.0 --port 18011
```

## 📊 Expected Output Format

### Job Submission Response

```json
{
  "id": "job_abc123",
  "status": "pending",
  "queue_position": 1,
  "video_path": "/path/to/video.mp4",
  "selected_pipelines": ["scene", "person", "face"],
  "created_at": "2025-08-26T10:30:00Z"
}
```

### Job Status Response

```json
{
  "id": "job_abc123",
  "status": "pending",
  "queue_position": 2,
  "created_at": "2025-08-26T10:30:00Z",
  "selected_pipelines": ["scene", "person", "face"]
}
```

### Job Results Response

```json
{
  "job_id": "job_abc123",
  "status": "completed",
  "pipeline_results": {
    "scene_detection": {
      "status": "completed",
      "processing_time": 15.2,
      "annotation_count": 8,
      "output_file": "/path/to/output/video_scene_detection.json"
    },
    "person_tracking": {
      "status": "completed",
      "processing_time": 45.7,
      "annotation_count": 156,
      "output_file": "/path/to/output/video_person_tracking.json"
    }
  },
  "output_dir": "/path/to/output/"
}
```

## 🎯 Next Steps

1. **Start the server**: `uv run videoannotator server`
2. **Submit a job**: `uv run videoannotator job submit your_video.mp4`
3. **Monitor progress**: `uv run videoannotator job status <job_id>`
4. **Get results**: `uv run videoannotator job results <job_id>`
5. **Explore API**: Visit `http://localhost:18011/docs` for interactive documentation

For more advanced usage, see:

- [Getting Started Guide](GETTING_STARTED.md) - Complete setup and workflow
- [Pipeline Specifications](pipeline_specs.md) - Detailed pipeline documentation
- [API Documentation](http://localhost:18011/docs) - Interactive API reference
