# VideoAnnotator

[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi&logoColor=white)](http://localhost:8000/docs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-package%20manager-FF4B4B?logo=uv&logoColor=white)](https://github.com/astral-sh/uv)
[![Docker](https://img.shields.io/badge/Docker-GPU%20Ready-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/)
[![Tests](https://img.shields.io/badge/tests-83%25%20passing-brightgreen.svg)](tests/)

A **modern REST API and toolkit** for comprehensive video analysis of human interactions. Built with FastAPI, simplified schemas, standards-based pipelines, and seamless annotation tool integration.

**🎯 Perfect for researchers** who need to integrate video analysis into their workflows via a simple HTTP API.

## 🚀 **API Server - Get Started in 30 Seconds**

VideoAnnotator now features a **modern FastAPI server** for easy integration into your research workflow:

```bash
# Quick setup
curl -LsSf https://astral.sh/uv/install.sh | sh  # Install uv package manager
git clone https://github.com/InfantLab/VideoAnnotator.git && cd VideoAnnotator
uv sync  # Install dependencies (30 seconds)

# Start the API server
uv run python api_server.py
```

**🎉 That's it!** Your API server is now running at:
- **📖 Interactive API docs**: http://localhost:8000/docs  
- **⚡ JSON API**: http://localhost:8000/
- **🔄 Health check**: http://localhost:8000/health

### 🔥 **Why Use the API Server?**

- **⚡ Fast startup** - Models load on-demand, not at boot
- **📱 Easy integration** - RESTful API works with any language  
- **🔄 Async processing** - Handle multiple videos simultaneously
- **📊 Real-time status** - Monitor job progress via API
- **🐳 Container ready** - Deploy anywhere with Docker

## ✨ Key Features

### 🏗️ **Modern Architecture**
- **FastAPI server** with automatic OpenAPI documentation
- **YOLO11-powered** person tracking and scene detection
- **Open-source models** for face analysis and audio processing  
- **Simplified JSON schemas** for maximum interoperability

### 🎯 **Research Ready**
- **RESTful API** for integration with research workflows
- **Annotation tool integration**: Direct export to CVAT, LabelStudio, ELAN
- **Batch processing**: Efficient multi-video API endpoints
- **Reproducible outputs**: Version-controlled processing

### 🚀 **Production Scalable**
- **Async job processing** with status tracking
- **GPU acceleration** for compute-intensive pipelines
- **Docker support** for containerized deployment
- **Cross-platform** Windows/macOS/Linux compatibility

## 📚 **API Usage Examples**

### Process a Video via API

```bash
# Start the server
uv run python api_server.py

# Upload and process a video (in another terminal)
curl -X POST "http://localhost:8000/v1/jobs/" \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/path/to/video.mp4",
    "pipelines": ["person", "scene", "audio"],
    "config": {"output_format": "coco"}
  }'
```

### Monitor Job Progress

```bash
# Check job status
curl "http://localhost:8000/v1/jobs/your-job-id"

# Get results when complete
curl "http://localhost:8000/v1/jobs/your-job-id/results"
```

### Interactive API Exploration

Visit **http://localhost:8000/docs** for the full interactive API documentation with:
- 🎮 **Try it out** - Test endpoints directly in your browser
- 📋 **Request examples** - Copy-paste ready code snippets  
- 📊 **Response schemas** - Understand data formats
- 🔧 **Authentication** - API key setup and usage

### 🔌 **Integration Examples**

<details>
<summary><strong>🐍 Python</strong></summary>

```python
import requests

# Start a video processing job
response = requests.post("http://localhost:8000/v1/jobs/", json={
    "video_path": "video.mp4",
    "pipelines": ["person", "scene"],
    "config": {"output_format": "coco"}
})
job_id = response.json()["job_id"]

# Check job status
status = requests.get(f"http://localhost:8000/v1/jobs/{job_id}")
print(f"Status: {status.json()['status']}")

# Get results when complete
results = requests.get(f"http://localhost:8000/v1/jobs/{job_id}/results")
annotations = results.json()
```
</details>

<details>
<summary><strong>🌐 JavaScript/Node.js</strong></summary>

```javascript
// Start video processing job
const jobResponse = await fetch('http://localhost:8000/v1/jobs/', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    video_path: 'video.mp4',
    pipelines: ['person', 'scene'],
    config: { output_format: 'coco' }
  })
});
const { job_id } = await jobResponse.json();

// Poll for completion
const checkStatus = async () => {
  const status = await fetch(`http://localhost:8000/v1/jobs/${job_id}`);
  const data = await status.json();
  
  if (data.status === 'completed') {
    const results = await fetch(`http://localhost:8000/v1/jobs/${job_id}/results`);
    const annotations = await results.json();
    console.log('Annotations ready:', annotations);
  }
};
```
</details>

<details>
<summary><strong>📊 R</strong></summary>

```r
library(httr)
library(jsonlite)

# Start video processing job
response <- POST("http://localhost:8000/v1/jobs/", 
  body = list(
    video_path = "video.mp4",
    pipelines = c("person", "scene"),
    config = list(output_format = "coco")
  ),
  encode = "json"
)
job_id <- content(response)$job_id

# Check job status
status_response <- GET(paste0("http://localhost:8000/v1/jobs/", job_id))
job_status <- content(status_response)$status

# Get results when complete
if (job_status == "completed") {
  results <- GET(paste0("http://localhost:8000/v1/jobs/", job_id, "/results"))
  annotations <- content(results)
}
```
</details>

## 🛠️ **Alternative Usage Methods**

For users who prefer command-line tools:

```bash
# Start API server
uv run python -m src.cli server --host 0.0.0.0 --port 8000

# API-based processing (submit jobs via web interface or curl)
curl -X POST "http://localhost:8000/api/v1/jobs" -H "Content-Type: application/json" -d '{"video_path": "video.mp4"}'
```

📖 **[Full Documentation](docs/)** | 🧪 **[Examples](examples/)** | 🔧 **[Installation Guide](docs/installation/INSTALLATION.md)**

## 👨‍💻 **Development & Deployment**

### API Server Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Start API server with auto-reload (for development)
uv run uvicorn api_server:app --reload --host 0.0.0.0 --port 8000

# Run API tests
uv run pytest tests/api/ -v
```

### Code Quality

```bash
# Run linting and formatting
uv run ruff check . && uv run ruff format .

# Run full test suite
uv run pytest

# Type checking
uv run mypy src
```

### Docker Deployment

```bash
# Build and run API server in container
docker build -f Dockerfile.gpu -t videoannotator:api .
docker run -p 8000:8000 --gpus all videoannotator:api

# Access API at http://localhost:8000/docs
```

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

📖 **[OpenFace 3.0 Integration Guide](docs/OPENFACE3_GUIDE.md)**

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
