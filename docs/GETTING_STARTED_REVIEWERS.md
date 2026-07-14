# VideoAnnotator Quick Start for JOSS Reviewers

**Target Time**: <15 minutes to understand, install, and test VideoAnnotator

This guide helps JOSS reviewers quickly evaluate VideoAnnotator's functionality, code quality, and documentation.

## Overview (1 minute)

**VideoAnnotator** is a modern, modular toolkit for analyzing human interaction videos using AI pipelines.

**Key Features**:
- 🎯 Multi-modal analysis (face, audio, scene, emotions)
- 🔌 Modular pipeline architecture
- 🚀 REST API + CLI interfaces
- 📊 Standard output formats (COCO, WebVTT, RTTM)
- 🧪 Comprehensive test coverage (1000+ tests, >80%)
- 🔒 Secure by default (API key authentication)

**Repository**: [github.com/InfantLab/VideoAnnotator](https://github.com/InfantLab/VideoAnnotator)

---

## Quick Installation (5 minutes)

### Prerequisites
- Python 3.12+
- 10GB free disk space
- (Optional) NVIDIA GPU for acceleration

### Install

```bash
# 1. Clone repository
git clone https://github.com/InfantLab/VideoAnnotator.git
cd VideoAnnotator

# 2. Install uv package manager (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Install dependencies (fast with uv)
uv sync

# 4. Verify installation
uv run videoannotator diagnose
```

**Expected output**: All checks pass ✅ (GPU optional)

**Troubleshooting**: See [Troubleshooting Guide](installation/troubleshooting.md) if any checks fail.

---

## Quick Test (10 minutes)

### 1. Start API Server (1 min)

```bash
# Start server
uv run videoannotator
```

**Output**: Server starts on `http://localhost:18011`, API key displayed

```
[OK] API key loaded: va_api_xxx...
[OK] Server started: http://localhost:18011
[OK] Docs available: http://localhost:18011/docs
```

**Save your API key**:
```bash
export API_KEY="va_api_xxx..."  # Replace with your key
```

### 2. Explore API Documentation (2 min)

Open in browser: `http://localhost:18011/docs`

- Interactive Swagger UI with all endpoints
- curl examples for each endpoint
- Request/response schemas
- Try-it-out functionality

### 3. List Available Pipelines (1 min)

```bash
curl -X GET "http://localhost:18011/api/v1/pipelines" \
  -H "X-API-Key: $API_KEY"
```

**Expected output**: List of 10+ pipelines with metadata

### 4. Submit Test Job (3 min)

Download sample video or use your own:
```bash
# Download sample (3 seconds, faces)
curl -o sample.mp4 "https://sample-videos.com/video123/mp4/480/big_buck_bunny_480p_1mb.mp4"
```

Submit job:
```bash
curl -X POST "http://localhost:18011/api/v1/jobs/" \
  -H "X-API-Key: $API_KEY" \
  -F "video=@sample.mp4" \
  -F "selected_pipelines=person,face,scene,audio"
```

**Expected output**: Job created with `job_id`

### 5. Check Job Status (1 min)

```bash
# Replace with your job_id
export JOB_ID="job_abc123..."

curl -X GET "http://localhost:18011/api/v1/jobs/$JOB_ID" \
  -H "X-API-Key: $API_KEY"
```

**Status flow**: `pending` → `running` → `completed`

### 6. Retrieve Results (2 min)

```bash
curl -X GET "http://localhost:18011/api/v1/jobs/$JOB_ID/results" \
  -H "X-API-Key: $API_KEY"
```

**Expected output**: Pipeline results with:
- Annotation counts
- Processing time
- Output file paths
- Download URLs

**View output files**:
```bash
ls -l reports/$JOB_ID/
# annotations/ - COCO JSON files
# videos/ - Annotated videos
# metadata.json - Job metadata
```

---

## Architecture Overview (5 minutes)

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     VideoAnnotator                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────┐    ┌──────────┐    ┌─────────────┐         │
│  │    CLI    │───→│   API    │───→│   Worker    │         │
│  │  (Typer)  │    │ (FastAPI)│    │ (Executor)  │         │
│  └───────────┘    └──────────┘    └─────────────┘         │
│                          │                 │               │
│                          ↓                 ↓               │
│                   ┌─────────────┐   ┌─────────────┐       │
│                   │  Database   │   │  Pipelines  │       │
│                   │  (SQLite)   │   │  (Registry) │       │
│                   └─────────────┘   └─────────────┘       │
│                                            │               │
│                                            ↓               │
│         ┌────────────────────────────────────────┐        │
│         │  Pipeline Implementations              │        │
│         ├────────────────────────────────────────┤        │
│         │ • OpenFace (face detection/tracking)   │        │
│         │ • Whisper (speech transcription)       │        │
│         │ • PyAnnote (speaker diarization)       │        │
│         │ • Scene Detection                      │        │
│         │ • Emotion Recognition (future)         │        │
│         └────────────────────────────────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Purpose | Location |
|-----------|---------|----------|
| **API** | REST endpoints, authentication, CORS | `src/videoannotator/api/` |
| **Pipelines** | Video processing implementations | `src/videoannotator/pipelines/` |
| **Registry** | Pipeline discovery and metadata | `src/videoannotator/registry/` |
| **Worker** | Job execution, cancellation, retries | `src/videoannotator/worker/` |
| **Database** | Job state, metadata persistence | `src/videoannotator/database/` |
| **Storage** | File management, results organization | `src/videoannotator/storage/` |
| **Validation** | Config and input validation | `src/videoannotator/validation/` |
| **Exporters** | Output format converters | `src/videoannotator/exporters/` |

### Data Flow

```
1. User submits video + pipeline selection via API
2. Job created in database (status: pending)
3. Worker picks up job, validates inputs
4. Pipeline(s) execute on video
5. Results written to storage (COCO, WebVTT, etc.)
6. Job status updated (status: completed)
7. User retrieves results via API
```

### Pipeline Registry

Pipelines are discovered via YAML metadata:

```yaml
# src/videoannotator/registry/metadata/face_openface3_embedding.yaml
name: face_openface3_embedding
display_name: OpenFace3 Face Embedding
tasks:
  - face-embedding
modalities:
  - image
  - video
outputs:
  - format: JSON
    types: [embeddings]
```

---

## Code Exploration (5 minutes)

### Where to Start Reading

**1. API Endpoints** (`src/videoannotator/api/v1/jobs.py`) - 5 min
- REST API implementation
- Job submission, status, results retrieval
- Well-documented with examples

**2. Pipeline Interface** (`src/videoannotator/pipelines/base_pipeline.py`) - 3 min
- Abstract base class for all pipelines
- `initialize()`, `process()`, `cleanup()` methods
- Standard interface contract

**3. Registry System** (`src/videoannotator/registry/pipeline_registry.py`) - 3 min
- Dynamic pipeline discovery
- Metadata loading and validation
- Singleton pattern

**4. Worker Execution** (`src/videoannotator/worker/job_processor.py`) - 5 min
- Job execution logic
- Error handling and retries
- Cancellation support

**5. Tests** (`tests/`) - 10 min
- Unit tests: `tests/unit/`
- Integration tests: `tests/integration/`
- Pipeline tests: `tests/pipelines/`
- 1000+ tests across 74 test files

### Code Quality Indicators

✅ **Type Hints**: Comprehensive type annotations
✅ **Docstrings**: All public APIs documented
✅ **Error Handling**: Consistent error envelope
✅ **Testing**: 1000+ tests across 74 test files
✅ **Logging**: Structured logging throughout
✅ **Configuration**: Environment-based config
✅ **Security**: Secure-by-default (API keys, CORS)

### Testing the Code

```bash
# Run all tests (fast)
uv run pytest -q

# Run with coverage
uv run python scripts/validate_coverage.py --html

# Run specific test file
uv run pytest tests/api/test_api_server.py -v

# Run integration tests
uv run pytest tests/integration/ -v
```

---

## Documentation Quality

### Available Documentation

| Guide | Location | Quality |
|-------|----------|---------|
| Installation | `docs/installation/INSTALLATION.md` | ⭐⭐⭐⭐⭐ |
| Getting Started | `docs/usage/GETTING_STARTED.md` | ⭐⭐⭐⭐⭐ |
| API Reference | Swagger UI `/docs` | ⭐⭐⭐⭐⭐ |
| Security | `docs/security/` | ⭐⭐⭐⭐⭐ |
| Testing | `docs/testing/` | ⭐⭐⭐⭐⭐ |
| Troubleshooting | `docs/installation/troubleshooting.md` | ⭐⭐⭐⭐⭐ |
| Contributing | `CONTRIBUTING.md` | ⭐⭐⭐⭐ |

### JOSS Requirements Met

✅ **Installation instructions**: Clear, tested, verified
✅ **Example usage**: Multiple examples with expected outputs
✅ **API documentation**: Interactive Swagger UI + docstrings
✅ **Community guidelines**: CODE_OF_CONDUCT.md, CONTRIBUTING.md
✅ **Tests**: 1000+ tests, automated CI
✅ **License**: MIT (open source)
✅ **Repository**: Clean, organized, active

---

## Common Reviewer Questions

### Q: How do I add a new pipeline?

**A**: Three steps:

1. Implement `BasePipeline` in `src/videoannotator/pipelines/your_pipeline/`
2. Create YAML metadata in `src/videoannotator/registry/metadata/`
3. Add tests in `tests/pipelines/`

See [Pipeline Specifications](usage/pipeline_specs.md)

### Q: How extensible is the system?

**A**: Highly modular:
- Pipelines: Plugin architecture via registry
- Storage: Configurable backends
- Auth: Pluggable middleware
- Exporters: Format converters

### Q: What about production deployment?

**A**: Multiple options:
- Docker: `Dockerfile.cpu`, `Dockerfile.gpu`
- Kubernetes: Manifests in `docs/deployment/`
- Systemd: Service files provided
- See [Deployment Guide](deployment/Docker.md)

### Q: How is testing organized?

**A**: Three tiers:
- Unit (fast, isolated): `tests/unit/`
- Integration (cross-component): `tests/integration/`
- Pipeline (real models): `tests/pipelines/`

Run selectively with pytest markers.

### Q: What about Windows support?

**A**: Fully supported:
- ASCII-safe output (no emoji in logs/CLI)
- Path handling via `pathlib`
- Tested on Windows 10/11
- Installation verification script checks compatibility

---

## Validation Checklist for Reviewers

Use this checklist for JOSS review:

- [ ] **Installation** (5 min)
  - [ ] Install with `uv sync` succeeds
  - [ ] Verification script passes
  - [ ] API server starts

- [ ] **Functionality** (10 min)
  - [ ] Submit job via API
  - [ ] Job completes successfully
  - [ ] Results retrievable
  - [ ] Output files valid

- [ ] **Documentation** (10 min)
  - [ ] Installation guide clear
  - [ ] API docs comprehensive
  - [ ] Examples work
  - [ ] Troubleshooting helpful

- [ ] **Code Quality** (15 min)
  - [ ] Code readable and well-structured
  - [ ] Type hints present
  - [ ] Docstrings comprehensive
  - [ ] Tests pass
  - [ ] Coverage >80%

- [ ] **Community** (5 min)
  - [ ] Contributing guide present
  - [ ] Code of conduct included
  - [ ] License appropriate (MIT)
  - [ ] Issue templates provided

**Total Time**: ~45 minutes for thorough review

---

## Need Help?

- **Documentation**: Start with [Main README](../README.md)
- **Issues**: [GitHub Issues](https://github.com/InfantLab/VideoAnnotator/issues)
- **Troubleshooting**: [Troubleshooting Guide](installation/troubleshooting.md)
- **Security**: [Security Documentation](security/README.md)

**Happy reviewing! 🎉**
