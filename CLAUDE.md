# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VideoAnnotator is a research-focused video analysis toolkit that processes human interaction videos through modular pipelines. The system uses modern AI models (YOLO11, OpenFace 3.0, Whisper, LAION) to extract comprehensive behavioral annotations in standardized formats compatible with annotation tools like CVAT, LabelStudio, and ELAN.

**Current Version**: 1.2.0 (2025-08-22) - Major modernization: migrated to uv package manager, Ruff linting/formatting, FastAPI server, and modern Python 3.12+ development workflow.

## Core Architecture

### Pipeline System
The codebase is built around a modular pipeline architecture where each pipeline inherits from `BasePipeline` (src/pipelines/base_pipeline.py):

- **Scene Detection**: PySceneDetect + CLIP for environment classification  
- **Person Tracking**: YOLO11 + ByteTrack for multi-person pose tracking
- **Face Analysis**: Multiple backends (OpenFace 3.0, LAION Face, OpenCV) for facial behavior analysis
- **Audio Processing**: Whisper + pyannote.audio for speech recognition and diarization

Each pipeline follows the same interface:
- `initialize()` - Load models and resources
- `process(video_path, start_time, end_time, pps, output_dir)` - Process video segments
- `cleanup()` - Release resources
- `get_schema()` - Return output JSON schema

### Configuration System
Uses YAML-based configuration with environment-specific configs in `configs/`:
- `default.yaml` - Standard settings
- `lightweight.yaml` - CPU-optimized processing
- `high_performance.yaml` - GPU-accelerated with all features
- `openface3.yaml` - OpenFace 3.0 comprehensive analysis

### Output Format Strategy
Pipelines output native industry formats rather than custom schemas:
- Person tracking → COCO format with keypoints
- Face analysis → COCO format with facial landmarks and emotions
- Audio → WebVTT for transcripts, RTTM for diarization
- Scene detection → Simple timestamped JSON arrays

## Development Commands

### Essential Commands (uv-based workflow)
```bash
# Install dependencies
uv sync                                         # Install all dependencies
uv sync --extra dev                             # Install with dev dependencies

# Run tests - 3-tier system (83.2% success rate)
# Fast development feedback (recommended)
uv run python scripts/test_fast.py             # ~30 seconds, 125+ unit tests

# Pre-commit validation  
uv run python scripts/test_integration.py      # ~5 minutes, unit + integration

# Complete validation
uv run python scripts/test_all.py              # Full suite with reporting

# Modern pytest commands
uv run pytest tests/ -v --cov=src --cov-report=html

# Run specific test tiers
uv run pytest tests/unit/ -v                   # Unit tests only
uv run pytest tests/integration/ -v            # Integration tests
uv run pytest tests/pipelines/ -v              # Pipeline tests

# Run by markers
uv run pytest -m unit                          # Unit tests
uv run pytest -m integration                   # Integration tests
uv run pytest -m pipeline                      # Pipeline tests

# Code quality checks (Ruff-based)
uv run ruff check .                             # Run Ruff linting
uv run ruff format .                            # Format with Ruff
uv run mypy src                                 # Run mypy type checking

# Performance testing
uv run pytest tests/ -m performance --benchmark-only

# Docker builds (modern)
docker build -f Dockerfile.cpu -t videoannotator:cpu .      # CPU version
docker build -f Dockerfile.gpu -t videoannotator:gpu .      # GPU version
```

### Demo and Examples
```bash
# Quick demo with sample video
uv run python demo.py

# API server
uv run python api_server.py                    # Start API server
uv run uvicorn api_server:app --reload         # Development server with auto-reload

# Individual pipeline testing
uv run python -m src.pipelines.scene_detection.scene_pipeline --input video.mp4
uv run python -m src.pipelines.person_tracking.person_pipeline --input video.mp4

# Example scripts
uv run python examples/basic_video_processing.py
uv run python examples/batch_processing.py
uv run python examples/test_individual_pipelines.py
```

### Batch Processing
```bash
# Process multiple videos
uv run python batch_demo.py --input_dir videos/ --output_dir results/

# Using main CLI
uv run python main.py --input videos/ --batch --parallel 4
```

## Key Technical Details

### Model Management & Directory Structure
- **Organized Model Storage**: All models auto-download to organized `models/` directory structure:
  - `models/yolo/` - YOLO pose estimation models (yolo11n-pose.pt, yolo11m-pose.pt)
  - `models/laion_face/` - LAION face emotion analysis models
  - `models/laion_voice/` - LAION voice emotion analysis models  
  - `models/whisper/` - Whisper speech recognition models
  - `models/openface/` - OpenFace 3.0 models (separate installation required)
- **Traditional Models**: Face analysis weights in `weights/` directory (RetinaFace, landmarks)
- **Pre-initialization**: All pipelines initialize during VideoAnnotator startup for optimal batch processing performance
- **Error Recovery**: Automatic model reinitialization when corruption is detected
- **Auto-configuration**: Ultralytics YOLO automatically uses `models/yolo/` as default weights directory

### GPU Acceleration
- PyTorch with CUDA support recommended for 10x speedup
- Install separately: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`
- Automatic device detection (CUDA/MPS/CPU) in pipeline initialization
- **Memory Management**: Automatic GPU cache clearing and resource cleanup to prevent memory leaks
- **Meta Tensor Handling**: Robust PyTorch model loading with `to_empty()` fallback for newer PyTorch versions

### Testing Strategy - 3-Tier System
- **Comprehensive 3-tier test organization** in `tests/` directory (see TESTING_OVERVIEW.md):
  - `tests/unit/` - Fast isolated tests (<30 seconds, 125+ tests)
  - `tests/integration/` - Cross-component tests (~5 minutes)
  - `tests/pipelines/` - Full pipeline tests with real models
- **Fast Development Workflow**: `python scripts/test_fast.py` for immediate feedback
- **Tiered Execution Scripts**: 
  - `scripts/test_fast.py` - Unit tests only (~30 seconds)
  - `scripts/test_integration.py` - Unit + integration tests (~5 minutes)
  - `scripts/test_all.py` - Complete suite with reporting
- **High Success Rate**: 83.2% stable success rate across all test tiers
- **Pipeline Coverage**: 100% person tracking, 93.3% face analysis, comprehensive audio/scene
- **Performance Benchmarks**: `-m performance` marker for benchmarking tests
- **Real Model Integration**: Environment-controlled integration tests with actual AI models

### Person Identity System
The project includes a sophisticated person identity system:
- `src/utils/person_identity.py` - Core identity matching logic
- `src/utils/automatic_labeling.py` - Automated labeling workflows
- Configuration via `configs/person_identity.yaml`
- Integration with person tracking pipeline for persistent IDs across videos

## Development Guidelines

### Adding New Pipelines
1. Inherit from `BasePipeline` in `src/pipelines/base_pipeline.py`
2. Implement required methods: `initialize()`, `process()`, `cleanup()`, `get_schema()`
3. Add configuration schema to appropriate config files
4. Create comprehensive tests following existing patterns
5. Update documentation and examples

### Testing Requirements & Guidelines
- **All new code must include tests** following the 3-tier system:
  - Unit tests → `tests/unit/[component]/` (fast, isolated, <1s per test)
  - Integration tests → `tests/integration/` (cross-component, <5min total)
  - Pipeline tests → `tests/pipelines/` (full workflows, real models)
- **Naming Conventions**: `test_{component}_{functionality}.py`
- **Quality Standards**: Maintain >83% test success rate across all tiers
- **Performance Benchmarks**: Include `-m performance` markers for compute-intensive operations
- **Pytest Markers**: Use appropriate markers (`@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.pipeline`)
- **Test Organization**: Place tests in appropriate tier based on scope and dependencies
- **Environment Controls**: Use `TEST_INTEGRATION=1` for enabling real model testing

### Code Quality Standards
- Use Black for code formatting (line length 88)
- Follow type hints (enforced by mypy)
- Pass flake8 linting checks
- Include comprehensive docstrings
- Follow existing architectural patterns

### Output Format Compliance
- Use industry-standard formats (COCO, WebVTT, RTTM)
- Include comprehensive metadata with version tracking
- Ensure annotation tool compatibility (CVAT, LabelStudio, ELAN)
- Validate output schemas against specifications

## Common Issues and Solutions

### Environment Setup
- Python 3.8-3.12 supported (3.13 has dependency conflicts)
- FFmpeg required for audio/video processing
- OpenFace 3.0 requires additional setup (see docs/OPENFACE3_GUIDE.md)

### Model Downloads
- Ensure stable internet for first-time model downloads
- Models require ~2GB disk space total
- Check model paths in config files if experiencing loading issues

### GPU Memory Management
- Large videos may require processing in segments
- Adjust batch sizes in config files for available GPU memory
- Use `lightweight.yaml` config for resource-constrained environments
- **v1.1.1**: Improved memory management with automatic cleanup and CUDA cache clearing

### Batch Processing Issues (v1.1.1 Fixes)
- **Clean Logs**: Verbose debug output from ByteTracker, YOLO, and numba is now suppressed
- **Model Errors**: PyTorch meta tensor errors automatically handled with fallback loading
- **Corruption Recovery**: Person pipeline automatically recovers from model corruption
- **Performance**: Pre-initialized pipelines provide significant speed improvements for batch processing

### Testing Debugging & Advanced Usage
- **Development Workflow**: Use `python scripts/test_fast.py` for immediate feedback during development
- **Pre-commit Validation**: Use `python scripts/test_integration.py` before committing changes
- **Detailed Output**: `pytest -v -s` for verbose test output with print statements
- **Coverage Reports**: Generated in `htmlcov/` directory after running coverage tests
- **Pipeline-Specific Testing**:
  ```bash
  pytest tests/pipelines/test_person_tracking.py -v
  pytest tests/pipelines/test_face_analysis.py -v
  TEST_INTEGRATION=1 pytest tests/pipelines/ -v  # Enable real model tests
  ```
- **Test Categories**:
  ```bash
  pytest -m unit                              # Fast isolated tests
  pytest -m integration                       # Cross-component tests  
  pytest -m performance --benchmark-only     # Performance benchmarks
  pytest -k "person_tracking"                # Pattern matching
  ```
- **Environment Variables**: Set `TEST_INTEGRATION=1` for real model testing
- **Known Issues**: Size analysis integration test may fail (functionality under development)
- **Package Names**: openface3 pip package is called `openface-test`

## Documentation Organization

This project follows a systematic documentation structure in the `docs/` folder:

### **Canonical User Documentation** (No Version Suffixes)
- **`docs/installation/`** - Setup and installation guides that are maintained and updated as functionality evolves
- **`docs/usage/`** - Reference documentation (pipeline_specs.md, demo_commands.md, etc.) - these are living documents that reflect current system capabilities
- **`docs/deployment/`** - Deployment and containerization guides

### **Versioned Development Documentation** 
- **`docs/development/`** - Active development plans and implementation docs with version suffixes:
  - `feature_name_v1.1.2.md` - Current development cycle
  - `feature_name_v1.2.0.md` - Next major release planning
- **`docs/testing/`** - QA checklists and testing documentation with version suffixes:
  - `testing_plan_v1.1.1.md` - Current release testing
  - `qa_checklist_v1.1.2.md` - Development cycle QA

### **Document Lifecycle Management**
1. **Active Development**: Documents in development/ and testing/ are tagged with version numbers
2. **Completion**: When development cycles complete, versioned docs move to `docs/archive/`
3. **User Docs Evolution**: Usage docs like `pipeline_specs.md` are updated in place to reflect new functionality
4. **Navigation**: `docs/README.md` provides comprehensive index of all documentation

### **Version Tracking**
- **Current Release**: v1.1.1 (stable production)
- **Current Development**: v1.1.2 (bug fixes and enhancements) 
- **Next Release**: v1.2.0 (major API changes and new features)

This approach ensures users always have current reference documentation while development work is clearly tracked by version cycle.