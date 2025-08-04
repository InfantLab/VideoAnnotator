# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VideoAnnotator is a research-focused video analysis toolkit that processes human interaction videos through modular pipelines. The system uses modern AI models (YOLO11, OpenFace 3.0, Whisper, LAION) to extract comprehensive behavioral annotations in standardized formats compatible with annotation tools like CVAT, LabelStudio, and ELAN.

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

### Essential Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests (currently 94% pass rate)
make test
pytest tests/ -v --cov=src --cov-report=html

# Run specific pipeline tests
pytest tests/test_face_pipeline_modern.py -v
pytest tests/test_person_pipeline_modern.py -v

# Code quality checks
make lint          # Run flake8 linting
make format        # Format with black + isort
make type-check    # Run mypy type checking
make quality-check # Run all quality checks

# Performance testing
pytest tests/ -m performance --benchmark-only

# Docker builds
make docker-build     # CPU version
make docker-build-gpu # GPU version
```

### Demo and Examples
```bash
# Quick demo with sample video
python demo.py

# Individual pipeline testing
python -m src.pipelines.scene_detection.scene_pipeline --input video.mp4
python -m src.pipelines.person_tracking.person_pipeline --input video.mp4

# Example scripts
python examples/basic_video_processing.py
python examples/batch_processing.py
python examples/test_individual_pipelines.py
```

### Batch Processing
```bash
# Process multiple videos
python batch_demo.py --input_dir videos/ --output_dir results/

# Using main CLI
python main.py --input videos/ --batch --parallel 4
```

## Key Technical Details

### Model Management
- Models auto-download on first use to `models/` directory
- OpenFace 3.0 requires separate installation (see requirements_openface.txt)
- LAION models stored in `models/laion_face/` and `models/laion_voice/`
- YOLO models cached in project root

### GPU Acceleration
- PyTorch with CUDA support recommended for 10x speedup
- Install separately: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`
- Automatic device detection (CUDA/MPS/CPU) in pipeline initialization

### Testing Strategy
- Comprehensive test suite in `tests/` directory with 94% success rate
- Separate test files for each pipeline (e.g., `test_face_pipeline_modern.py`)
- Performance benchmarks included with `-m performance` marker
- Integration tests for full pipeline workflows

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

### Testing Requirements
- All new code must include tests in `tests/` directory
- Follow existing naming patterns: `test_{pipeline_name}_modern.py`
- Maintain >90% test success rate
- Include performance benchmarks for compute-intensive operations
- Test both individual components and integration workflows

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

### Testing Debugging
- Use `pytest -v -s` for detailed test output
- Individual pipeline tests can be run in isolation
- Check `htmlcov/` directory for coverage reports after running tests