# VideoAnnotator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A modern, modular toolkit for analyzing, processing, and visualizing human interaction videos with comprehensive video, audio, and annotation workflows. Built with scalable pipeline architecture and support for both local and containerized development.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/InfantLab/VideoAnnotator.git
cd VideoAnnotator

# Install dependencies
pip install -r requirements.txt

# Run a basic video annotation pipeline
python main.py --input video.mp4 --output results/ --config configs/default.yaml
```

For detailed setup instructions, troubleshooting, and advanced configuration, see the [Installation Guide](docs/INSTALLATION.md).

## Features

### Core Pipeline Architecture
- **Scene Detection**: YOLO11-based object detection and CLIP scene classification
- **Person Tracking**: Advanced multi-person tracking with pose estimation
- **Face Analysis**: Multi-modal face detection, recognition, and emotion analysis
- **Audio Processing**: Speech recognition, diarization, and audio feature extraction

### Key Capabilities
- **Modular Design**: Mix and match pipelines for custom workflows
- **Batch Processing**: Process multiple videos efficiently with parallel support
- **Flexible Configuration**: YAML-based configuration system with presets
- **Modern Standards**: Built with latest ML models and best practices
- **Comprehensive Testing**: Full test suite with performance benchmarks
- **Cross-Platform**: Support for Windows, macOS, and Linux environments

## Getting Started

### Basic Usage

```bash
# Process a single video with default settings
python main.py --input video.mp4

# Process with custom configuration
python main.py --input video.mp4 --config configs/high_performance.yaml

# Batch process multiple videos
python main.py --input videos/ --batch --parallel 4

# Run specific pipelines only
python main.py --input video.mp4 --pipelines scene_detection,face_analysis
```

### Pipeline Examples

See the [examples directory](examples/) for detailed usage examples:
- [Basic Video Processing](examples/basic_video_processing.py)
- [Batch Processing](examples/batch_processing.py)
- [Individual Pipeline Testing](examples/test_individual_pipelines.py)
- [Custom Pipeline Configuration](examples/custom_pipeline_config.py)

### Configuration

The system uses YAML configuration files for flexible pipeline setup:
- `configs/default.yaml` - Balanced settings for most use cases
- `configs/high_performance.yaml` - Maximum accuracy, slower processing
- `configs/lightweight.yaml` - Fast processing, reduced accuracy

See [Configuration Guide](configs/README.md) for detailed configuration options.

## Architecture

### Pipeline Structure

```
src/pipelines/
├── base_pipeline.py          # Base pipeline interface
├── scene_detection/          # Scene analysis and object detection
├── person_tracking/          # Multi-person tracking and pose estimation
├── face_analysis/            # Face detection, recognition, emotions
└── audio_processing/         # Speech recognition and audio features
```

### Data Schemas

All pipelines output standardized data formats with comprehensive metadata:
- Scene annotations with object detection and classification
- Person tracking with pose keypoints and movement analysis
- Face analysis with detection, recognition, and emotion scores
- Audio processing with speech transcription and speaker diarization

See [schemas documentation](src/schemas/) for detailed output formats.

## Documentation

### Core Documentation
- [Getting Started Guide](docs/GETTING_STARTED.md)
- [Installation & Setup Guide](docs/INSTALLATION.md)
- [Configuration Guide](configs/README.md)
- [Testing Standards](docs/TESTING_STANDARDS.md)

### API Documentation
- [Data Schemas & Output Formats](docs/OUTPUT_FORMATS.md)
- [Technical Specification](docs/TECHNICAL_SPECIFICATION.md)
- [Development Roadmap](docs/ROADMAP.md)
- [Troubleshooting Guide](docs/TROUBLESHOOTING.md)

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_pipelines.py -k "test_integration"
python -m pytest tests/test_pipelines.py -k "test_performance"

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## Contributing

1. Follow the [Testing Standards](docs/TESTING_STANDARDS.md) for all contributions
2. Add tests to existing test files in `tests/` directory
3. Update documentation for new features
4. Run the full test suite before submitting changes

## Requirements

- Python 3.8+
- FFmpeg for audio/video processing
- CUDA (optional, for GPU acceleration)
- See [requirements.txt](requirements.txt) for full dependencies

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

VideoAnnotator builds upon research methodologies from computational video analysis and machine learning communities. Special thanks to the open-source ML ecosystem including YOLO, OpenAI Whisper, and PyTorch.
