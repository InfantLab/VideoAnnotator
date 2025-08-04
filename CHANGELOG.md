# Changelog

All notable changes to VideoAnnotator will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Future development

## [1.1.1] - 2025-08-04

### Fixed
- **PyTorch Meta Tensor Errors**: Fixed "Cannot copy out of meta tensor" errors in face analysis and audio pipelines by implementing proper `to_empty()` fallback handling
- **Person Pipeline Model Corruption**: Added robust error recovery for "'Conv' object has no attribute 'bn'" errors with automatic model reinitialization
- **Batch Processing Stability**: Enhanced error handling and recovery mechanisms across all pipelines

### Improved
- **Logging System**: Suppressed verbose debug output from ByteTracker, YOLO, and numba for cleaner batch processing logs
- **Performance Optimization**: Pre-initialize all pipelines during setup instead of lazy loading for each video, significantly improving batch processing speed
- **GPU Memory Management**: Added proper cleanup methods with CUDA cache clearing and resource management
- **Error Recovery**: Implemented automatic model reinitialization when corruption is detected during processing

### Changed
- **Pipeline Initialization**: Models now load once during VideoAnnotator initialization rather than per-video for better batch performance
- **Memory Management**: Added destructor and cleanup methods to prevent GPU memory leaks during batch processing

## [1.1.0] - 2025-08-04

### Added - PersonID System
- **PersonIdentityManager** for consistent person identification across pipelines
- **Automatic labeling system** with size-based and spatial heuristics for person role detection
- **Face-to-person linking** across all face analysis pipelines using IoU matching
- **Person identity configuration** via `configs/person_identity.yaml`
- **Comprehensive test suite** for person identity functionality in `tests/test_phase2_integration.py`
- **Command-line tools** for person labeling and validation in `scripts/` directory

### Added - OpenFace 3.0 Integration
- **OpenFace 3.0 pipeline** with comprehensive facial behavior analysis
- **98-point facial landmarks** (2D and 3D coordinates)
- **Facial Action Units (AUs)** intensity and presence detection
- **Head pose estimation** with rotation and translation parameters
- **Gaze direction tracking** and eye movement analysis
- **COCO format output** for annotation tool compatibility
- **Demo scripts** showcasing full OpenFace 3.0 capabilities

### Added - LAION Face & Voice Pipelines
- **LAION Face pipeline** with CLIP-based face analysis and emotion detection
- **LAION Voice pipeline** with advanced voice emotion recognition
- **40+ emotion categories** for comprehensive emotional analysis
- **Multimodal emotion analysis** combining face and voice modalities
- **High-precision embeddings** for research applications

### Enhanced
- **All face analysis pipelines** now support person identity linking
- **Person tracking pipeline** exports consistent person IDs in COCO format
- **Cross-pipeline data sharing** through standardized person tracks files
- **COCO format compliance** with industry-standard annotation fields
- **Configuration system** extended with person identity settings
- **Testing framework** enhanced with integration and performance tests

### Changed
- **Documentation consolidation**: PersonID phase completion files merged into main documentation
- **File organization**: Legacy backup files and duplicates removed
- **Test structure**: All tests properly organized in `tests/` directory with pytest framework

### Fixed
- **Legacy file cleanup**: Removed backup files and duplicates (`speech_pipeline_backup.py`, etc.)
- **Documentation consistency**: Updated all docs to reflect current implementation status
- **Test organization**: Moved standalone test files to proper test directory structure

## [1.0.0] - 2025-01-09

### Added
- Initial release of modernized VideoAnnotator
- Complete pipeline architecture implementation
- Comprehensive documentation and examples
- Full testing suite with unit, integration, and performance tests
- Docker support for development and production
- CI/CD pipeline with automated testing and deployment

## [0.3.0] - 2024-12-01 (Legacy)

### Added
- Basic video annotation capabilities
- Jupyter notebook examples
- Initial audio processing features

### Changed
- Improved video processing performance
- Updated dependencies

### Fixed
- Various bug fixes and stability improvements

## [0.2.0] - 2024-10-01 (Legacy)

### Added
- Face detection and analysis
- Person tracking capabilities
- Data visualization tools

### Changed
- Refactored code organization
- Updated documentation

### Fixed
- Memory usage optimization
- Cross-platform compatibility

## [0.1.0] - 2024-08-01 (Legacy)

### Added
- Initial project structure
- Basic video processing
- Scene detection capabilities
- Audio extraction
- Data annotation framework

---

## Migration Guide

### From v0.x to v1.0.0

The v1.0.0 release introduces significant architectural changes. Here's how to migrate:

#### Configuration Changes

**Old (v0.x):**
```python
# Direct pipeline initialization
from src.processors.video_processor import VideoProcessor
processor = VideoProcessor(config_dict)
```

**New (v1.0.0):**
```python
# Modern pipeline architecture
from src.pipelines import SceneDetectionPipeline
pipeline = SceneDetectionPipeline(config)
```

#### API Changes

**Old:**
```python
# Direct method calls
results = processor.process_video(video_path)
```

**New:**
```python
# Standardized pipeline interface
results = pipeline.process(video_path, start_time=0, end_time=None)
```

#### Configuration Format

**Old:**
```python
# Python dictionary configuration
config = {
    'video_settings': {'fps': 30},
    'audio_settings': {'sample_rate': 16000}
}
```

**New:**
```yaml
# YAML configuration
video:
  fps: 30
audio:
  sample_rate: 16000
```

#### CLI Changes

**Old:**
```bash
python process_video.py --video video.mp4 --output output/
```

**New:**
```bash
python main.py --input video.mp4 --output output/ --config configs/default.yaml
```

### Breaking Changes

1. **Pipeline Architecture**: Complete rewrite of processing pipelines
2. **Configuration System**: Moved from Python dictionaries to YAML files
3. **CLI Interface**: New unified command-line interface
4. **Output Formats**: Standardized output schemas
5. **Dependencies**: Updated to modern ML libraries

### Deprecation Notices

- Legacy processor classes will be removed in v2.0.0
- Python dictionary configuration deprecated in favor of YAML
- Old CLI scripts will be removed in v2.0.0

### Upgrade Path

1. **Update Dependencies**: `pip install -r requirements.txt`
2. **Convert Configuration**: Use new YAML format
3. **Update Code**: Migrate to new pipeline architecture
4. **Test Integration**: Run comprehensive tests
5. **Update Documentation**: Review API changes

For technical specifications, see the [Pipeline Specs](docs/Pipeline%20Specs.md).

---

## Contributors

Special thanks to all contributors who helped shape VideoAnnotator:

### Core Team
- Development Team - Core architecture and implementation
- Research Team - Algorithm development and optimization
- Documentation Team - Comprehensive documentation and examples

### Community Contributors
- Bug reports and feature requests
- Code contributions and improvements
- Documentation improvements
- Testing and validation

### Acknowledgments

This project builds upon the excellent work of:
- [BabyJokes](https://github.com/InfantLab/babyjokes) - Original research foundation
- Open source computer vision and machine learning communities
- Contributors to the libraries and tools we depend on

---

For more information about releases and changes, see the [GitHub Releases](https://github.com/your-org/VideoAnnotator/releases) page.
