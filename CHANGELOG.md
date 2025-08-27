# Changelog

All notable changes to VideoAnnotator will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Enhanced pipeline configuration system
- Advanced batch processing optimizations
- Extended annotation tool integration
- Multi-language CLI support

## [1.2.0] - 2025-08-26

### 🚀 Major Features - Production-Ready API System

#### Added
- **🎯 Modern FastAPI Server**: Complete REST API with interactive documentation at `/docs`
- **⚡ Integrated Background Processing**: Built-in job processing system - no separate worker processes needed
- **🛠️ Modern CLI Interface**: Comprehensive `uv run videoannotator` command-line tools for server and job management
- **📊 Real-time Job Status**: Live job tracking with detailed progress updates and results retrieval
- **🔄 Async Job Processing**: Handle multiple video processing jobs simultaneously
- **🌐 Cross-platform API**: RESTful endpoints compatible with Python, JavaScript, R, and any HTTP client

#### Enhanced Architecture
- **🏗️ API-First Design**: All pipelines accessible through standardized HTTP endpoints
- **📋 Job Management System**: Complete job lifecycle with submit → status → results workflow  
- **🔧 Configuration API**: Validate and manage pipeline configurations via API
- **📁 File Management**: Secure video upload, processing, and result file downloads
- **🔐 Authentication Ready**: JWT token infrastructure for secure API access

#### Modern Development Stack
- **📦 uv Package Manager**: Migrated from pip to uv for 10x faster dependency management
- **🧹 Ruff Integration**: Modern linting and formatting with Ruff (replaces Black, isort, flake8)
- **🐳 Fixed Docker Support**: Resolved build issues with proper file copying and modern license formats
- **📖 DeepWiki Integration**: Interactive documentation available at deepwiki.com/InfantLab/VideoAnnotator

### 🛠️ API Endpoints & Usage

#### Core Job Management
```bash
# Submit video processing job
POST /api/v1/jobs/
# Monitor job status  
GET /api/v1/jobs/{job_id}
# Retrieve detailed results
GET /api/v1/jobs/{job_id}/results
# Download specific pipeline outputs
GET /api/v1/jobs/{job_id}/results/files/{pipeline}
```

#### System Management
```bash
# Health check and server info
GET /health
GET /api/v1/debug/server-info
# List available pipelines
GET /api/v1/pipelines
# Configuration validation
POST /api/v1/config/validate
```

#### Modern CLI Commands
```bash
# Start integrated API server
uv run videoannotator server --port 8000

# Job management via CLI
uv run videoannotator job submit video.mp4 --pipelines scene,person,face
uv run videoannotator job status <job_id>  
uv run videoannotator job results <job_id>
uv run videoannotator job list --status completed

# System information
uv run videoannotator info
uv run videoannotator pipelines --detailed
```

### 📚 Documentation & User Experience

#### Updated Documentation
- **📖 Complete Documentation Refresh**: Updated all docs for v1.2.0 with modern API patterns
- **🧭 Navigation System**: Added consistent navigation bars across all documentation files
- **🎮 Interactive Examples**: Updated demo_commands.md with modern CLI and API usage patterns  
- **🔗 Cross-references**: Fixed all internal documentation links with proper relative paths
- **📋 API Reference**: Complete API documentation with request/response examples

#### Migration from Legacy Patterns
- **Replaced**: Old `python demo.py` patterns → Modern `uv run videoannotator` CLI
- **Updated**: Direct pipeline usage → API-first architecture examples
- **Enhanced**: Configuration examples with modern YAML structure
- **Improved**: Getting started guide with 30-second setup process

### 🔧 Technical Improvements

#### Development Workflow
- **⚡ Fast Package Management**: uv provides 10-100x faster dependency resolution
- **🧹 Unified Tooling**: Single Ruff command replaces multiple linting/formatting tools
- **🏗️ Modern Build System**: Updated pyproject.toml with modern license format and dependency groups
- **🐳 Container Optimization**: Fixed Docker builds with proper source file copying

#### Infrastructure
- **🔄 Integrated Processing**: Background job processing runs within API server process
- **📊 Status Tracking**: Real-time job status updates with detailed pipeline progress
- **🗄️ Database Integration**: SQLite-based job storage with full CRUD operations
- **🔐 Security Framework**: JWT authentication ready for production deployment

### 🛡️ Compatibility & Migration

#### Breaking Changes
- **CLI Interface**: Legacy `python demo.py` replaced with `uv run videoannotator` commands
- **Configuration**: Updated to API-first workflow - direct pipeline usage now for development only
- **Dependencies**: Requires uv package manager for optimal performance

#### Migration Path
```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/Mac
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Update existing installation
uv sync  # Fast dependency installation
uv sync --extra dev  # Include development dependencies

# Start using modern API server
uv run videoannotator server  # Replaces old direct processing
```

#### Backward Compatibility
- **✅ Pipeline Architecture**: All pipelines remain fully functional with same output formats
- **✅ Configuration Files**: Existing YAML configs work with new API system
- **✅ Output Formats**: JSON schemas unchanged - existing analysis code continues working
- **✅ Docker Support**: Updated containers with same functionality

### 🎯 Production Readiness

#### Deployment Features  
- **🚀 Single Command Startup**: `uv run videoannotator server` starts complete system
- **📊 Health Monitoring**: Built-in health endpoints for system monitoring
- **🔄 Graceful Shutdowns**: Proper cleanup of background processes and resources
- **📱 API Documentation**: Auto-generated OpenAPI/Swagger documentation
- **🐳 Container Support**: Fixed Docker builds for both CPU and GPU deployment

#### Performance & Reliability
- **⚡ Fast Startup**: Models load on-demand, reducing initial startup time  
- **🔄 Concurrent Processing**: Handle multiple video jobs simultaneously
- **💾 Resource Management**: Proper cleanup prevents memory leaks
- **🛡️ Error Recovery**: Robust error handling with detailed status reporting

### 🧪 Quality Assurance

#### Testing & Validation
- **✅ Comprehensive API Testing**: Full test coverage for job management and processing workflows
- **✅ Integration Testing**: End-to-end tests with real video processing
- **✅ Docker Validation**: Verified container builds and deployments  
- **✅ Documentation Accuracy**: All examples tested and validated for v1.2.0

#### Development Standards
- **🧹 Modern Code Quality**: Ruff-based linting and formatting with consistent style
- **📋 Type Safety**: Maintained mypy type checking across codebase
- **📊 Test Coverage**: High test coverage maintained across API and processing layers

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
