# VideoAnnotator Documentation

## Current Release: v1.2.0 ✨
**Major modernization complete!** Now using uv package manager, Ruff linting, FastAPI server, and modern Python 3.12+ development workflow.

## Next Development: v1.2.1

This documentation is organized into clear sections for different user types and development phases.

## 📖 User Documentation

### Installation
- [Installation Guide](installation/INSTALLATION.md) - **Modern uv-based setup** with CUDA support  
- [Environment Setup](installation/ENVIRONMENT_SETUP.md) - HuggingFace and configuration  
- [Python Development 2025](installation/PythonDev2025.md) - **Modern development practices** with uv, Ruff, and Docker

### Usage  
- [Getting Started](usage/GETTING_STARTED.md) - Quick start guide for new users
- [Pipeline Specifications](usage/pipeline_specs.md) - Detailed pipeline documentation
- [Scene Detection Guide](usage/scene_detection.md) - Scene detection usage
- [Scene Detection Guide (Advanced)](usage/scene_detection_guide.md) - Advanced scene detection usage
- [Demo Commands](usage/demo_commands.md) - Example commands and workflows
- [Output Formats](usage/output_formats.md) - Understanding output data formats  
- [Troubleshooting](usage/troubleshooting.md) - Common issues and solutions
- [Troubleshooting Guide](usage/troubleshooting_guide.md) - Extended troubleshooting

### Deployment
- [Docker Guide](deployment/docker.md) - Container deployment
- [Docker Setup](deployment/Docker.md) - Docker configuration
- [Docker Guide Extended](deployment/Docker_Guide.md) - Advanced Docker usage

## 🔧 Development Documentation

### Current Development (v1.1.2)
- [Batch Processing Plan](development/batch_processing_plan_v1.1.2.md) - Batch processing improvements
- [Diarization Implementation](development/diarization_implementation_v1.1.2.md) - Audio diarization features
- [Diarization Pipeline](development/diarization_pipeline_v1.1.2.md) - Pipeline implementation details
- [Person ID Implementation](development/person_id_implementation_v1.1.2.md) - Person identification features
- [Size Based Analysis](development/size_based_analysis_v1.1.2.md) - Size-based person analysis

### Next Release (v1.2.0)
- [API Upgrade](development/api_upgrade_v1.2.0.md) - Major API changes for v1.2.0
- [LAION Implementation Plan](development/laion_implementation_plan_v1.2.0.md) - LAION model integration
- [Roadmap](development/roadmap_v1.2.0.md) - Long-term development roadmap

## 🧪 Testing & QA

### Current Testing (v1.1.1)
- [Testing Overview](testing/testing_overview_v1.1.1.md) - Complete testing strategy and results
- [Testing Standards](testing/testing_standards_v1.1.1.md) - Quality assurance standards
- [Batch Testing Guide](testing/batch_testing_guide_v1.1.1.md) - Batch processing test procedures

## 📁 Archive

Historical and outdated documentation for reference:
- [Archive Folder](archive/) - Contains superseded documentation

## 📋 Document Lifecycle

### Active Development
Documents in `development/` are tagged with version numbers and represent current development work:
- `feature_name_v1.1.2.md` - Current development cycle
- `feature_name_v1.2.0.md` - Next major release

### Testing & QA  
Documents in `testing/` follow the same versioning pattern:
- `test_plan_v1.1.1.md` - Current release testing
- `qa_checklist_v1.1.2.md` - Development cycle QA

### Completion & Archive
When development cycles complete:
1. Current development docs move to `archive/`
2. Next version becomes current 
3. New development docs created for future versions

## 🔄 Version Status

- **v1.1.1** (Current Release) - Stable production version
- **v1.1.2** (Current Development) - Bug fixes and minor enhancements
- **v1.2.0** (Next Release) - Major API upgrades and new features