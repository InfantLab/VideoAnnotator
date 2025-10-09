# VideoAnnotator Documentation

## Current Release: v1.2.0 ✨
**Major breakthrough achieved!** Complete integrated background job processing, modern FastAPI server, and production-ready video processing API.

## Next Development: v1.2.1 (Polish & Documentation Updates)

This documentation is organized into clear sections for different user types and development phases.

## 📖 User Documentation

### Installation
- [Installation Guide](installation/INSTALLATION.md) - **Modern uv-based setup** with CUDA support  
- macOS specifics: see the macOS section inside the Installation Guide for libomp, Node, ffmpeg, and PATH fixes.
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

### Current Release (v1.2.0)
- [v1.2.0 Release Summary](development/v1.2.0_RELEASE_SUMMARY.md) - Complete release overview and achievements
- [v1.2.0 Roadmap](development/roadmap_v1.2.0.md) - Development roadmap and completed milestones
- [API Upgrade Guide](development/api_upgrade_v1.2.0.md) - Major API changes implemented in v1.2.0
- [Database Implementation](development/database_implementation_plan_v1.2.0.md) - Database architecture

### Active Development & Roadmap
- **[Roadmap Overview](development/roadmap_overview.md)** - Complete release strategy v1.3.0 → v2.0.0
- [v1.3.0 Roadmap](development/roadmap_v1.3.0.md) - **CURRENT**: Production reliability & critical fixes (Q4 2025)
- [v1.4.0 Roadmap](development/roadmap_v1.4.0.md) - **NEXT**: First public release + JOSS paper (Q2 2026)
- [v1.2.1 Roadmap](development/roadmap_v1.2.1.md) - Polish and documentation updates (completed)
- [Examples CLI Update Plan](development/EXAMPLES_CLI_UPDATE_CHECKLIST.md) - CLI modernization checklist

## 🧪 Testing & QA

### Current Testing (v1.2.0)
- [v1.2.0 QA Checklist](testing/qa_checklist_v1.2.0.md) - Complete QA validation for v1.2.0 release
- [Testing Overview](testing/testing_overview.md) - Complete testing strategy and results  
- [Testing Standards](testing/testing_standards.md) - Quality assurance standards
- [Batch Testing Guide](testing/batch_testing_guide.md) - Batch processing test procedures

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

- **v1.2.0** (Current Release) - Production-ready API with complete integrated video processing
- **v1.2.1** (Next Development) - Documentation updates and CLI modernization
- **v1.3.0** (Future Release) - Advanced features, security, and scalability enhancements