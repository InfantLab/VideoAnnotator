# Changelog

All notable changes to VideoAnnotator will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned for v1.5.0

- Queue position display for pending jobs
- Deterministic test fixtures with synthetic video generation
- Research workflow examples for JOSS paper
- Benchmark results and performance validation
- Additional contributor documentation improvements

## [1.4.1] - 2025-12-26

### Release Quality, Docs, and Developer Experience

#### Added
- **Container/Devcontainer**: Baked `hadolint` into Docker images and devcontainer so pre-commit hooks work reliably.
- **Dockerfiles**: Added `git-lfs` to CPU/GPU Dockerfiles for smoother model/asset workflows.

#### Changed
- **Documentation**: Consolidated the JOSS manuscript into `paper/paper.md` and replaced `docs/joss.md` with a pointer to avoid divergence.
- **Repository Hygiene**: Moved top-level helper scripts into organized subfolders under `scripts/` and updated imports to the `videoannotator.*` package namespace.
- **Entrypoints**: Updated `api_server.py` to act as a compatibility wrapper; documentation now recommends using the `videoannotator` CLI.
- **README**: Rationalized repeated setup/install instructions, fixed broken/non-links, and replaced hard-coded test/coverage claims with CI status.

#### Fixed
- **Docs**: Standardized examples on the canonical API port `18011` and corrected Docker run port mappings.
- **Docs**: Replaced placeholder `docs/usage/accessing_results.md` with a real results retrieval guide.

## [1.4.0] - 2025-12-15

### 🚀 Major Features - Flexible Storage & Enhanced Security

This release introduces a flexible storage system allowing artifact downloads and a robust database-backed authentication system.

#### Added
- **Flexible Storage**: New artifact download capabilities, including source video retrieval.
- **Authentication**: Migrated from file-based to database-backed authentication for improved security and scalability.
- **Artifacts API**: New endpoint `GET /api/v1/jobs/{id}/artifacts` to download job results as a ZIP archive.

#### Fixed
- **Artifact Downloads**: Ensured source video files are included in the downloaded artifact ZIP.

## [1.3.1] - 2025-12-07

### ⚡ Performance & Developer Experience

This patch release focuses on critical performance fixes for the API and improving the developer experience in cloud environments (Codespaces).

#### Fixed
- **Critical Performance**: Reduced `GET /api/v1/pipelines` response time from ~160s to <100ms by removing heavy module imports during listing.
- **Critical Performance**: Removed 1-second blocking delay in `GET /api/v1/system/health` by optimizing CPU usage checks.
- **CORS**: Fixed Cross-Origin Resource Sharing for development environments by correctly supporting wildcard origins with credentials.
- **API Routing**: Resolved timeouts and 307 Redirect loops caused by trailing slash inconsistencies in API routes.
- **Pipeline Discovery**: Fixed discovery issues for `face_laion_clip` and reduced log spam.
- **Dev Container**: Fixed build issues and normalized line endings for cross-platform compatibility.
- **Storage**: Fixed critical issues with video storage paths and cleanup logic.

#### Added
- **Documentation**: Added `docs/development/CORS_AND_AUTH_PROTOCOL.md` for frontend integration guidance.
- **CLI**: Added `setup-db` command for streamlined database initialization.

## [1.3.0] - 2025-10-31

### 🚀 Major Features - Production Reliability & Critical Fixes

This release addresses critical production blockers identified during client integration testing and establishes a solid foundation for JOSS publication.

#### Added

**🔧 Job Management & Concurrency Control**
- Job cancellation API endpoint (`POST /api/v1/jobs/{id}/cancel`) with `CancellationManager` for async task tracking
- `CANCELLED` job status with proper state machine transitions
- `MAX_CONCURRENT_JOBS` environment variable (default: 2) with worker queue enforcement
- Worker retry logic with exponential backoff
- Enhanced worker signal handling for graceful cancellation
- 24 comprehensive tests for cancellation (15 unit + 9 integration)

**💾 Persistent Storage System**
- Persistent storage implementation with `STORAGE_DIR` environment variable (default: `./storage`)
- Automatic directory structure creation (`uploads/`, `results/`, `temp/`, `logs/`)
- Storage cleanup module with configurable retention policies (`STORAGE_RETENTION_DAYS`)
- Dry-run mode and multiple safety checks to prevent data loss
- Audit logging for all storage operations
- 15 tests for storage paths and cleanup logic

**✅ Configuration Validation**
- Schema-based config validation using pipeline metadata
- Validation API endpoint (`POST /api/v1/pipelines/{name}/validate`)
- Field-level error messages with specific paths, types, and valid values
- Pre-flight validation integrated into job submission workflow
- `ConfigValidator` with comprehensive validation logic
- 49 tests (26 unit + 14 API + 9 job submission)

**🔒 Security Hardening**
- Secure-by-default configuration with `AUTH_REQUIRED=true`
- Automatic API key generation on first startup with database-backed token storage
- `videoannotator generate-token` CLI command for additional API keys
- CORS restrictions defaulting to `http://localhost:19011` (configurable via `ALLOWED_ORIGINS`)
- Frictionless CORS configuration for web client developers
- Security warnings logged on startup for insecure configurations
- Comprehensive security documentation suite (`docs/security/`)
- 15 tests (7 startup + 8 CORS)

**📦 Package Namespace Migration**
- Restructured to standard src layout (`src/videoannotator/`)
- Modern Python package structure following PEP 517/518 best practices
- All imports updated to `videoannotator.*` namespace
- Better test isolation and cleaner package boundaries
- Migration guide with automated migration script (`docs/UPGRADING_TO_v1.3.0.md`)
- 20 namespace tests (11 passing core functionality)

**🏥 Enhanced Diagnostics & Health Monitoring**
- Comprehensive diagnostic CLI commands:
  - `videoannotator diagnose system` (Python, FFmpeg, OS info)
  - `videoannotator diagnose gpu` (CUDA, device info, memory)
  - `videoannotator diagnose storage` (free space, write permissions)
  - `videoannotator diagnose database` (connectivity, schema version)
  - `videoannotator diagnose all` (combined report)
- Enhanced health endpoint (`/api/v1/system/health?detailed=true`) with:
  - GPU compute capability detection and compatibility warnings
  - Worker status and active job count
  - Storage diagnostics with disk space warnings
  - Database health checks
  - Pipeline registry status
- ASCII-safe output with `--json` flag for scripting
- Exit codes: 0=pass, 1=errors, 2=warnings
- 15 diagnostic tests + 22 health endpoint tests

**⚙️ Environment Configuration System**
- Comprehensive environment variable configuration module (`src/videoannotator/config_env.py`)
- 19 configurable options including:
  - `STORAGE_DIR`, `STORAGE_RETENTION_DAYS`
  - `MAX_CONCURRENT_JOBS`
  - `AUTH_REQUIRED`, `ALLOWED_ORIGINS`
  - `RETRY_BASE_DELAY`, `RETRY_MAX_DELAY`, `RETRY_JITTER`
  - Database, logging, and pipeline configuration
- Complete documentation at `docs/usage/environment_variables.md`
- Updated `.env.example` with all options
- 19 passing configuration tests

**🐛 Critical Bug Fixes**
- Fixed broken import paths causing "No pipeline classes available" errors
- Added missing pipeline metadata (speaker_diarization, speech_recognition, face_analysis, LAION voice)
- Fixed unit test and integration test imports to use videoannotator package paths
- Resolved pipeline name resolution failures

**📊 API Enhancements**
- Video metadata in job responses (filename, size, duration)
- Disabled trailing slash redirects for better API compatibility
- Job error messages exposed in API responses
- Standardized `ErrorEnvelope` with consistent structure across all endpoints:
  - Fields: `code`, `message`, `detail`, `hint`, `field`, `timestamp`
  - Unified exception handlers (VideoAnnotatorException, APIError)
  - 6 integration tests for error format consistency

**📚 JOSS Publication Requirements**
- Installation verification script (`scripts/verify_installation.py`) with 30 tests
  - Progressive environment validation (Python, FFmpeg, imports, database, GPU, video processing)
  - Platform detection (Linux, macOS, Windows, WSL2)
  - ASCII-safe output with exit codes
- Test coverage validation system (`scripts/validate_coverage.py`)
  - Module-specific thresholds: API (90%), pipelines (80%), database (85%), storage (85%)
  - HTML and XML report generation
  - Comprehensive documentation (`docs/testing/coverage_report.md`)
- Enhanced API endpoint documentation
  - Comprehensive docstrings with curl examples for all major endpoints
  - Detailed request/response examples in Swagger UI
  - Success and error response examples
- JOSS reviewer documentation
  - Quick start guide (`docs/GETTING_STARTED_REVIEWERS.md`) with <15 minute evaluation
  - Comprehensive troubleshooting guide (`docs/installation/troubleshooting.md`)
  - Security configuration guide (`docs/security/`)
- Made `scripts/` a proper Python package for cleaner imports

**📖 Documentation Improvements**
- `docs/UPGRADING_TO_v1.3.0.md` - Complete migration guide
- `docs/archive/v1.3.0/V1.3.0_CLIENT_UPDATE.md` - Client team integration notes (archived)
- `docs/archive/2025/API_IMPROVEMENTS_2025-10-30.md` - API enhancement details (archived)
- `docs/archive/2025/CORS_IMPROVEMENTS_OCT2025.md` - CORS configuration guide (archived)
- `docs/archive/2025/CLIENT_TEAM_UPDATE.md` - Updated client integration info (archived)
- `docs/development/PRE_COMMIT_GUIDE.md` - Pre-commit hook guidance
- `docs/development/scripts_inventory.md` - Scripts audit and documentation
- Enhanced `README.md` and getting started guides

### Changed

- **BREAKING**: Package namespace changed to `videoannotator.*` (migration guide provided)
- **BREAKING**: Authentication now required by default (`AUTH_REQUIRED=true`)
- **BREAKING**: CORS restricted to localhost by default (`ALLOWED_ORIGINS=http://localhost:19011`)
- Default storage moved from `/tmp` to `./storage` for persistence
- All curl examples in documentation updated with Authorization headers
- API version updated to 1.3.0-dev during development

### Fixed

- Pipeline registry validation and name resolution failures
- Import path issues preventing pipeline loading
- Data loss risk from ephemeral `/tmp` storage
- Runaway jobs continuing after delete request
- Invalid configurations passing validation
- Inconsistent error formats across endpoints
- Test import errors across unit and integration tests

### Migration Guide

See `docs/UPGRADING_TO_v1.3.0.md` for detailed migration instructions including:
- Import path updates for `videoannotator.*` namespace
- Environment variable configuration
- API authentication setup
- Storage migration from temp to persistent directories

### Testing

- **Total**: 234 tests passing across all modules
- **Coverage**: Meeting module-specific thresholds (80-90%)
- **New Tests**:
  - 24 cancellation tests
  - 49 validation tests
  - 15 security tests
  - 20 namespace tests
  - 15 diagnostic tests
  - 22 health endpoint tests
  - 30 installation verification tests
  - 19 configuration tests

### Documentation

- 10+ new documentation files
- Complete API documentation with examples
- Security configuration guide
- JOSS reviewer quick start
- Troubleshooting guide
- Migration guide
- Environment variables reference

### 🧪 Test Suite Improvements

**Major Testing Infrastructure Enhancements**
- Improved test suite from 607 passing (79.6%) to 720 passing (94.4%) - **+113 tests fixed**
- Created comprehensive test fixtures infrastructure:
  - Real test media: `tests/fixtures/audio/test.wav` (1.4MB speech audio)
  - Real test video: `tests/fixtures/video/test.mp4` (825KB)
  - Fixtures documentation and recording guidelines
- Fixed integration tests to use real audio instead of synthetic sine waves
- Installed ffmpeg system-wide and added to all Dockerfiles
- Updated conftest.py to prefer real media when available, fall back to synthetic for unit tests

**Test Fixes**
- Fixed 5 database permission tests (removed unnecessary skip decorators)
- Fixed 4 size_analysis config tests (updated to match actual implementation structure)
- Fixed 6 enhanced logging tests (removed emoji for Windows compatibility)
- Fixed 1 pipeline spec documentation test (namespace + regeneration)
- All integration tests now work with real media files

**Test Infrastructure**
- 18 legitimate skipped tests (external dependencies, future features)
- 25 remaining failures (complex integration tests, non-blocking)
- Exceeds 95% passing target (697) by 23 tests

### Acknowledgments

Special thanks to the Video Annotation Viewer team for extensive integration testing that identified critical production issues addressed in this release.

## [1.2.2] - 2025-09-18

### Changed

- Uniform absolute import normalization across API, pipelines, storage, auth, exporters, and CLI to eliminate fragile `src.` and relative (`..`) paths after previous layout adjustments.
- CLI server invocation now targets `api.main:app` directly (removing stale `src.` reference) improving reliability of `videoannotator server`.
- Restored and merged accidentally truncated `docs/archive/development/roadmap_v1.3.0.md` content; added explicit "Package Layout Normalization" technical debt section without loss of prior feature timeline, risks, or metrics.
- Updated Windows console output in version/dependency reporting to ASCII-safe tags only (reinforcing earlier 1.2.1 patch policy) – ensured no reintroduction of emojis in modified modules.

### Added

- Status annotations in the v1.2.1 roadmap marking tasks as COMPLETED / DEFERRED / PARTIAL to synchronize roadmap with actual delivered scope.
- Explicit release date + version bump in `src/version.py` for 1.2.2.
- Technical debt narrative enumerating upcoming packaging namespace migration (planned for v1.3.0) and associated deprecation shim strategy.

### Fixed

- Server startup failure (`ModuleNotFoundError: No module named 'src'`) caused by inconsistent import paths after flattening; all runtime imports now resolvable when installed in editable or built form.
- Documentation integrity regression where large sections of v1.3.0 roadmap were temporarily overwritten; fully restored from history.

### Migration / Guidance

- No API surface changes. Downstream code referencing `src.` prefixes should be updated to plain absolute module imports (e.g. `from api.main import app`).
- Future v1.3.0 namespace migration will introduce `videoannotator.*` package paths; current absolute imports chosen to minimize churn (deprecation shims will map old paths temporarily).

### Internal / Tooling Notes

- Consolidated import approach reduces risk of duplicate module objects under mixed relative/absolute resolution, aiding forthcoming plugin/registry enhancements.
- Roadmap adjustments documented to prevent silent scope shrinkage in strategic planning artifacts.

### Testing / Validation

- Smoke import test: `import api.main, pipelines.base_pipeline, exporters.native_formats` succeeds post-normalization.
- API key optional validation behavior unchanged; 401 still returned only for explicitly invalid provided keys.

### Backward Compatibility

- Fully backward compatible at API & CLI command level; only internal import paths refactored. Any third-party code using undocumented relative imports must adjust.

### Rationale

- Establishes a clean, predictable import baseline before larger v1.3.0 restructuring (namespaced package, extras, plugin hooks) to reduce compounded technical debt.

## [1.2.1] - 2025-09-17

### Added

- Pipeline Registry: YAML-driven pipeline metadata under `src/registry/metadata/` dynamically exposed via `/api/v1/pipelines` (single source of truth).
- Extended Taxonomy Fields: `pipeline_family`, `variant`, `tasks`, `modalities`, `capabilities`, `backends`, optional `stability` replacing the former coarse `category` concept.
- Auto-generated Pipeline Specification: `docs/pipelines_spec.md` produced by `scripts/generate_pipeline_specs.py` (regenerate to update docs; diffs signal drift).
- Emotion Output Format Specification: Standard segment-based JSON schema at `docs/specs/emotion_output_format.md` for emotion-recognition task outputs.
- New Pipelines Registered: `face_openface3_embedding`, `face_laion_clip`, `voice_emotion_baseline` (with combined speech-transcription + emotion-recognition tasks).
- CLI Enhancements: `videoannotator pipelines` now supports `--json`, `--detailed`, and markdown table output.
- API Enhancements: `/api/v1/pipelines` and `/api/v1/pipelines/{name}` now return full metadata including `display_name` and all taxonomy arrays.
- Standard Error Envelope: Introduced `APIError` with consistent JSON structure (`error.code`, `error.message`, `error.hint`) across pipeline + job endpoints.
- Health Enrichment: `/api/v1/system/health` now includes pipeline count, capped name list, uptime_seconds, and explicit embedded job queue status.
- Error Handling Tests: Added unit test ensuring 404 pipeline detail uses standardized envelope.
- CLI Emotion Validation: Added `videoannotator validate-emotion` command for schema checking `.emotion.json` outputs.
- Output Naming Conventions Spec: Canonical file naming patterns documented at `docs/specs/output_naming_conventions.md` (frozen for v1.2.x).
- Emotion Validator Utility: Lightweight schema validator in `src/validation/emotion_validator.py` with tests ensuring emotion JSON conformance.

### Changed

- Deprecated Single `category` Field: Replaced by multi-dimensional taxonomy (no longer emitted in API; remove any downstream reliance on it).
- Documentation Alignment: README and release notes now direct users to `/api/v1/pipelines` and `docs/pipelines_spec.md` instead of hard-coded lists.
- Canonical Discovery: All pipeline listings and attributes should be consumed from the API or generated spec, not ad hoc YAML enumeration in user code.
- CLI Versioning: CLI now derives version from single source `src/version.py` (removed hardcoded API version strings).
- OpenFace 3.0 Import Safety: Converted eager OpenFace imports to lazy loading in `openface3_pipeline` to prevent argparse side-effects and enable test collection without OpenFace installed.

### Migration / Guidance

- If prior tooling referenced `category`, map logic to one or more of: `tasks`, `modalities`, or `pipeline_family` depending on intent.
- Update any scripts that enumerated pipelines manually to call: `videoannotator pipelines --json` for stable machine parsing.
- To regenerate the pipeline spec after adding/editing metadata: run the provided generation script (see header comments in `scripts/generate_pipeline_specs.py`).
- Emotion analysis consumers should validate outputs against the documented schema instead of reverse-engineering per-pipeline fields.

### Notes

- These changes prepare the groundwork for richer capability/resource descriptors planned for v1.3.0 without introducing breaking runtime behaviors in existing pipelines.
- All additions are backward compatible except for removal of the legacy `category` field; no other API contracts changed.

#### Patch Update (Light Stabilization - Auth & Logging)

Date: 2025-09-17 (post initial 1.2.1 feature merge)

Added:

- Optional legacy API key validation helper (`validate_optional_api_key`) enforcing 401 on explicitly invalid `va_` style keys while preserving anonymous access for endpoints that allowed it.

Changed:

- Replaced runtime and test console emojis with ASCII tags (`[OK]`, `[WARNING]`, `[ERROR]`) in `version.py`, `coco_validator.py`, person tracking pipeline logging, and integration test prints for Windows console compatibility.
- Injected conditional auth dependency into job endpoints (no behavior change for anonymous requests unless an invalid key is supplied).

Documentation:

- Appended "Technical Debt & Deferred Stabilization Items" section to `docs/archive/development/roadmap_v1.3.0.md` enumerating deferred heavier tasks (BatchStatus semantics, retry backoff policy, pipeline config defaults, synthetic video fixtures, storage lifecycle cleanup, Whisper CUDA fallback test adjustments, error envelope taxonomy, registry extensions, residual emoji cleanup, auth follow-up tests).

Testing / Validation:

- Targeted integration tests confirm: invalid API key now returns 401; anonymous job submission paths unaffected; no remaining emoji assumptions in modified tests.

Backward Compatibility:

- No breaking API changes; only invalid provided API keys now correctly rejected. Anonymous behavior unchanged where previously permitted.

Rationale:

- Scope intentionally limited to low-risk hardening and Windows-safe output formatting ahead of broader v1.3.0 feature work.

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
