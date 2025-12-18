# VideoAnnotator Documentation

## Current Release: v1.4.1

This documentation is organized into clear sections for different user types and development phases.

## 📖 User Documentation

### Installation

- [Installation Guide](installation/INSTALLATION.md) - **Modern uv-based setup** with CUDA support
- macOS specifics: see the macOS section inside the Installation Guide for libomp, Node, ffmpeg, and PATH fixes.
- [Environment Setup](installation/ENVIRONMENT_SETUP.md) - HuggingFace and configuration
- [Python Development 2025](installation/PythonDev2025.md) - **Modern development practices** with uv, Ruff, and Docker

### Usage

- [Getting Started](usage/GETTING_STARTED.md) - Quick start guide for new users
- [Accessing Results](usage/accessing_results.md) - **New**: Downloading annotations and artifacts
- [Configuration](usage/configuration.md) - Configuration guide
- [Pipeline Specifications](usage/pipeline_specs.md) - Detailed pipeline documentation
- [Scene Detection Guide](usage/scene_detection.md) - Scene detection usage
- [Demo Commands](usage/demo_commands.md) - Example commands and workflows
- [Troubleshooting](usage/troubleshooting.md) - Common issues and solutions

## Command Shortcuts (Containers)

If you are using the provided Docker/devcontainer images, a few convenience commands are available on `PATH`.
These are optional shortcuts; the canonical CLI remains `uv run videoannotator ...`.

| Action | Shortcut | Equivalent |
|---|---|---|
| Initialize the database + create an admin token | `setupdb --admin-email you@example.com --admin-username you` | `uv run videoannotator setup-db --admin-email you@example.com --admin-username you` |
| Run the VideoAnnotator CLI (any subcommand) | `va ...` | `uv run videoannotator ...` |
| Start the API server (recommended defaults) | `va` | `uv run videoannotator` |
| Start the API server (explicit subcommand) | `server ...` | `uv run videoannotator server ...` |
| Generate a new API token | `newtoken ...` | `uv run videoannotator generate-token ...` |
| List available pipelines (detailed) | `va pipelines --detailed` | `uv run videoannotator pipelines --detailed` |
| Validate a config file | `va config --validate configs/default.yaml` | `uv run videoannotator config --validate configs/default.yaml` |
| Submit a processing job | `va job submit video.mp4 --pipelines scene,person` | `uv run videoannotator job submit video.mp4 --pipelines scene,person` |
| Check job status | `va job status <job_id>` | `uv run videoannotator job status <job_id>` |
| List jobs | `va job list` | `uv run videoannotator job list` |
| Run all tests (quick/quiet) | `vatest` | `uv run pytest -q` |
| Run some tests (quick/quiet) | `vatest tests/unit/` | `uv run pytest -q tests/unit/` |

For more copy-pasteable CLI workflows, see `usage/demo_commands.md`.

### Deployment

- [Docker Guide](deployment/Docker.md) - Container deployment (Docker Compose-first)

## 🔧 Development Documentation

### Active Development & Roadmap

- **[Roadmap Overview (Archived)](archive/development/roadmap_overview.md)** - Historical release strategy notes
- [v1.4.0 Roadmap](development/roadmap_v1.4.0.md) - Roadmap for the v1.4.0 cycle
- [v1.5.0 Roadmap](development/roadmap_v1.5.0.md) - Roadmap for the v1.5.0 cycle
- [Examples CLI Update Plan](development/EXAMPLES_CLI_UPDATE_CHECKLIST.md) - CLI modernization checklist

## 🧪 Testing & QA

### Current Testing

- [Testing Overview](testing/testing_overview.md) - Complete testing strategy and results
- [Testing Standards](testing/testing_standards.md) - Quality assurance standards
- [Batch Testing Guide](testing/batch_testing_guide.md) - Batch processing test procedures

## 📁 Archive

Historical and superseded documentation lives under `archive/`:

- [Archive Root](archive/) - Top-level archive index
- [Release Notes](archive/release-notes/) - Historical release notes
- [Development (Archived)](archive/development/) - Completed roadmaps, checklists, and completion summaries
- [Testing (Archived)](archive/testing/) - Historical QA checklists
- [Dated Updates](archive/2025/) - Dated update memos
- **v1.3.0** (Future Release) - Advanced features, security, and scalability enhancements
