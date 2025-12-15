# Implementation Plan: Flexible Storage & Artifact Access

**Branch**: `002-flexible-storage` | **Date**: 2025-12-08 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/002-flexible-storage/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

This feature introduces a flexible storage architecture to VideoAnnotator, allowing users to configure the root storage location (e.g., for mounted drives) and download job artifacts via CLI/API. It establishes a `StorageProvider` abstraction to support future cloud backends (S3, OneDrive) while defaulting to the local filesystem for the MVP.

## Technical Context

**Language/Version**: Python 3.12
**Primary Dependencies**: `typer` (CLI), `fastapi` (API), `pydantic` (Models)
**Storage**: Local Filesystem (MVP), Abstraction for future Cloud
**Testing**: `pytest`
**Target Platform**: Linux (Docker container)
**Project Type**: Python API + CLI
**Performance Goals**: Efficient file streaming for downloads.
**Constraints**: No emoji in logs, Windows-safe paths.
**Scale/Scope**: Single-node storage for now.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Low-friction local usability**: PASSED. Default remains local storage.
- **Predictable, non-breaking**: PASSED. New config is optional/backward compatible.
- **Standard formats**: PASSED. No new formats, just file access.
- **Future extensibility**: PASSED. `StorageProvider` enables this.
- **Clear, ASCII-safe console**: PASSED. Will follow logging rules.

## Project Structure

### Documentation (this feature)

```
specs/002-flexible-storage/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # Phase 2 output
```

### Source Code (repository root)

```
src/
├── videoannotator/
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── config.py        # Update to read from main config
│   │   ├── providers/       # NEW: Storage providers
│   │   │   ├── base.py      # NEW: StorageProvider interface
│   │   │   └── local.py     # NEW: LocalStorageProvider
│   │   └── manager.py       # NEW: StorageManager factory
│   ├── api/
│   │   └── v1/
│   │       └── endpoints/
│   │           └── artifacts.py # NEW: Download endpoints
│   └── cli/
│       └── commands/
│           └── job.py       # Update: Add download-annotations
```

**Structure Decision**: We will expand the existing `videoannotator.storage` module to include a `providers` sub-package for the abstraction.

## Complexity Tracking

N/A - No violations.
