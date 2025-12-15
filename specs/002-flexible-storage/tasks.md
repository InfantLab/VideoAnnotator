# Implementation Tasks: Flexible Storage & Artifact Access

**Feature**: Flexible Storage & Artifact Access
**Branch**: `002-flexible-storage`
**Spec**: [spec.md](spec.md)

## Phase 1: Setup

*Goal: Initialize project structure and configuration for the new feature.*

- [x] **T001**: Create storage module structure
  - Create `src/videoannotator/storage/providers/` directory
  - Create `src/videoannotator/storage/providers/__init__.py`
  - Create `src/videoannotator/api/v1/endpoints/artifacts.py` placeholder
  - **File**: `src/videoannotator/storage/providers/__init__.py`

- [x] **T002**: Update configuration schema
  - Add `StorageConfig` model to `src/videoannotator/schemas/config.py` (or equivalent)
  - Update `configs/default.yaml` with `storage` section
  - **File**: `configs/default.yaml`

## Phase 2: Foundational (Blocking)

*Goal: Implement the core storage abstraction and local provider. These are prerequisites for all user stories.*

- [x] **T003**: Implement StorageProvider Interface
  - Define `StorageProvider` ABC with methods: `save`, `load`, `list`, `get_path`, `exists`
  - Define `JobArtifact` data class/model
  - **File**: `src/videoannotator/storage/providers/base.py`

- [x] **T004**: Implement LocalStorageProvider
  - Implement `LocalStorageProvider` inheriting from `StorageProvider`
  - Implement methods using `pathlib` and `shutil`
  - Ensure it respects the configured root path
  - **File**: `src/videoannotator/storage/providers/local.py`

- [x] **T005**: Implement StorageManager Factory
  - Create `get_storage_provider()` factory function
  - Read configuration to instantiate the correct provider (default: Local)
  - **File**: `src/videoannotator/storage/manager.py`

## Phase 3: User Story 1 - Download Job Annotations (P1)

*Goal: Allow users to download all annotations for a job via CLI/API.*

- [x] **T006**: [P] Implement Artifact Listing Service
  - Create service function to list all artifacts for a given `job_id`
  - Filter for annotation files (exclude temp files/videos if needed)
  - **File**: `src/videoannotator/services/artifacts.py`

- [x] **T007**: [P] Implement ZIP Streaming Logic
  - Create utility to stream files from `StorageProvider` into a ZIP archive
  - Use temporary file approach (MVP decision from research)
  - **File**: `src/videoannotator/utils/compression.py`

- [x] **T008**: Implement API Endpoint for Download
  - Add `GET /jobs/{job_id}/artifacts` endpoint
  - Use streaming response with ZIP content type
  - **File**: `src/videoannotator/api/v1/endpoints/artifacts.py`

- [x] **T009**: Register API Router
  - Include `artifacts` router in main API application
  - **File**: `src/videoannotator/api/v1/api.py`

- [x] **T010**: Implement CLI Download Command
  - Add `download-annotations` command to `videoannotator job` group
  - Call API endpoint (or service directly if local) to get the ZIP
  - Save to specified output path
  - **File**: `src/videoannotator/cli/commands/job.py`

## Phase 4: User Story 2 - Configurable Local Storage Root (P2)

*Goal: Allow users to configure the storage root path.*

- [x] **T011**: Update Config Loading Logic
  - Ensure `get_storage_root()` in `src/videoannotator/storage/config.py` reads from the new YAML config structure
  - Deprecate/Update old environment variable logic if necessary to prioritize YAML
  - **File**: `src/videoannotator/storage/config.py`

- [x] **T012**: Validate Storage Path on Startup
  - Add check in `StorageManager` to ensure root path exists or is creatable
  - Log warning if path is not writable
  - **File**: `src/videoannotator/storage/manager.py`

## Phase 5: User Story 3 - Pluggable Storage Architecture (P3)

*Goal: Ensure architecture supports future providers (Verification).*

- [x] **T013**: Refactor Job Creation to use StorageProvider
  - Identify where jobs are currently created (likely `src/videoannotator/api/v1/jobs.py`)
  - Replace direct `os.mkdir`/`pathlib.Path.mkdir` calls with `storage_provider.create_job_dir(job_id)`
  - **File**: `src/videoannotator/api/v1/jobs.py`

- [x] **T014**: Refactor Pipeline Output to use StorageProvider
  - Update pipeline runners to write results via `StorageProvider` (or pass the resolved absolute path from provider)
  - **File**: `src/videoannotator/pipelines/runner.py`

## Phase 6: Polish & Cross-Cutting

- [x] **T015**: Add Logging
  - Add structured logging to all storage operations
  - Ensure no emoji/unicode in logs
  - **File**: `src/videoannotator/storage/providers/local.py`

- [x] **T016**: Update Documentation
  - Update `docs/usage/configuration.md` with new storage options
  - Update `docs/api/openapi.yaml` (if manual update needed)
  - **File**: `docs/usage/configuration.md`

## Dependencies

- **Phase 1 & 2** must be completed before **Phase 3, 4, 5**.
- **T003 (Interface)** -> **T004 (Local Impl)** -> **T005 (Manager)**
- **T005** -> **T006**, **T007**, **T013**, **T014**
- **T008 (API)** depends on **T006** & **T007**
- **T010 (CLI)** depends on **T008**

## Implementation Strategy

1.  **MVP (Stories 1 & 2)**: Focus on getting the `LocalStorageProvider` working and the download endpoint live. This delivers the immediate value of accessing data.
2.  **Refactoring (Story 3)**: Once the new path is established, progressively refactor existing job creation/writing logic to use the provider. This can be done incrementally.
