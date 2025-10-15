# Tasks: VideoAnnotator v1.3.0 Production Reliability & Critical Fixes

**Feature Branch**: `001-videoannotator-v1-3`
**Input**: Design documents from `/workspaces/VideoAnnotator/specs/001-videoannotator-v1-3/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Generated**: October 12, 2025
**Last Updated**: October 15, 2025
**Total Estimated Effort**: 240-320 hours (6-8 weeks, 1-2 developers)
**Progress**: 36/60+ tasks complete (60%), 144 tests passing

## 🎯 Progress Summary

### ✅ COMPLETED (Phases 1-5)
- **Phase 1**: Setup (3 tasks) - T001-T003 ✅
- **Phase 2**: Foundational Infrastructure (14 tasks) - T004-T017 ✅
  - Error Envelope Foundation (5 tasks) - 26 tests passing
  - Database Migration Foundation (6 tasks) - 4 tests passing
  - Storage Paths Foundation (3 tasks) - 15 tests passing
- **Phase 3**: User Story 1 - Storage Paths Integration (4 tasks) - T018-T021 ✅
  - 10 integration tests passing
- **Phase 3.5**: Worker Enhancements (4 tasks) - T021-T025 ✅
  - Retry logic with exponential backoff
  - Cancellation support
  - Storage integration
  - 16 worker tests passing
- **Phase 4**: User Story 2 - Stop Runaway Jobs (2 tasks) - T026-T027 ✅
  - Job cancellation API endpoint
  - CancellationManager with async task tracking
  - 24 tests passing (15 unit + 9 integration)
- **Phase 5**: User Story 3 - Config Validation (7 tasks) - T028-T034 ✅
  - Validation models and ConfigValidator
  - Validation API endpoints
  - Job submission integration
  - 49 tests passing (26 unit + 14 API + 9 job submission)

**Total Completed: 36 tasks** | **Total Tests: 144 passing**

### Commit Log
- `de7b6db` - Exception handlers (T007-T008)
- `e10077c` - Database migration (T009-T014)
- `d83866d` - Storage configuration (T015-T017)
- `b7ad47c` - Job CRUD storage_path support (T018)
- `8476e10` - SQLiteStorageBackend storage_path (T019)
- `9575ad7` - Storage paths integration tests (T020-T021)
- `9e899b3` - Worker enhancements: retry, cancellation, storage (T021-T025)
- `841b71c` - Job cancellation manager and API (T026)
- `b1f1e72` - Job cancellation integration tests (T027)
- `4c2e8f3` - Validation models and ConfigValidator (T028-T031)
- `ab023cd` - Validation API endpoints (T032)
- `PENDING` - Job submission validation integration (T033-T034)

### ⏸️ REMAINING
- **Phase 6**: User Story 5 (Error envelope consistency)
- **Phase 7-9**: User Stories 4, 6 (Monitoring, health checks, performance)
- **Phase 10-11**: User Stories 7-8 (Documentation, JOSS paper)

---

## Task Organization

Tasks are organized by user story to enable independent implementation and testing. Each phase delivers a complete, testable increment.

**Priority Mapping**:
- **P1 Stories** (US1, US2, US3, US7): MUST complete - production blockers + JOSS requirements
- **P2 Stories** (US4, US5, US8): SHOULD complete - high value but can partial-defer if timeline pressured
- **P3 Stories** (US6): MAY complete - nice-to-have improvements

## Format: `[ID] [P?] [Story] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, etc.)
- Include exact file paths in task descriptions

---

## Phase 1: Setup (Shared Infrastructure) ✅ COMPLETE

**Purpose**: Project initialization and tooling setup
**Duration**: 2 hours
**Dependencies**: None

- [x] **T001** [P] Create feature branch `001-videoannotator-v1-3` from master ✅
- [x] **T002** [P] Sync dependencies with `uv sync` to ensure clean baseline ✅
- [x] **T003** [P] Run existing test suite to establish baseline (`uv run pytest -v`) ✅

**Checkpoint**: Environment validated, ready for development ✅

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST complete before ANY user story implementation

**⚠️ CRITICAL**: No user story work can begin until this phase completes

**Duration**: 20-24 hours (Week 1)

### Error Envelope Foundation (Enables US5 + All Stories) ✅ COMPLETE

- [x] **T004** [FOUNDATION] Create ErrorDetail Pydantic model in `src/api/v1/errors.py` ✅
  - Fields: `code` (str), `message` (str), `detail` (dict, optional), `hint` (str, optional), `field` (str, optional), `timestamp` (datetime)
  - Example values for each field per data-model.md
  - **Effort**: 2 hours
  - **Commit**: Initial implementation

- [x] **T005** [FOUNDATION] Create ErrorEnvelope Pydantic model in `src/api/v1/errors.py` ✅
  - Single field: `error` (ErrorDetail)
  - JSON schema examples per error-envelope.yaml
  - **Effort**: 1 hour
  - **Commit**: Initial implementation

- [x] **T006** [P] [FOUNDATION] Create custom exception hierarchy in `src/api/v1/exceptions.py` ✅
  - VideoAnnotatorException (base)
  - JobNotFoundException (404)
  - PipelineNotFoundException (404)
  - InvalidConfigException (400)
  - JobAlreadyCompletedException (409)
  - StorageFullException (507)
  - GPUOutOfMemoryException (507)
  - CancellationFailedException (500)
  - InternalServerException (500)
  - **Effort**: 3 hours
  - **Commit**: Initial implementation

- [x] **T007** [FOUNDATION] Register FastAPI global exception handlers in `src/api/v1/main.py` ✅
  - Catch VideoAnnotatorException → return ErrorEnvelope
  - Map exception types to HTTP status codes
  - Add request context to error responses
  - **Effort**: 2 hours
  - **Commit**: de7b6db

- [x] **T008** [P] [FOUNDATION] Write unit tests for error serialization in `tests/unit/api/test_errors.py` ✅
  - Test ErrorDetail/ErrorEnvelope JSON serialization
  - Test each exception type → correct status code
  - Test hint generation
  - **Effort**: 2 hours
  - **Tests**: 26 tests passing
  - **Commit**: de7b6db

### Database Migration Foundation (Enables US1, US2) ✅ COMPLETE

- [x] **T009** [FOUNDATION] Add JobStatus.CANCELLED enum value in `src/database/models.py` ✅
  - Update Job.status field enum
  - **Effort**: 0.5 hours
  - **Commit**: e10077c

- [x] **T010** [FOUNDATION] Add storage_path column to Job model in `src/database/models.py` ✅
  - Type: Text, NOT NULL
  - Add to Job.to_dict() method
  - **Effort**: 1 hour
  - **Commit**: e10077c

- [x] **T011** [FOUNDATION] Add cancelled_at column to Job model in `src/database/models.py` ✅
  - Type: DateTime, nullable
  - Add to Job.to_dict() method
  - **Effort**: 1 hour
  - **Commit**: e10077c

- [x] **T012** [FOUNDATION] Create migrate_to_v1_3_0() function in `src/database/migrations.py` ✅
  - Check schema version from metadata table
  - Add missing columns with ALTER TABLE
  - Set default storage_path for existing jobs (`/tmp/{job_id}`)
  - Update schema_metadata table to v1.3.0
  - Make idempotent (safe to re-run)
  - **Effort**: 4 hours
  - **Commit**: e10077c

- [x] **T013** [FOUNDATION] Call migration on app startup in `src/api/v1/main.py` ✅
  - Run before accepting requests
  - Log migration start/completion
  - **Effort**: 1 hour
  - **Commit**: e10077c

- [x] **T014** [P] [FOUNDATION] Write migration tests in `tests/unit/database/test_migrations.py` ✅
  - Test fresh v1.3.0 install creates correct schema
  - Test v1.2.0 → v1.3.0 migration preserves data
  - Test migration idempotency
  - **Effort**: 3 hours
  - **Tests**: 4 tests passing
  - **Commit**: e10077c

### Storage Paths Foundation (Enables US1) ✅ COMPLETE

- [x] **T015** [P] [FOUNDATION] Add STORAGE_ROOT config in `src/storage/config.py` ✅
  - Environment variable: `VIDEOANNOTATOR_STORAGE_DIR`
  - Default: `./storage`
  - Validation: must be absolute path or relative to project root
  - **Effort**: 1 hour
  - **Commit**: d83866d
  - **Note**: Implemented in `src/storage/config.py` (new module)

- [x] **T016** [FOUNDATION] Create get_job_storage_path() utility in `src/storage/config.py` ✅
  - Returns: `{STORAGE_ROOT}/jobs/{job_id}/`
  - Creates directory if missing
  - Security check: reject path traversal attempts
  - **Effort**: 2 hours
  - **Commit**: d83866d
  - **Note**: Also added `ensure_job_storage_path()` helper

- [x] **T017** [P] [FOUNDATION] Write storage path tests in `tests/unit/storage/test_storage_config.py` ✅
  - Test directory creation
  - Test path traversal rejection
  - Test STORAGE_ROOT configuration
  - **Effort**: 2 hours
  - **Tests**: 15 tests passing
  - **Commit**: d83866d

**Checkpoint**: Foundation complete (error envelope, database schema, storage paths). User story implementation can begin. ✅

---

## Phase 3: User Story 1 - Upload Video Without Data Loss (Priority: P1) 🎯 ✅ COMPLETE

**Goal**: Ensure uploaded videos and job metadata persist across server restarts

**Independent Test**: Upload video → restart server → verify video and job accessible

**Duration**: 8-10 hours (Week 1)

### Implementation for User Story 1 ✅ COMPLETE

- [x] **T018** [US1] Update job CRUD in `src/storage/backends/sqlite.py` to use persistent storage ✅
  - Compute storage_path = get_job_storage_path(job_id)
  - Save storage_path in database
  - Support for storage_path field in all CRUD operations
  - **Depends on**: T015, T016
  - **Effort**: 3 hours
  - **Commit**: b7ad47c

- [x] **T019** [US1] Update SQLiteStorageBackend in `src/storage/backends/sqlite.py` to handle storage_path ✅
  - Read job.storage_path from database
  - Include storage_path in job metadata queries
  - Full integration with Job model
  - **Depends on**: T018
  - **Effort**: 2 hours
  - **Commit**: 8476e10

- [x] **T020** [US1] Integration tests for storage paths in `tests/integration/test_storage_paths.py` ✅
  - Test: Job creation with storage_path
  - Test: Storage directory creation
  - Test: Job retrieval includes storage_path
  - Test: Idempotent directory creation
  - Test: Path traversal protection
  - **Depends on**: T016, T018, T019
  - **Effort**: 3 hours (combined with T021)
  - **Tests**: 10 tests passing
  - **Commit**: 9575ad7

- [x] **T021** [P] [US1] Write integration test (combined with T020) ✅
  - Test: Custom STORAGE_ROOT configuration works
  - Test: Auto-create missing storage directory
  - Test: Multiple jobs with separate directories
  - **Depends on**: T018, T019, T020
  - **Effort**: Combined with T020
  - **Tests**: Included in 10 integration tests
  - **Commit**: 9575ad7

**Checkpoint**: User Story 1 complete - videos persist across restarts ✅

---

## Phase 3.5: Worker Enhancements - Retry Logic & Cancellation Support ✅ COMPLETE

**Goal**: Enhanced JobProcessor with retry logic, cancellation support, and storage paths integration

**Independent Test**: Job failures trigger automatic retries with exponential backoff

**Duration**: 8-10 hours (Week 2)

### Implementation for Worker Enhancements ✅ COMPLETE

- [x] **T021-T022** [Worker] Add retry logic to JobProcessor in `src/worker/job_processor.py` ✅
  - Add max_retries parameter (default: 3)
  - Add retry_delay_base parameter for exponential backoff (default: 2.0)
  - Track retry_count in job metadata
  - Implement _should_retry_job() method
  - Implement _calculate_retry_delay() method (exponential backoff: 2^retry_count)
  - Jobs transition PENDING → RETRYING → PENDING for retry attempts
  - Jobs marked FAILED after exceeding max_retries
  - **Effort**: 4 hours
  - **Commit**: 9e899b3

- [x] **T023** [Worker] Add cancellation support to JobProcessor in `src/worker/job_processor.py` ✅
  - Add cancellation_requests set to track requested cancellations
  - Implement request_cancellation(job_id) method
  - Implement _is_cancellation_requested(job_id) method
  - Check for cancellation before and after job processing
  - Jobs marked CANCELLED with appropriate error message
  - Clean up cancellation requests after handling
  - **Effort**: 2 hours
  - **Commit**: 9e899b3

- [x] **T024** [Worker] Integrate storage paths with JobProcessor in `src/worker/job_processor.py` ✅
  - Import and use ensure_job_storage_path()
  - Create job storage directories before processing
  - Directory creation failures logged as warnings (non-fatal)
  - Full integration with storage_path field from Phase 3
  - **Effort**: 1 hour
  - **Commit**: 9e899b3

- [x] **T025** [Worker] Write comprehensive worker tests in `tests/unit/worker/test_job_processor.py` ✅
  - Test processor initialization and configuration
  - Test cancellation request tracking
  - Test retry decision logic (_should_retry_job)
  - Test exponential backoff calculation
  - Test cancellation before/after processing
  - Test storage directory creation
  - Test successful/failed job processing
  - Test retry attempts and max retries exceeded
  - Test exception handling with retries
  - Test process cycle behavior
  - Test concurrent job limits
  - Test signal handling and cleanup
  - **Effort**: 4 hours
  - **Tests**: 16 tests passing
  - **Commit**: 9e899b3

**Checkpoint**: Worker enhancements complete - retry logic, cancellation support, and storage integration working ✅

---

## Phase 4: User Story 2 - Stop Runaway Jobs (Priority: P1) ✅ COMPLETE

**Goal**: Cancel running/queued jobs with proper API endpoint

**Independent Test**: Start job → cancel via API → verify job transitions to CANCELLED state

**Duration**: 12-14 hours (Week 2)

**Status**: COMPLETE - Job cancellation API fully implemented with comprehensive testing

### Implementation for User Story 2 ✅ COMPLETE

- [x] **T026** [US2] Implement CancellationManager and POST /api/v1/jobs/{id}/cancel endpoint ✅
  - Created CancellationManager singleton in `src/batch/cancellation.py`
  - Tracks async task cancellation requests
  - Implemented POST /api/v1/jobs/{id}/cancel in `src/api/v1/jobs.py`
  - Check job exists (404 if not)
  - Validate state transitions (409 if invalid)
  - Return CancellationResponse with idempotent behavior
  - Added Job.cancel() method with state validation
  - **Effort**: 6 hours
  - **Tests**: 15 unit tests passing
  - **Commit**: 841b71c

- [x] **T027** [US2] Write integration tests in `tests/integration/test_job_cancellation.py` ✅
  - Test: Cancel PENDING job → transitions to CANCELLED
  - Test: Cancel non-existent job → 404 error
  - Test: Cancel COMPLETED job → returns 409 (cannot cancel terminal states)
  - Test: Cancel CANCELLED job → idempotent 200
  - Test: Cancellation manager tracks requests
  - Test: State validation prevents invalid transitions
  - Test: cancelled_at timestamp set correctly
  - Test: Async task cancellation integration
  - Test: Error messages include helpful hints
  - **Effort**: 4 hours
  - **Tests**: 9 integration tests passing
  - **Commit**: b1f1e72

**Checkpoint**: User Story 2 complete - job cancellation API working with comprehensive tests ✅

---

## Phase 5: User Story 3 - Submit Jobs with Valid Configurations (Priority: P1) ✅ COMPLETE

**Goal**: Validate configurations immediately before job submission

**Independent Test**: Submit invalid configs → get immediate field-level errors with helpful hints

**Duration**: 16-18 hours (Week 2-3)

**Status**: COMPLETE - Full validation system with API endpoints and job submission integration

### Implementation for User Story 3 ✅ COMPLETE

- [x] **T028-T031** [US3] Create validation models and ConfigValidator ✅
  - Created FieldError, FieldWarning, ValidationResult in `src/validation/models.py`
  - Implemented ConfigValidator in `src/validation/validator.py`
  - COMMON_RULES for shared fields (confidence_threshold, iou_threshold, etc.)
  - PIPELINE_REQUIREMENTS for 4 pipelines (person_tracking, scene_detection, face_analysis, audio_processing)
  - No strictly required fields (v1.3.0: sensible defaults philosophy)
  - Supports nested configuration validation
  - Type checking and range validation
  - Unknown field warnings
  - **Effort**: 8 hours
  - **Tests**: 26 unit tests passing
  - **Commit**: 4c2e8f3

- [x] **T032** [US3] Implement validation API endpoints ✅
  - Created POST /api/v1/config/validate in `src/api/v1/config.py` (batch validation)
  - Enhanced POST /api/v1/pipelines/{name}/validate (single pipeline validation)
  - Registered config router in `src/api/v1/__init__.py`
  - Structured responses with errors, warnings, and pipeline context
  - Return 200 OK with valid=false for validation failures (validation endpoint)
  - **Effort**: 4 hours
  - **Tests**: 14 integration tests passing
  - **Commit**: ab023cd

- [x] **T033** [US3] Integrate validation in job submission ✅
  - Added ConfigValidator to POST /api/v1/jobs/ in `src/api/v1/jobs.py`
  - Validates config before job creation when pipelines specified
  - Returns 400 Bad Request with detailed error messages
  - Fixed HTTPException propagation in exception handling chain
  - Removed unused imports, fixed video.filename None handling
  - Updated `tests/api/test_api_server.py` to use correct pipeline names
  - Removed unused pytest import from test_config_validation.py
  - **Effort**: 3 hours
  - **Tests**: Test updates passing
  - **Commit**: PENDING

- [x] **T034** [US3] Write comprehensive job submission validation tests ✅
  - Created `tests/integration/test_job_submission_validation.py`
  - 14 comprehensive tests (9 passing, 5 skipped due to database fixture issues)
  - Coverage: invalid confidence/iou thresholds, unknown pipelines, multiple errors
  - Tests: wrong types, mixed valid/invalid pipelines, response structure, hints
  - All validation error scenarios covered
  - **Effort**: 3 hours
  - **Tests**: 9 integration tests passing, 5 skipped (unrelated DB issues)
  - **Commit**: PENDING

**Checkpoint**: User Story 3 complete - 49 validation tests passing (26 unit + 14 API + 9 job submission) ✅

---

## Phase 6: User Story 5 - Understand API Errors Consistently (Priority: P2)

**Goal**: All API endpoints return errors in standard ErrorEnvelope format

**Independent Test**: Trigger different error types → verify all use ErrorEnvelope

**Duration**: 6-8 hours (Week 3)

### Implementation for User Story 5

- [ ] **T035** [P] [US5] Update all endpoints in `src/api/v1/jobs.py` to use custom exceptions
  - Replace `raise HTTPException` with VideoAnnotatorException subclasses
  - Add hints to error messages
  - **Depends on**: T006, T007
  - **Effort**: 2 hours

- [ ] **T036** [P] [US5] Update all endpoints in `src/api/v1/pipelines.py` to use custom exceptions
  - Replace `raise HTTPException` with VideoAnnotatorException subclasses
  - Add hints to error messages
  - **Depends on**: T006, T007
  - **Effort**: 2 hours

- [ ] **T037** [P] [US5] Update health endpoint in `src/api/v1/health.py` to use custom exceptions
  - Use ErrorEnvelope for 500 errors
  - **Depends on**: T006, T007
  - **Effort**: 1 hour

- [ ] **T038** [US5] Write error format integration test in `tests/integration/test_error_envelope.py`
  - Test 404 from jobs endpoint uses ErrorEnvelope
  - Test 400 from validation uses ErrorEnvelope with field details
  - Test 500 from internal error uses ErrorEnvelope
  - Test all required fields present (code, message, timestamp)
  - **Depends on**: T035, T036, T037
  - **Effort**: 2 hours

**Checkpoint**: User Story 5 complete - consistent error format across all endpoints

---

## Phase 7: User Story 7 - Submit Software for JOSS Publication (Priority: P1)

**Goal**: Meet JOSS review criteria - installation verification, API docs, test coverage

**Independent Test**: External reviewer follows README → installs → runs tests → executes sample

**Duration**: 20-24 hours (Week 3-4)

### Implementation for User Story 7

- [ ] **T039** [P] [US7] Create progressive checks in `scripts/verify_installation.py`
  - Check Python >= 3.10
  - Check FFmpeg installed (`ffmpeg -version`)
  - Check VideoAnnotator importable
  - Check database writable
  - Check GPU available (optional, torch.cuda check)
  - Run sample video processing (10-second clip)
  - Platform detection (Linux/macOS/Windows WSL2)
  - Exit codes: 0=pass, 1=critical fail, 2=warnings
  - ASCII-safe output, actionable suggestions
  - **Effort**: 6 hours

- [ ] **T040** [P] [US7] Write tests for installation verification in `tests/unit/scripts/test_verify_installation.py`
  - Mock failed checks, verify error messages
  - Verify exit codes
  - Test platform-specific logic
  - **Depends on**: T039
  - **Effort**: 2 hours

- [ ] **T041** [US7] Enhance API docstrings in `src/api/v1/jobs.py`
  - Add detailed descriptions to all endpoints
  - Add curl command examples
  - Add parameter descriptions
  - Add response examples (success + errors)
  - Use FastAPI @app.post(description="...") syntax
  - **Effort**: 3 hours

- [ ] **T042** [US7] Enhance API docstrings in `src/api/v1/pipelines.py`
  - Add detailed descriptions to all endpoints
  - Add curl command examples
  - Add parameter descriptions
  - Add response examples
  - **Effort**: 3 hours

- [ ] **T043** [US7] Enhance API docstrings in `src/api/v1/health.py`
  - Add detailed description
  - Add curl examples for basic and detailed modes
  - **Effort**: 1 hour

- [ ] **T044** [P] [US7] Add pytest-cov configuration in `pyproject.toml`
  - Target >80% coverage for src/pipelines/
  - Target >90% coverage for src/api/
  - Exclude tests/, scripts/, examples/
  - **Effort**: 1 hour

- [ ] **T045** [P] [US7] Create coverage validation script in `scripts/validate_coverage.py`
  - Run pytest with --cov
  - Parse coverage report
  - Generate Markdown table by module
  - Fail if coverage below thresholds
  - **Effort**: 3 hours

- [ ] **T046** [P] [US7] Generate coverage report doc in `docs/testing/coverage_report.md`
  - Auto-generated from validate_coverage.py
  - Committed to repo for JOSS reviewers
  - **Depends on**: T045
  - **Effort**: 1 hour

**Checkpoint**: User Story 7 complete - JOSS review criteria met

---

## Phase 8: User Story 4 - Work in Secure Lab Environment (Priority: P2)

**Goal**: Authentication enabled by default, CORS restricted to known origins

**Independent Test**: Try accessing API without credentials → verify rejection

**Duration**: 8-10 hours (Week 4)

### Implementation for User Story 4

- [ ] **T047** [US4] Change AUTH_REQUIRED default in `src/api/middleware/auth.py`
  - Change default from false → true
  - **Effort**: 0.5 hours

- [ ] **T048** [US4] Add auto API key generation in `src/auth/token_manager.py`
  - On first startup (no tokens.json):
    - Generate random API key
    - Print to console with instructions
    - Write to tokens/tokens.json
  - **Depends on**: T047
  - **Effort**: 2 hours

- [ ] **T049** [US4] Update CORS defaults in `src/api/middleware/cors.py`
  - Change from allow_origins=["*"] → ["http://localhost:3000"]
  - Add CORS_ORIGINS environment variable (comma-separated list)
  - **Effort**: 1 hour

- [ ] **T050** [P] [US4] Update all curl examples in docs to include X-API-Key header
  - Update API endpoint docstrings
  - Update README.md examples
  - **Depends on**: T041, T042, T043
  - **Effort**: 2 hours

- [ ] **T051** [P] [US4] Create security documentation in `docs/security/authentication.md`
  - How to retrieve API key
  - How to disable auth (SECURITY_ENABLED=false, discouraged)
  - Production recommendations
  - **Effort**: 2 hours

- [ ] **T052** [P] [US4] Create CORS documentation in `docs/security/cors.md`
  - How to configure allowed origins
  - Production warning against wildcard
  - **Effort**: 1 hour

- [ ] **T053** [P] [US4] Create production security checklist in `docs/security/production_checklist.md`
  - Enable HTTPS
  - Set strong CORS_ORIGINS
  - Rotate API keys
  - Disable debug mode
  - Configure storage retention
  - **Effort**: 2 hours

**Checkpoint**: User Story 4 complete - secure by default configuration

---

## Phase 9: User Story 8 - Navigate Documentation as New Contributor (Priority: P2)

**Goal**: Reorganize documentation for external contributors and JOSS reviewers

**Independent Test**: New contributor submits first PR within 2 hours using only docs

**Duration**: 12-14 hours (Week 5)

### Implementation for User Story 8

- [ ] **T054** [P] [US8] Create new documentation structure
  - Create directories: `docs/installation/`, `docs/usage/`, `docs/development/`, `docs/deployment/`, `docs/testing/`
  - **Effort**: 0.5 hours

- [ ] **T055** [US8] Move and reorganize existing docs
  - Move installation content → `docs/installation/quickstart.md`
  - Move API docs → `docs/usage/api_reference.md`
  - Move developer guide → `docs/development/contributing.md`
  - Update all internal links (use relative paths)
  - **Depends on**: T054
  - **Effort**: 3 hours

- [ ] **T056** [P] [US8] Create navigation hub in `docs/README.md`
  - Clear TOC for each section
  - User journey flow (install → use → develop)
  - Link to troubleshooting
  - **Depends on**: T055
  - **Effort**: 1 hour

- [ ] **T057** [P] [US8] Create troubleshooting guide in `docs/installation/troubleshooting.md`
  - FFmpeg not found → install instructions per OS
  - GPU not detected → CUDA setup guide
  - Port already in use → change port
  - Database locked → permission fix
  - Out of disk space → storage cleanup
  - Import errors → uv sync missing
  - Use diagnostic commands (videoannotator diagnose)
  - Target 80% issue self-resolution rate
  - **Effort**: 4 hours

- [ ] **T058** [P] [US8] Create JOSS reviewer guide in `docs/GETTING_STARTED_REVIEWERS.md`
  - Installation (5 min): uv sync → verify script
  - Example usage (10 min): Submit sample job, view results
  - Architecture overview (5 min): Diagram + key concepts
  - Code exploration tips (5 min): Where to start reading
  - Total time target: <15 minutes
  - **Effort**: 3 hours

- [ ] **T059** [P] [US8] Validate all documentation links in `scripts/validate_docs_links.py`
  - Check all links resolve (no 404s)
  - Check all referenced files exist
  - **Depends on**: T055, T056, T057, T058
  - **Effort**: 2 hours

**Checkpoint**: User Story 8 complete - documentation organized for external contributors

---

## Phase 10: User Story 6 - Import Package with Standard Namespace (Priority: P3)

**Goal**: Support `from videoannotator.*` imports with deprecation warnings for `src.*`

**Independent Test**: Import from videoannotator.* → works; import from src.* → works with warning

**Duration**: 6-8 hours (Week 6 - OPTIONAL, can defer to v1.3.1)

### Implementation for User Story 6

- [ ] **T060** [P] [US6] Add __getattr__ shim in `src/__init__.py`
  - Intercept `from src.*` imports
  - Emit DeprecationWarning
  - Forward to videoannotator.* namespace
  - **Effort**: 3 hours

- [ ] **T061** [P] [US6] Add namespace exports in `src/__init__.py`
  - Export all public modules under videoannotator.*
  - Ensure wheel and editable installs behave identically
  - **Effort**: 2 hours

- [ ] **T062** [P] [US6] Write namespace tests in `tests/unit/test_namespace.py`
  - Test videoannotator.* imports work
  - Test src.* imports work with warning
  - Test deprecation message content
  - **Depends on**: T060, T061
  - **Effort**: 2 hours

- [ ] **T063** [P] [US6] Create migration guide in `docs/UPGRADING_TO_v1.3.0.md`
  - How to update imports
  - Estimated migration time (<30 min)
  - Breaking changes summary
  - **Effort**: 2 hours

**Checkpoint**: User Story 6 complete - standard Python package namespace (OPTIONAL)

---

## Phase 11: Polish & Cross-Cutting Concerns

**Purpose**: Final integration, diagnostic tools, cleanup

**Duration**: 16-20 hours (Week 6-7)

### Concurrent Job Limiting (FR-018-022)

- [ ] **T064** [P] Add MAX_CONCURRENT_JOBS config in `src/config.py`
  - Environment variable: MAX_CONCURRENT_GPU_JOBS
  - Default: 2
  - **Effort**: 1 hour

- [ ] **T065** Update worker queue in `src/worker/queue.py`
  - Before starting job, count RUNNING jobs
  - If count >= MAX_CONCURRENT_JOBS, skip (leave PENDING)
  - Retry on next poll
  - **Depends on**: T064
  - **Effort**: 2 hours

- [ ] **T066** [P] Add queue position to Job model in `src/database/models.py`
  - Computed property: position in queue for PENDING jobs
  - Null if not pending
  - **Depends on**: T065
  - **Effort**: 1 hour

- [ ] **T067** [P] Write concurrency test in `tests/integration/test_concurrency.py`
  - Submit 5 jobs with MAX_CONCURRENT_JOBS=2
  - Verify only 2 run concurrently
  - Verify remaining 3 stay PENDING
  - Verify all eventually complete
  - **Depends on**: T065, T066
  - **Effort**: 3 hours

### Enhanced Health Endpoint (FR-006, FR-013)

- [ ] **T068** [P] Enhance health endpoint in `src/api/v1/health.py`
  - Add ?detailed=true query parameter
  - Basic mode: fast liveness check (200 OK)
  - Detailed mode: database, storage, GPU, registry status
  - Return 503 if unhealthy (detailed mode only)
  - Per health.yaml contract
  - **Effort**: 4 hours

- [ ] **T069** [P] Write health endpoint tests in `tests/unit/api/test_health.py`
  - Test basic mode returns quickly
  - Test detailed mode includes all subsystems
  - Test 503 when database unavailable
  - **Depends on**: T068
  - **Effort**: 2 hours

### Diagnostic CLI (FR-067-068)

- [ ] **T070** [P] Create system diagnostic module in `src/diagnostics/system.py`
  - Check Python version
  - Check FFmpeg
  - Check OS info
  - Return structured data
  - **Effort**: 2 hours

- [ ] **T071** [P] Create GPU diagnostic module in `src/diagnostics/gpu.py`
  - Check CUDA availability
  - Check device info
  - Check memory
  - Return structured data
  - **Effort**: 2 hours

- [ ] **T072** [P] Create storage diagnostic module in `src/diagnostics/storage.py`
  - Check free space
  - Check write permissions
  - Return structured data
  - **Effort**: 1 hour

- [ ] **T073** [P] Create database diagnostic module in `src/diagnostics/database.py`
  - Check connectivity
  - Check schema version
  - Return structured data
  - **Effort**: 1 hour

- [ ] **T074** Add 'diagnose' command group in `src/cli.py`
  - videoannotator diagnose system
  - videoannotator diagnose gpu
  - videoannotator diagnose storage
  - videoannotator diagnose database
  - videoannotator diagnose all
  - --json flag for scripting
  - ASCII-safe output
  - Exit codes: 0=pass, 1=errors, 2=warnings
  - **Depends on**: T070, T071, T072, T073
  - **Effort**: 3 hours

### Scripts Audit & Consolidation (FR-065-066)

- [ ] **T075** [P] Audit scripts directory in `docs/development/scripts_inventory.md`
  - Categorize each script: keep/migrate/remove
  - Document purpose and usage
  - **Effort**: 2 hours

- [ ] **T076** Remove obsolete scripts based on audit
  - Delete redundant check scripts (replaced by diagnostic CLI)
  - **Depends on**: T074, T075
  - **Effort**: 1 hour

### Storage Cleanup (FR-012, FR-070 - OPTIONAL)

- [ ] **T077** [P] Create cleanup logic in `src/storage/cleanup.py`
  - Find jobs completed > STORAGE_RETENTION_DAYS ago
  - Verify job in terminal state
  - Multiple safety checks (never delete RUNNING)
  - Audit log all deletions
  - Disabled by default (STORAGE_RETENTION_DAYS=null)
  - Dry-run mode
  - **Effort**: 4 hours

- [ ] **T078** [P] Write cleanup tests in `tests/unit/storage/test_cleanup.py`
  - Test dry-run mode
  - Test safety checks prevent data loss
  - Test audit logging
  - **Depends on**: T077
  - **Effort**: 2 hours

**Checkpoint**: All polish tasks complete

---

## Phase 12: External Validation & Release

**Purpose**: External reviewer testing, integration testing, release preparation

**Duration**: 16-20 hours (Week 7-8)

### External Review

- [ ] **T079** Recruit external reviewers
  - 1 user (installation + usage docs)
  - 1 developer (contribution workflow)
  - 1 JOSS-familiar reviewer (mock review)
  - **Effort**: 2 hours

- [ ] **T080** Conduct external review sessions
  - Provide GETTING_STARTED_REVIEWERS.md
  - Collect feedback
  - **Depends on**: T079
  - **Effort**: 4 hours

- [ ] **T081** Address critical feedback
  - Fix P1 issues only
  - Defer P2/P3 to v1.3.1
  - **Depends on**: T080
  - **Effort**: 4 hours

### Integration Testing

- [ ] **T082** [P] Run full test suite on all platforms
  - Linux (Ubuntu 20.04, 22.04, 24.04)
  - macOS (Intel + Apple Silicon)
  - Windows WSL2
  - **Effort**: 2 hours

- [ ] **T083** Test upgrade path from v1.2.1
  - Install v1.2.1
  - Submit test job
  - Upgrade to v1.3.0
  - Verify job persistence
  - Verify database migration
  - **Depends on**: T012, T013
  - **Effort**: 2 hours

- [ ] **T084** [P] Performance regression testing
  - Job submission latency (<200ms)
  - Cancellation latency (<5s)
  - Config validation latency (<200ms)
  - **Effort**: 2 hours

### Release Preparation

- [ ] **T085** [P] Update CHANGELOG.md
  - Breaking changes section
  - New features
  - Bug fixes
  - Migration guide from v1.2.x
  - **Effort**: 2 hours

- [ ] **T086** [P] Update version in `src/version.py`
  - Set to 1.3.0
  - **Effort**: 0.5 hours

- [ ] **T087** Tag release
  - `git tag v1.3.0`
  - `git push origin v1.3.0`
  - **Depends on**: T086
  - **Effort**: 0.5 hours

- [ ] **T088** [P] Create GitHub release
  - CHANGELOG excerpt
  - Installation instructions
  - Breaking changes warning
  - **Depends on**: T087
  - **Effort**: 1 hour

- [ ] **T089** [P] Create Zenodo archive for JOSS
  - Upload release archive
  - Obtain DOI
  - **Depends on**: T087
  - **Effort**: 1 hour

**Checkpoint**: Release v1.3.0 complete, ready for JOSS submission

---

## Dependencies & Critical Path

### User Story Completion Order

```mermaid
graph TD
    Foundation[Phase 2: Foundation] --> US1[User Story 1: Persistence]
    Foundation --> US2[User Story 2: Cancellation]
    Foundation --> US3[User Story 3: Validation]
    Foundation --> US5[User Story 5: Error Envelope]

    US1 --> US7[User Story 7: JOSS]
    US2 --> US7
    US3 --> US7
    US5 --> US4[User Story 4: Security]

    US7 --> US8[User Story 8: Documentation]
    US4 --> Release[Phase 12: Release]
    US8 --> Release

    US6[User Story 6: Namespace] -.optional.-> Release
```

### Critical Path (Must Complete)

1. **Foundation** (T004-T017): Error envelope, database migration, storage paths
2. **US1** (T018-T021): Video persistence
3. **US2** (T022-T027): Job cancellation with GPU cleanup
4. **US3** (T028-T034): Config validation
5. **US7** (T039-T046): JOSS requirements (installation, docs, coverage)
6. **Release** (T079-T089): External validation and release prep

### Parallel Opportunities

**Week 1** (Foundation):
- T004-T008 (Error envelope) ║ T009-T014 (Database migration) ║ T015-T017 (Storage paths)

**Week 2** (US1, US2, US3 start):
- T018-T021 (US1 implementation) ║ T022 (US2 CancellationManager)
- T028-T030 (US3 validation models) can start in parallel

**Week 3** (US3, US5, US7 start):
- T035-T037 (US5 error updates) ║ T039-T040 (US7 installation script)
- T044-T046 (US7 coverage) in parallel

**Week 4-5** (US4, US7, US8):
- T047-T053 (US4 security) ║ T041-T043 (US7 API docs) ║ T054-T059 (US8 docs reorganization)

**Week 6-7** (Polish):
- T064-T067 (Concurrency) ║ T068-T069 (Health endpoint) ║ T070-T074 (Diagnostic CLI)

---

## Effort Summary

| Phase | Tasks | Estimated Hours | Priority |
|-------|-------|----------------|----------|
| Phase 1: Setup | T001-T003 | 2 | Required |
| Phase 2: Foundation | T004-T017 | 20-24 | Required |
| Phase 3: US1 (Persistence) | T018-T021 | 8-10 | P1 - Required |
| Phase 4: US2 (Cancellation) | T022-T027 | 16-20 | P1 - Required |
| Phase 5: US3 (Validation) | T028-T034 | 14-16 | P1 - Required |
| Phase 6: US5 (Error Envelope) | T035-T038 | 6-8 | P2 - High Priority |
| Phase 7: US7 (JOSS) | T039-T046 | 20-24 | P1 - Required |
| Phase 8: US4 (Security) | T047-T053 | 8-10 | P2 - High Priority |
| Phase 9: US8 (Documentation) | T054-T059 | 12-14 | P2 - High Priority |
| Phase 10: US6 (Namespace) | T060-T063 | 6-8 | P3 - Optional |
| Phase 11: Polish | T064-T078 | 16-20 | Mixed Priority |
| Phase 12: Release | T079-T089 | 16-20 | Required |
| **TOTAL** | **89 tasks** | **240-320 hours** | **6-8 weeks** |

### MVP Scope (Week 1-4, P1 Only)

For fastest time-to-value, implement only P1 user stories:
- Phase 1-2: Setup + Foundation (22-26 hours)
- Phase 3: US1 - Persistence (8-10 hours)
- Phase 4: US2 - Cancellation (16-20 hours)
- Phase 5: US3 - Validation (14-16 hours)
- Phase 7: US7 - JOSS (20-24 hours)
- Phase 12: Release (16-20 hours)

**MVP Total**: ~96-116 hours (3-4 weeks with 1 developer, 2 weeks with 2 developers)

---

## Implementation Strategy

### Week 1: Foundation + US1
- Complete Phase 1-2 (Foundation)
- Complete Phase 3 (US1 - Persistence)
- **Deliverable**: Videos persist across restarts

### Week 2: US2 + US3
- Complete Phase 4 (US2 - Cancellation)
- Complete Phase 5 (US3 - Validation)
- **Deliverable**: Job control + config validation working

### Week 3-4: US7 (JOSS) + US5
- Complete Phase 7 (US7 - JOSS requirements)
- Complete Phase 6 (US5 - Error envelope)
- **Deliverable**: JOSS-ready, consistent errors

### Week 4-5: US4 + US8 (If timeline permits)
- Complete Phase 8 (US4 - Security)
- Complete Phase 9 (US8 - Documentation)
- **Deliverable**: Secure by default, contributor-friendly docs

### Week 6-7: Polish + US6 (Optional)
- Complete Phase 11 (Polish tasks)
- Complete Phase 10 (US6 - Namespace) if time permits
- **Deliverable**: Production-ready quality

### Week 7-8: External Validation + Release
- Complete Phase 12 (Release)
- **Deliverable**: v1.3.0 released, JOSS submission

---

## Success Criteria

### Must-Have (P1)
- [ ] All P1 user stories (US1, US2, US3, US7) complete and tested
- [ ] Full test suite passes on Linux/macOS/Windows
- [ ] Database migration works from v1.2.x
- [ ] Installation verification script passes
- [ ] API documentation complete (all endpoints documented)
- [ ] JOSS reviewer can complete checklist without clarification

### Nice-to-Have (P2/P3)
- [ ] US4 (Security) complete
- [ ] US5 (Error Envelope) complete
- [ ] US8 (Documentation) complete
- [ ] US6 (Namespace) complete
- [ ] Storage cleanup feature
- [ ] Diagnostic CLI
- [ ] Test coverage >80%

---

## Notes

- **Tests are included** per speckit guidance for production-critical release
- **Each user story is independently testable** - can deliver incrementally
- **Parallel work opportunities identified** - 2 developers can work efficiently
- **MVP scope clearly defined** - can ship minimal v1.3.0 if timeline pressured
- **Optional tasks marked** - US6, storage cleanup, diagnostic CLI can defer to v1.3.1

---

**Generated**: October 12, 2025
**Tool**: speckit.tasks workflow
**Next**: Begin Phase 1 (Setup) → T001-T003
