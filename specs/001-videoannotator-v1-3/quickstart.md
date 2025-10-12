# Implementation Quickstart: VideoAnnotator v1.3.0

**Date**: October 11, 2025
**Status**: Implementation Guide
**For**: Development Team

---

## Purpose

This quickstart provides **ordered implementation guidance** for v1.3.0, ensuring:
- Dependencies are built in correct sequence
- Testing happens at appropriate milestones
- Parallel work streams don't conflict
- Critical path items complete first

**NOT a substitute for**: Detailed task breakdown (coming in Phase 2 tasks generation)

---

## Phase Overview

| Phase | Focus | Duration | Risk |
|-------|-------|----------|------|
| **1A: Foundation** | Error envelope, DB migration, storage paths | Week 1-2 | LOW |
| **1B: Core Features** | Job cancellation, config validation, concurrency | Week 2-3 | MEDIUM |
| **1C: JOSS Readiness** | Installation checks, API docs, test coverage | Week 3-4 | LOW |
| **1D: Security** | Auth default-on, CORS restriction | Week 4-5 | LOW |
| **1E: Documentation** | Docs reorganization, troubleshooting, reviewer guide | Week 5-6 | LOW |
| **1F: Cleanup** | Script audit, diagnostic CLI, storage cleanup | Week 6-7 | LOW |
| **External Validation** | Reviewer testing, JOSS mock review | Week 7-8 | MEDIUM |

---

## Phase 1A: Foundation (Week 1-2)

**Goal**: Establish structural changes that all other features depend on

### 1A.1: Error Envelope (2 days)

**Why first**: All new endpoints return standardized errors

```
Files to Create:
- src/api/v1/errors.py          # ErrorEnvelope, ErrorDetail models
- src/api/v1/exceptions.py      # Custom exception classes
- tests/unit/api/test_errors.py # Error serialization tests

Files to Modify:
- src/api/v1/main.py            # Register global exception handlers
- src/api/v1/jobs.py            # Use new exception classes

Key Tasks:
1. Define ErrorDetail + ErrorEnvelope Pydantic models (per data-model.md)
2. Create exception hierarchy:
   - VideoAnnotatorException (base)
   - JobNotFoundException (404)
   - PipelineNotFoundException (404)
   - InvalidConfigException (400)
   - JobAlreadyCompletedException (409)
   - InternalServerException (500)
3. Register FastAPI exception handlers (catch VideoAnnotatorException → ErrorEnvelope)
4. Write tests for error serialization (JSON matches OpenAPI schema)

Success Criteria:
- All exceptions inherit from VideoAnnotatorException
- Handlers return ErrorEnvelope with correct HTTP status
- Tests cover 404, 400, 409, 500 cases
- OpenAPI docs auto-generate error schemas
```

**Validation**: Run `uv run pytest tests/unit/api/test_errors.py -v`

### 1A.2: Database Schema Migration (2 days)

**Why second**: Job model changes needed for persistence + cancellation

```
Files to Create:
- src/database/migrations.py       # migrate_to_v1_3_0() function
- tests/unit/database/test_migrations.py

Files to Modify:
- src/database/models.py           # Add storage_path, cancelled_at to Job
- src/database/connection.py       # Run migration on startup
- src/api/v1/main.py               # Initialize DB on app startup

Key Tasks:
1. Add JobStatus.CANCELLED enum value
2. Add storage_path (TEXT, NOT NULL) column to Job
3. Add cancelled_at (TIMESTAMP, NULL) column to Job
4. Create migrate_to_v1_3_0() function:
   - Check schema_version in metadata table
   - Add missing columns (ALTER TABLE)
   - Set default storage_path for existing jobs
   - Update schema_version to "1.3.0"
5. Call migration on app startup (before accepting requests)

Success Criteria:
- Fresh v1.3.0 install creates correct schema
- v1.2.0 database migrates without data loss
- Existing jobs get default storage_path = /tmp/{job_id}
- Migration is idempotent (safe to re-run)
```

**Validation**:
```bash
# Test fresh install
rm test.db
uv run pytest tests/unit/database/test_migrations.py::test_fresh_install

# Test migration from v1.2.0
cp fixtures/v1.2.0.db test.db
uv run pytest tests/unit/database/test_migrations.py::test_migrate_from_v1_2_0
```

### 1A.3: Persistent Storage Paths (2 days)

**Why third**: Job submission needs to write to new persistent location

```
Files to Create:
- src/storage/paths.py              # Storage path utilities
- tests/unit/storage/test_paths.py

Files to Modify:
- src/config.py                     # Add STORAGE_ROOT config
- src/api/v1/jobs.py                # Use storage paths on job creation
- src/worker/executor.py            # Write results to storage_path

Key Tasks:
1. Add STORAGE_ROOT environment variable (default: ./custom_storage)
2. Create get_job_storage_path(job_id) function:
   - Returns: {STORAGE_ROOT}/jobs/{job_id}/
   - Creates directory if missing
   - Validates path is within STORAGE_ROOT (security check)
3. Update job creation:
   - Compute storage_path = get_job_storage_path(job_id)
   - Store in database
4. Update worker:
   - Write results to job.storage_path instead of /tmp/

Success Criteria:
- STORAGE_ROOT configurable via environment variable
- Job directories created with correct permissions
- Path traversal attempts rejected (security check)
- Results persist after server restart
```

**Validation**:
```bash
STORAGE_ROOT=/tmp/test_storage uv run pytest tests/unit/storage/test_paths.py
```

**Checkpoint**: Run full unit test suite
```bash
uv run pytest tests/unit/ -v
```

---

## Phase 1B: Core Features (Week 2-3)

**Goal**: Implement P1 functional requirements (job cancellation, config validation, concurrency)

### 1B.1: Job Cancellation Endpoint (3 days)

**Why first**: Highest-risk feature, needs extensive testing

```
Files to Create:
- src/worker/cancellation.py        # CancellationManager class
- tests/unit/worker/test_cancellation.py
- tests/integration/test_job_cancellation.py

Files to Modify:
- src/api/v1/jobs.py                # Add POST /jobs/{id}/cancel endpoint
- src/worker/executor.py            # Add cancellation flag checking
- src/database/models.py            # Add cancel() method to Job

Key Tasks:
1. Implement CancellationManager:
   - Track running jobs (job_id → multiprocessing.Process)
   - cancel_job(job_id) method:
     a. Send SIGTERM to process
     b. Wait 5 seconds for graceful shutdown
     c. If still alive, send SIGKILL
     d. Explicitly release GPU (torch.cuda.empty_cache in worker)
2. Add POST /api/v1/jobs/{job_id}/cancel endpoint:
   - Check job exists (404 if not)
   - Check job status (200 if already terminal)
   - Call CancellationManager.cancel_job()
   - Update job: status=CANCELLED, cancelled_at=now
   - Return CancellationResponse
3. Update worker:
   - Check cancellation flag between pipeline stages
   - Cleanup GPU on SIGTERM (signal handler)
4. Write tests:
   - Unit: CancellationManager signal logic
   - Integration: Full API → worker cancellation flow

Success Criteria:
- Cancelling RUNNING job stops worker within 5s
- GPU memory released (verify with nvidia-smi)
- Cancelling COMPLETED job returns 200 (idempotent)
- Cancelling non-existent job returns 404
```

**Validation**:
```bash
# Unit tests (fast)
uv run pytest tests/unit/worker/test_cancellation.py -v

# Integration tests (requires GPU, slow)
uv run pytest tests/integration/test_job_cancellation.py -v --gpu
```

### 1B.2: Config Validation Module (2 days)

**Why second**: Needed for validation endpoint + pre-submission checks

```
Files to Create:
- src/validation/validator.py       # ConfigValidator class
- src/validation/models.py          # ValidationResult, FieldError models
- tests/unit/validation/test_validator.py

Files to Modify:
- src/api/v1/pipelines.py           # Add POST /pipelines/validate endpoint

Key Tasks:
1. Create ConfigValidator class:
   - load_schema(pipeline_name) → Pydantic model (from registry)
   - validate(pipeline_name, config) → ValidationResult
   - Cache schemas (avoid re-parsing YAML)
2. Implement POST /api/v1/pipelines/validate endpoint:
   - Accept ValidationRequest (pipeline + config)
   - Call validator.validate()
   - Return ValidationResult (200 OK even if valid=false)
3. Add validation to job submission:
   - Validate config before creating job
   - Return 400 + ValidationResult if invalid
4. Write tests:
   - Valid config returns valid=true
   - Missing required field returns FieldError
   - Out-of-range value returns FieldError with hint
   - Unknown pipeline returns PIPELINE_NOT_FOUND

Success Criteria:
- Validation completes in <200ms (cached schemas)
- Field-level errors include hints
- Job submission rejects invalid configs
- Validation endpoint returns 200 for both valid/invalid (not 400)
```

**Validation**:
```bash
uv run pytest tests/unit/validation/ -v
curl -X POST http://localhost:8000/api/v1/pipelines/validate \
  -H "Content-Type: application/json" \
  -d '{"pipeline": "face", "config": {"detection": {"confidence_threshold": 1.5}}}'
```

### 1B.3: Concurrent Job Limiting (1 day)

**Why third**: Quick win, prevents resource exhaustion

```
Files to Create:
- tests/integration/test_concurrency.py

Files to Modify:
- src/worker/queue.py               # Add MAX_CONCURRENT_JOBS check
- src/config.py                     # Add MAX_CONCURRENT_JOBS config

Key Tasks:
1. Add MAX_CONCURRENT_JOBS environment variable (default: 2)
2. Update worker queue logic:
   - Before starting job, count RUNNING jobs
   - If count >= MAX_CONCURRENT_JOBS, skip (leave PENDING)
   - Retry on next poll
3. Write integration test:
   - Submit 5 jobs with MAX_CONCURRENT_JOBS=2
   - Verify only 2 run concurrently
   - Verify remaining 3 stay PENDING
   - Verify all eventually complete

Success Criteria:
- Concurrency limit respected
- Queue processes fairly (FIFO order)
- Configuration via environment variable
```

**Validation**:
```bash
MAX_CONCURRENT_JOBS=2 uv run pytest tests/integration/test_concurrency.py -v
```

**Checkpoint**: Run integration test suite
```bash
uv run pytest tests/integration/ -v
```

---

## Phase 1C: JOSS Readiness (Week 3-4)

**Goal**: Prepare for external review (installation, documentation, test coverage)

### 1C.1: Installation Verification Script (2 days)

**Why first**: Needed for "Getting Started" validation

```
Files to Create:
- scripts/verify_installation.py    # Progressive checks script
- tests/unit/scripts/test_verify_installation.py

Key Tasks:
1. Create verify_installation.py with checks:
   - Python version >= 3.10
   - FFmpeg installed (ffmpeg -version)
   - VideoAnnotator package importable
   - Database writable (test.db creation)
   - GPU available (optional, torch.cuda check)
   - Sample video processing (10-second clip)
2. Output format:
   - Plain text (ASCII-safe, no emoji)
   - Clear PASS/FAIL/SKIP for each check
   - Actionable suggestions on failure
   - Exit codes: 0=all pass, 1=critical fail, 2=warnings only
3. Platform detection:
   - Use platform.system() for OS-specific checks
   - Test on Linux, macOS, Windows WSL2
4. Write tests:
   - Mock failed checks, verify error messages
   - Verify exit codes

Success Criteria:
- Script runs on Linux/macOS/Windows WSL2
- Clear output for non-technical users
- Failures include fix suggestions (e.g., "Install FFmpeg: apt install ffmpeg")
- Sample processing validates core functionality
```

**Validation**:
```bash
uv run python scripts/verify_installation.py
# Should see all checks PASS

uv run pytest tests/unit/scripts/test_verify_installation.py -v
```

### 1C.2: API Documentation Enhancement (3 days)

**Why second**: Required for JOSS "Functionality" check

```
Files to Modify:
- src/api/v1/jobs.py                # Add detailed docstrings + examples
- src/api/v1/pipelines.py           # Add detailed docstrings + examples
- src/api/v1/health.py              # Add detailed docstrings + examples
- docs/usage/api_reference.md       # Auto-generated from docstrings

Key Tasks:
1. Add comprehensive docstrings to all endpoints:
   - Description (what it does)
   - Parameters (with examples)
   - Response examples (success + errors)
   - curl command examples
   - Common use cases
2. Use FastAPI @app.post(description="...") syntax
3. Ensure OpenAPI schema generation includes:
   - Request body examples
   - Response examples (per status code)
   - Error envelope schemas
4. Generate docs/usage/api_reference.md:
   - Script to extract OpenAPI spec → Markdown
   - Include curl examples for all endpoints
5. Write tests:
   - OpenAPI spec includes all endpoints
   - Each endpoint has at least one example

Success Criteria:
- Every endpoint has detailed docstring with examples
- curl commands are copy-pasteable
- Generated docs match OpenAPI spec
- No endpoints missing from documentation
```

**Validation**:
```bash
# Check OpenAPI spec completeness
curl http://localhost:8000/openapi.json | jq '.paths | keys'

# Regenerate docs
uv run python scripts/generate_api_docs.py

# Verify examples work
curl http://localhost:8000/api/v1/health?detailed=true
```

### 1C.3: Test Coverage Validation (2 days)

**Why third**: Required for JOSS "Software Quality" check

```
Files to Create:
- scripts/validate_coverage.py      # Coverage report generator
- docs/testing/coverage_report.md   # Auto-generated coverage report

Files to Modify:
- pytest.ini or pyproject.toml      # Add coverage config
- .github/workflows/test.yml        # Add coverage badge (if using CI)

Key Tasks:
1. Add pytest-cov configuration:
   - Target: >80% coverage for src/pipelines/
   - Target: >90% coverage for src/api/
   - Exclude: tests/, scripts/, examples/
2. Create validate_coverage.py script:
   - Run pytest with --cov
   - Parse coverage report
   - Generate Markdown table by module
   - Fail if coverage below thresholds
3. Add coverage badge to README.md (if using CI)
4. Document coverage in docs/testing/coverage_report.md

Success Criteria:
- Core pipelines (face, person, audio, scene) >80% covered
- API endpoints >90% covered
- Coverage report auto-generated and committed
- CI fails if coverage drops below threshold
```

**Validation**:
```bash
uv run pytest --cov=src --cov-report=term --cov-report=html
uv run python scripts/validate_coverage.py
# Should see PASS for all modules
```

**Checkpoint**: JOSS checklist validation
```bash
# Verify all JOSS requirements met
uv run python scripts/verify_installation.py  # Installation works
curl http://localhost:8000/docs                # API docs available
uv run python scripts/validate_coverage.py     # Coverage adequate
```

---

## Phase 1D: Security (Week 4-5)

**Goal**: Enable security features by default (v1.3.0 breaking change)

### 1D.1: Auth Default-On (2 days)

**Why first**: Most critical security change

```
Files to Modify:
- src/api/middleware/auth.py        # Make AUTH_REQUIRED=true default
- src/auth/token_manager.py         # Generate default token on first run
- docs/installation/quickstart.md   # Document token generation
- tests/api/test_auth.py            # Update tests for new default

Key Tasks:
1. Change AUTH_REQUIRED default from false → true
2. On first startup (no tokens.json):
   - Generate random API key
   - Print to console: "[START] Generated API key: {key}"
   - Write to tokens/tokens.json
   - Print instructions for use
3. Update all example curl commands to include X-API-Key header
4. Add docs/security/authentication.md:
   - How to retrieve API key
   - How to disable auth (SECURITY_ENABLED=false, discouraged)
   - Production recommendations
5. Update tests:
   - All API tests must provide auth

Success Criteria:
- Fresh install requires authentication
- API key auto-generated on first run
- Clear instructions printed to console
- Easy opt-out for development (SECURITY_ENABLED=false)
- All tests pass with auth enabled
```

**Validation**:
```bash
# Test with auth (should succeed)
API_KEY=$(uv run python get_api_key.py)
curl -H "X-API-Key: $API_KEY" http://localhost:8000/api/v1/health

# Test without auth (should fail 401)
curl http://localhost:8000/api/v1/health
```

### 1D.2: CORS Restriction (1 day)

**Why second**: Prevent unauthorized browser access

```
Files to Modify:
- src/api/middleware/cors.py        # Restrict CORS origins
- src/config.py                     # Add CORS_ORIGINS config
- docs/security/cors.md             # Document CORS configuration

Key Tasks:
1. Change CORS default from allow_origins=["*"] → ["http://localhost:3000"]
2. Add CORS_ORIGINS environment variable (comma-separated list)
3. Update docs with production recommendations:
   - Set CORS_ORIGINS to specific domains
   - Never use "*" in production
4. Write tests:
   - OPTIONS request from allowed origin succeeds
   - OPTIONS request from disallowed origin fails

Success Criteria:
- Default CORS only allows localhost:3000
- Production deployments can customize via env var
- Documentation warns against wildcard origins
```

**Validation**:
```bash
# Test allowed origin
curl -H "Origin: http://localhost:3000" \
     -H "Access-Control-Request-Method: POST" \
     -X OPTIONS http://localhost:8000/api/v1/jobs

# Test disallowed origin (should be rejected)
curl -H "Origin: http://evil.com" \
     -H "Access-Control-Request-Method: POST" \
     -X OPTIONS http://localhost:8000/api/v1/jobs
```

### 1D.3: Security Documentation (1 day)

**Why third**: Users need guidance on secure deployment

```
Files to Create:
- docs/security/production_checklist.md
- docs/security/authentication.md
- docs/security/cors.md

Key Tasks:
1. Create production_checklist.md:
   - Enable HTTPS (reverse proxy)
   - Set strong CORS_ORIGINS
   - Rotate API keys regularly
   - Use SECRET_KEY for JWT signing
   - Disable debug mode (DEBUG=false)
   - Configure storage retention
2. Add security section to README.md
3. Link from docs/README.md

Success Criteria:
- Production checklist covers all security features
- Clear migration guide from v1.2.x (auth now required)
- Examples show secure configuration
```

**Checkpoint**: Security audit
```bash
# Verify auth required by default
uv run python -c "from src.config import settings; assert settings.AUTH_REQUIRED"

# Verify CORS restricted
uv run python -c "from src.config import settings; assert '*' not in settings.CORS_ORIGINS"
```

---

## Phase 1E: Documentation (Week 5-6)

**Goal**: Reorganize docs for external contributors (JOSS reviewers)

### 1E.1: Documentation Reorganization (2 days)

**Why first**: Enables subsequent doc writing

```
Files to Create/Move:
docs/
  README.md                         # Navigation hub
  installation/
    quickstart.md                   # Getting started (<15 min)
    detailed_setup.md               # Full installation options
    troubleshooting.md              # Common issues
  usage/
    basic_workflow.md               # Submit job → get results
    api_reference.md                # Auto-generated from OpenAPI
    cli_reference.md                # videoannotator commands
  development/
    contributing.md                 # How to contribute
    architecture.md                 # System overview
    testing.md                      # Running tests
  deployment/
    production.md                   # Deployment guide
    docker.md                       # Docker setup
  testing/
    coverage_report.md              # Auto-generated coverage
    pipeline_validation.md          # Pipeline test results

Key Tasks:
1. Move existing docs to new structure
2. Update all internal links (use relative paths)
3. Create docs/README.md as navigation hub
4. Update main README.md to link to docs/
5. Write tests:
   - All links resolve (no 404s)
   - All referenced files exist

Success Criteria:
- Logical structure (user journey: install → use → develop)
- No broken links
- Each section has clear purpose
- Navigation hub makes docs discoverable
```

**Validation**:
```bash
# Check for broken links
uv run python scripts/validate_docs_links.py
```

### 1E.2: Troubleshooting Guide (2 days)

**Why second**: High-value for JOSS "Installation" check

```
Files to Create:
- docs/installation/troubleshooting.md

Key Tasks:
1. Document common issues:
   - FFmpeg not found → Install instructions per OS
   - GPU not detected → CUDA setup guide
   - Port 8000 already in use → Change port
   - Database locked → Permission fix
   - Out of disk space → Storage cleanup
   - Import errors → uv sync missing
2. Structure by symptom:
   - Error message → diagnosis → fix
   - Use <details> blocks for verbose output
3. Add diagnostic commands:
   - uv run videoannotator diagnose system
   - uv run videoannotator diagnose gpu
4. Link from README.md

Success Criteria:
- Top 10 user issues documented
- Each issue has clear fix steps
- Diagnostic commands provided
- Target: 80% issue self-resolution rate
```

### 1E.3: JOSS Reviewer Guide (1 day)

**Why third**: Makes JOSS review smooth

```
Files to Create:
- docs/GETTING_STARTED_REVIEWERS.md

Key Tasks:
1. Create fast-path review guide:
   - Installation (5 min): uv sync → verify script
   - Example usage (10 min): Submit sample job, view results
   - Architecture overview (5 min): Diagram + key concepts
   - Code exploration tips (5 min): Where to start reading
2. Link from paper.md (JOSS submission)
3. Test with external reviewer (Week 7)

Success Criteria:
- Reviewer can install + test in <15 minutes
- No assumed knowledge (explain uv, pipelines, etc.)
- Clear success criteria ("you should see...")
```

**Checkpoint**: External review simulation
```bash
# Follow GETTING_STARTED_REVIEWERS.md step-by-step
# Time each section, ensure <15 min total
```

---

## Phase 1F: Cleanup (Week 6-7)

**Goal**: Consolidate scripts, improve developer experience

### 1F.1: Scripts Audit (1 day)

**Why first**: Understand current state before cleanup

```
Files to Create:
- docs/development/scripts_inventory.md

Key Tasks:
1. Audit all scripts/ files:
   - What it does
   - Who uses it (dev/user/CI)
   - Last modified date
   - Keep/Migrate/Remove decision
2. Categorize:
   - Keep: Active, maintained scripts (test runners, docs generators)
   - Migrate: Move to CLI (demo scripts → videoannotator demo)
   - Remove: Obsolete, redundant, broken scripts
3. Document inventory in scripts_inventory.md

Success Criteria:
- Every script categorized
- Clear rationale for each decision
- No "unknown purpose" scripts
```

### 1F.2: Diagnostic CLI Implementation (2 days)

**Why second**: Consolidates multiple check scripts

```
Files to Create:
- src/diagnostics/system.py         # System checks
- src/diagnostics/gpu.py            # GPU checks
- src/diagnostics/storage.py        # Storage checks
- src/diagnostics/database.py       # Database checks
- tests/unit/diagnostics/            # Tests for each module

Files to Modify:
- src/cli.py                        # Add 'diagnose' command group

Key Tasks:
1. Implement diagnostic commands:
   - videoannotator diagnose system   (Python, FFmpeg, OS)
   - videoannotator diagnose gpu      (CUDA, device info, memory)
   - videoannotator diagnose storage  (free space, write test)
   - videoannotator diagnose database (schema version, connectivity)
   - videoannotator diagnose all      (run all checks)
2. Output format:
   - Plain text (ASCII-safe)
   - Optional --json flag for scripting
   - Exit codes: 0=all pass, 1=errors, 2=warnings
3. Use in troubleshooting docs

Success Criteria:
- All checks return structured output
- JSON output parseable by scripts
- Exit codes useful for CI/automation
- Replaces need for multiple check scripts
```

**Validation**:
```bash
uv run videoannotator diagnose all --json | jq
```

### 1F.3: Storage Cleanup Implementation (2 days)

**Why third**: Optional feature, can be deferred if time-constrained

```
Files to Create:
- src/storage/cleanup.py            # Cleanup logic
- tests/unit/storage/test_cleanup.py

Files to Modify:
- src/config.py                     # Add STORAGE_RETENTION_DAYS
- src/api/v1/main.py                # Add cleanup background task (if enabled)

Key Tasks:
1. Implement cleanup logic:
   - Find jobs completed > STORAGE_RETENTION_DAYS ago
   - Verify job in terminal state (COMPLETED/FAILED/CANCELLED)
   - Check storage_path exists
   - Move to archive or delete
   - Log all deletions (audit trail)
2. Safety checks:
   - Disabled by default (STORAGE_RETENTION_DAYS=null)
   - Dry-run mode available
   - Never delete RUNNING jobs
   - Backup prompt if >10GB deletion
3. Background task:
   - Run daily at 2 AM (configurable)
   - Log summary (X jobs cleaned, Y GB freed)

Success Criteria:
- Disabled by default (no surprises)
- Multiple safety checks (never lose data)
- Audit log for all deletions
- Dry-run mode for testing
```

**Validation**:
```bash
# Test dry-run
STORAGE_RETENTION_DAYS=7 uv run videoannotator storage cleanup --dry-run

# Test real cleanup (on test database)
STORAGE_RETENTION_DAYS=7 uv run videoannotator storage cleanup --confirm
```

**Checkpoint**: Script consolidation complete
```bash
# Verify diagnostic CLI works
uv run videoannotator diagnose all

# Verify obsolete scripts removed
ls scripts/ | wc -l  # Should be ~5-10 (down from 20+)
```

---

## Phase 2: External Validation (Week 7-8)

**Goal**: Validate with external users before release

### 2.1: External Reviewer Testing (3 days)

```
Tasks:
1. Recruit 2+ external reviewers:
   - 1 user (test installation + usage docs)
   - 1 developer (test contribution workflow)
   - 1 JOSS-familiar reviewer (mock review)
2. Provide docs/GETTING_STARTED_REVIEWERS.md
3. Collect feedback:
   - Installation issues
   - Unclear documentation
   - Missing features
4. Address critical feedback (P1 issues only)

Success Criteria:
- 80% of reviewers complete setup in <15 min
- No critical installation blockers
- Docs clarity rated >4/5
- JOSS reviewer gives "likely accept" signal
```

### 2.2: Integration Testing (2 days)

```
Tasks:
1. Run full test suite (all platforms):
   - Linux (Ubuntu 20.04, 22.04, 24.04)
   - macOS (Intel + Apple Silicon)
   - Windows WSL2
2. Test upgrade path:
   - Install v1.2.1
   - Submit test job
   - Upgrade to v1.3.0
   - Verify job persistence
   - Verify database migration
3. Performance regression testing:
   - Job submission latency (<200ms)
   - Cancellation latency (<5s)
   - Config validation latency (<200ms)

Success Criteria:
- All tests pass on all platforms
- Upgrade path works without data loss
- No performance regressions
```

### 2.3: Release Preparation (2 days)

```
Tasks:
1. Update CHANGELOG.md:
   - Breaking changes section (auth, CORS)
   - New features (cancellation, validation, diagnostics)
   - Bug fixes
   - Migration guide from v1.2.x
2. Create migration guide:
   - docs/UPGRADING_TO_v1.3.0.md
   - Cover auth setup, CORS configuration, storage path migration
3. Update version in src/version.py
4. Tag release: git tag v1.3.0
5. Create GitHub release with:
   - CHANGELOG excerpt
   - Installation instructions
   - Breaking changes warning
6. Create Zenodo archive (JOSS requirement)

Success Criteria:
- CHANGELOG comprehensive
- Migration guide tested by external reviewer
- Release artifacts published
- Zenodo DOI obtained
```

**Final Checkpoint**: Pre-JOSS submission validation
```bash
# Run full checklist
uv run python scripts/validate_joss_readiness.py
# Should see all checks PASS

# JOSS submission ready!
```

---

## Dependencies & Sequencing

### Critical Path
```
1A.1 (Error Envelope) → 1B.1 (Cancellation) → 1B.2 (Validation) → External Validation → Release
      ↓
1A.2 (DB Migration) → 1B.3 (Concurrency) → 1C.3 (Coverage) → Integration Testing
      ↓
1A.3 (Storage Paths) → 1F.3 (Cleanup)
```

### Parallel Work Streams
```
Stream A (Backend):    1A → 1B → 1D
Stream B (JOSS):       1C → 1E.3
Stream C (Docs):       1E.1 → 1E.2
Stream D (Cleanup):    1F.1 → 1F.2 → 1F.3
```

### Blocking Relationships
- 1B (Core Features) blocks on 1A (Foundation)
- 1C.2 (API Docs) blocks on 1A.1 (Error Envelope)
- 1C.3 (Coverage) blocks on 1B (Core Features)
- 1F.2 (Diagnostic CLI) blocks on 1C.1 (Verification Script)

---

## Testing Strategy

### Unit Tests (Fast, <1 min)
- Run after each file edit
- Target: >90% coverage for new code
- Mock external dependencies (GPU, filesystem)

### Integration Tests (Slow, ~5 min)
- Run after each phase completion
- Test cross-component interactions
- Use test fixtures (sample videos, configs)

### End-to-End Tests (Very Slow, ~15 min)
- Run before each checkpoint
- Full workflow: submit → run → cancel → retrieve
- Test all pipelines (face, person, audio, scene)

### External Validation (Slowest, ~3 days)
- Run once before release
- Real users, real environments
- Catch usability issues missed by developers

---

## Risk Mitigation

### Risk: GPU Cancellation Doesn't Release Memory
**Trigger**: Integration tests show memory leak after cancellation
**Mitigation**:
1. Add explicit torch.cuda.empty_cache() in worker cleanup
2. Test with nvidia-smi before/after cancellation
3. Add GPU memory check to diagnostic CLI
4. Document manual cleanup in troubleshooting guide

### Risk: Breaking Changes Break Existing Users
**Trigger**: External reviewer reports upgrade failure
**Mitigation**:
1. Provide clear migration guide (docs/UPGRADING_TO_v1.3.0.md)
2. Auto-generate API key on first v1.3.0 startup
3. Print breaking changes warning to console
4. Provide easy opt-out (SECURITY_ENABLED=false)

### Risk: JOSS Review Delayed
**Trigger**: Week 7 external reviewer gives "major issues" feedback
**Mitigation**:
1. Prioritize P1 issues only (defer P2/P3 to v1.3.1)
2. Extend timeline by 1 week (acceptable for quality)
3. Submit to JOSS with known minor issues (disclose in paper)

### Risk: Test Coverage Below 80%
**Trigger**: Coverage validation fails in Week 4
**Mitigation**:
1. Identify untested modules (likely edge cases)
2. Add smoke tests (not comprehensive, but improves percentage)
3. Document why some modules excluded (e.g., deprecated code)
4. Commit to 80% in v1.3.1 (not blocking for v1.3.0)

---

## Success Criteria (Gate to Release)

### Must-Have (P1)
- [ ] All P1 functional requirements implemented
- [ ] Full test suite passes on Linux/macOS/Windows
- [ ] Database migration works from v1.2.x
- [ ] Installation verification script passes
- [ ] API documentation complete (all endpoints documented)
- [ ] JOSS reviewer gives "likely accept" signal
- [ ] No critical security issues (auth enabled by default)

### Nice-to-Have (P2/P3)
- [ ] Storage cleanup feature (can defer to v1.3.1)
- [ ] Diagnostic CLI (can use separate scripts if time-constrained)
- [ ] Script consolidation complete (can defer cleanup)
- [ ] Test coverage >80% (may be 75% acceptable)

---

## Tools & Commands Reference

### Development
```bash
# Sync environment
uv sync

# Run tests (unit only)
uv run pytest tests/unit/ -v

# Run tests (integration, slow)
uv run pytest tests/integration/ -v

# Run tests (with coverage)
uv run pytest --cov=src --cov-report=term

# Run specific test file
uv run pytest tests/unit/api/test_errors.py -v

# Start API server
uv run videoannotator serve

# Run diagnostic checks
uv run videoannotator diagnose all
```

### Validation
```bash
# Installation verification
uv run python scripts/verify_installation.py

# Coverage validation
uv run python scripts/validate_coverage.py

# JOSS readiness check
uv run python scripts/validate_joss_readiness.py

# Docs link validation
uv run python scripts/validate_docs_links.py
```

### Release
```bash
# Update version
echo "1.3.0" > src/version.py

# Tag release
git tag v1.3.0
git push origin v1.3.0

# Build package
uv build

# Publish to PyPI (if applicable)
uv publish
```

---

## Next Steps

1. **Generate Task Breakdown** (Phase 2 of speckit workflow):
   - Break each phase into atomic tasks
   - Estimate effort (hours)
   - Assign owners (if team project)

2. **Update Agent Context**:
   - Run `.specify/scripts/bash/update-agent-context.sh copilot`
   - Ensure AI assistants aware of implementation plan

3. **Create GitHub Project Board** (optional):
   - Columns: To Do, In Progress, Review, Done
   - Cards for each phase (1A, 1B, etc.)
   - Track progress visually

4. **Start Phase 1A**:
   - Create feature branch: `git checkout -b feature/v1.3.0-foundation`
   - Begin with 1A.1 (Error Envelope)
   - Open draft PR for early feedback

---

**Status**: Quickstart complete, ready for implementation
**Next**: Start Phase 1A.1 (Error Envelope Implementation)
