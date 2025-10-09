# 🚀 VideoAnnotator v1.3.0 Development Roadmap

## Release Overview

VideoAnnotator v1.3.0 is the **Production Reliability & Critical Fixes** release, addressing blocking issues discovered during v1.2.x client integration testing. This release focuses on making the system production-ready by fixing data loss risks, job management problems, and security defaults.

**Target Release**: Q4 2025 (6-8 weeks)  
**Current Status**: Active Development  
**Main Goal**: Production-ready reliability for real-world research use

---

## 🎯 Core Principles

This release is **strictly scoped** to critical fixes only:
- ✅ Fix blocking issues preventing production use
- ✅ Address data loss and stability risks
- ✅ Secure by default configuration
- ✅ Standardize error handling
- ❌ NO new features or advanced capabilities
- ❌ NO architectural refactoring beyond namespace migration
- ❌ NO advanced ML, plugins, or enterprise features (deferred to v1.4.0+)

---

## 🔥 Critical Issues (From Client Team QA)

These issues were identified during Video Annotation Viewer v0.4.0 integration testing and **block production deployment**:

### 1. ❌ Pipeline Name Resolution Failures
**Issue**: Jobs failing with "Unknown pipeline: audio_processing"  
**Impact**: Core functionality broken, 8/8 test jobs failed  
**Root Cause**: Mismatch between pipeline names in registry vs actual implementations

### 2. ❌ Ephemeral Storage (Data Loss Risk)
**Issue**: Uploaded videos stored in `/tmp`, lost on server restart  
**Impact**: CRITICAL - User data loss, cannot retry failed jobs  
**Root Cause**: No persistent storage strategy

### 3. ❌ Cannot Stop Running Jobs
**Issue**: Delete endpoint only removes DB record, processing continues  
**Impact**: GPU OOM, server crashes, runaway jobs  
**Root Cause**: No job cancellation mechanism or concurrency control

### 4. ⚠️ Config Validation Always Returns Valid
**Issue**: Invalid configurations pass validation, jobs fail at runtime  
**Impact**: Poor UX, cryptic late-stage failures  
**Root Cause**: Validation only checks YAML syntax, not semantic correctness

### 5. ⚠️ Insecure Defaults
**Issue**: Optional authentication, permissive CORS  
**Impact**: Security risk in shared lab environments  
**Root Cause**: Development defaults inappropriate for production

### 6. ⚠️ Inconsistent Error Formats
**Issue**: Different endpoints return different error structures  
**Impact**: Client must handle multiple error patterns  
**Root Cause**: No standardized error envelope

---

## 📋 v1.3.0 Deliverables

### Phase 1: Critical Data & Stability Fixes (Weeks 1-2)

#### 1.1 Pipeline Registry Audit & Validation
- [ ] **Audit Script**: Compare registry metadata vs actual pipeline implementations
- [ ] **Startup Validation**: Fail fast if pipeline missing from registry
- [ ] **Name Mapping Fix**: Resolve `audio_processing` and any other name mismatches
- [ ] **Registry Test Coverage**: Add tests for each registered pipeline
- [ ] **Documentation**: Update pipeline naming conventions

**Acceptance Criteria**:
- Server starts only if all registered pipelines are loadable
- Clear error messages if pipeline missing
- `/api/v1/pipelines` returns only actually available pipelines

#### 1.2 Persistent Storage Implementation
- [ ] **Environment Variable**: `VIDEOANNOTATOR_STORAGE_DIR` (default: `./storage`)
- [ ] **Directory Structure**: `uploads/`, `results/`, `temp/`, `logs/`
- [ ] **Migration Path**: Move existing temp files to persistent storage
- [ ] **Storage Lifecycle**: Configurable retention policy (default: keep completed jobs 30 days)
- [ ] **Health Endpoint**: Add storage path and disk space to `/health`
- [ ] **Documentation**: Storage requirements and configuration guide

**Acceptance Criteria**:
- Uploaded videos persist across server restarts
- Configurable storage location via env var
- Automatic cleanup of old completed jobs (opt-in)
- Storage info visible in health endpoint

#### 1.3 Job Cancellation & Concurrency Control
- [ ] **Database Schema**: Add `CANCELLED` status to job state machine
- [ ] **Cancel Endpoint**: `POST /api/v1/jobs/{id}/cancel`
- [ ] **GPU Cleanup**: Release GPU memory on cancellation
- [ ] **Concurrency Limits**: `MAX_CONCURRENT_GPU_JOBS` env var (default: 2)
- [ ] **Job Queue**: Queue jobs when limit reached, show position
- [ ] **Worker Signal Handling**: Graceful shutdown on cancel signal
- [ ] **Tests**: Cancellation scenarios including mid-processing

**Acceptance Criteria**:
- Running jobs can be cancelled via API
- GPU memory released within 5 seconds of cancellation
- Server doesn't OOM with multiple concurrent jobs
- Queue position visible in job status

### Phase 2: Quality & Security Hardening (Weeks 3-4)

#### 2.1 Schema-Based Config Validation
- [ ] **Registry Integration**: Use `config_schema` from pipeline metadata for validation
- [ ] **Validation Endpoint**: `POST /api/v1/pipelines/{name}/validate` (accepts config, returns errors)
- [ ] **Field-Level Errors**: Return specific field path, error type, valid values
- [ ] **Job Submission Validation**: Validate before queuing (fail fast)
- [ ] **Error Messages**: Human-readable with examples of valid values
- [ ] **Tests**: Valid/invalid config scenarios for each pipeline

**Acceptance Criteria**:
- Invalid configs rejected at submission time
- Clear field-level error messages
- Validation endpoint available for pre-flight checks

#### 2.2 Secure-by-Default Configuration
- [ ] **Auth Required**: `VIDEOANNOTATOR_REQUIRE_AUTH=true` (default)
- [ ] **CORS Restrictions**: `ALLOWED_ORIGINS` env var (default: `http://localhost:19011`)
- [ ] **Security Warnings**: Log warnings on startup if insecure mode enabled
- [ ] **Documentation**: Security configuration guide with production checklist
- [ ] **Example Configs**: Secure production `.env.example`

**Acceptance Criteria**:
- Authentication required by default
- CORS restricted to configured origins
- Clear warnings in logs if security disabled

#### 2.3 Standardized Error Envelope
- [ ] **Error Schema**: Define standard error envelope structure
  ```json
  {
    "error": {
      "code": "PIPELINE_NOT_FOUND",
      "message": "Pipeline 'audio_processing' not found",
      "detail": "Available pipelines: person_tracking, face_analysis, audio_diarization",
      "hint": "Run 'videoannotator pipelines' to list available pipelines",
      "field": null,
      "timestamp": "2025-10-09T12:34:56Z"
    }
  }
  ```
- [ ] **Exception Handler**: Custom FastAPI exception handler for all errors
- [ ] **Error Code Taxonomy**: Define error codes (4xx, 5xx categories)
- [ ] **Endpoint Migration**: Update all endpoints to use error envelope
- [ ] **OpenAPI Spec**: Document error responses in schema
- [ ] **Tests**: Error format consistency tests

**Acceptance Criteria**:
- All endpoints return errors in standard envelope
- Error codes documented and consistent
- Client can parse errors reliably

### Phase 3: Technical Debt Resolution (Weeks 5-6)

#### 3.1 Package Namespace Migration
- [ ] **Directory Rename**: `src/` → `videoannotator/`
- [ ] **Import Updates**: Update all imports to `from videoannotator.pipelines`
- [ ] **Deprecation Shims**: Add compatibility layer for old imports (one release grace period)
- [ ] **Setup.py Update**: Package name and structure
- [ ] **Test Updates**: Update all test imports
- [ ] **Documentation**: Migration guide for users with custom integrations

**Acceptance Criteria**:
- Package importable as `videoannotator`
- Old imports work with deprecation warnings
- All tests pass with new imports

#### 3.2 Batch/Job Semantics Fixes (From v1.2.1 Debt)
- [ ] **Success Rate Fix**: Return `0.0` (not null) when total_pipelines == 0
- [ ] **Timestamp Refinement**: Add `started_at` distinct from `queued_at`
- [ ] **Retry Configuration**: Expose `RETRY_BASE_DELAY`, `RETRY_MAX_DELAY`, `RETRY_JITTER` env vars
- [ ] **Tests**: Edge cases (zero pipelines, retry timing)

**Acceptance Criteria**:
- Success rate calculation consistent
- Timestamps accurately reflect job lifecycle
- Retry backoff configurable

#### 3.3 Deterministic Test Fixtures
- [ ] **Synthetic Video Generator**: Create videos with known properties (frames, duration, audio)
- [ ] **Mock Video Capture**: Mock OpenCV capture to avoid file I/O in unit tests
- [ ] **Fixture Library**: Reusable test videos for different scenarios
- [ ] **Flaky Test Fixes**: Eliminate minimal-file errors in tests

**Acceptance Criteria**:
- Tests generate deterministic media
- No file I/O in unit tests
- Test suite runs reliably in CI

---

## 📊 Success Metrics

### Must-Have for Release
- [ ] Zero job failures due to pipeline naming
- [ ] Zero data loss on server restart
- [ ] All running jobs cancellable within 5 seconds
- [ ] Invalid configs rejected at submission time
- [ ] Authentication required by default
- [ ] All API errors use standard envelope
- [ ] Package namespace migrated

### Performance Targets
- [ ] Job submission latency < 200ms (with validation)
- [ ] Cancellation response time < 5s
- [ ] Storage cleanup overhead < 1% CPU
- [ ] Concurrent job limit prevents OOM

---

## 🚫 Explicitly Out of Scope for v1.3.0

The following are **intentionally deferred** to v1.4.0 or later:

### Deferred to v1.4.0 (First Public Release)
- Version info endpoint (`/api/v1/system/version`)
- Enhanced health endpoint details (GPU memory, storage stats, uptime)
- Comprehensive YAML loader edge case tests
- Legacy example deprecation and cleanup
- Structured logging (JSON format option)

### Deferred to v1.5.0+ (Advanced Features)
- Active learning system
- Quality assessment pipeline
- Multi-modal correlation analysis
- Plugin system architecture
- Real-time streaming
- GraphQL API
- Enterprise features (SSO, RBAC, multi-tenancy)
- Advanced analytics dashboard
- Cloud provider integration
- Microservice decomposition

---

## 📅 Release Schedule

### Week 1-2: Critical Fixes Sprint
**Focus**: Data loss prevention and job management  
**Deliverables**:
- Pipeline registry audit script
- Persistent storage implementation
- Job cancellation MVP

**Exit Criteria**:
- All registered pipelines loadable at startup
- Videos survive server restart
- Jobs can be cancelled via API

### Week 3-4: Quality & Security Sprint
**Focus**: Validation and secure defaults  
**Deliverables**:
- Schema-based config validation
- Secure-by-default configuration
- Standardized error envelope

**Exit Criteria**:
- Invalid configs rejected at submission
- Authentication required by default
- All errors use standard format

### Week 5-6: Technical Debt Sprint
**Focus**: Namespace migration and test reliability  
**Deliverables**:
- Package namespace migration (`videoannotator/`)
- Batch/job semantics fixes
- Deterministic test fixtures

**Exit Criteria**:
- Package importable as `videoannotator`
- Success rate calculations correct
- Test suite runs reliably in CI

### Week 7-8: Beta Testing & Documentation
**Focus**: Integration testing and release prep  
**Deliverables**:
- Client team integration testing
- Migration guide for v1.2.x → v1.3.0
- Security configuration documentation

**Exit Criteria**:
- Client team signs off on fixes
- Documentation complete
- Release notes finalized

---

## 🧪 Testing Strategy

### Unit Tests
- Pipeline registry validation logic
- Storage lifecycle management
- Job cancellation state transitions
- Config validation rules
- Error envelope formatting

### Integration Tests
- End-to-end job submission with validation
- Persistent storage across restarts
- Job cancellation during processing
- Authentication enforcement
- CORS restriction behavior

### Manual QA Checklist
- [ ] Submit job with invalid pipeline name → clear error
- [ ] Restart server → uploaded videos still accessible
- [ ] Cancel running GPU job → memory released
- [ ] Submit invalid config → field-level errors returned
- [ ] Access API without token → 401 (when auth required)
- [ ] CORS from unauthorized origin → blocked

---

## 📖 Documentation Updates

### User-Facing Docs
- [ ] **Migration Guide**: v1.2.x → v1.3.0 upgrade steps
- [ ] **Security Guide**: Production deployment security checklist
- [ ] **Storage Guide**: Persistent storage configuration and management
- [ ] **Troubleshooting**: Common issues and solutions

### Developer Docs
- [ ] **Architecture**: Updated diagrams with storage and queue components
- [ ] **API Reference**: Error envelope format and codes
- [ ] **Contributing**: Updated import paths and package structure

### Release Materials
- [ ] **Release Notes**: Breaking changes and migration steps
- [ ] **Announcement**: Blog post or mailing list message
- [ ] **Demo Video**: Showcasing new reliability features

---

## 🔍 Monitoring & Rollback Plan

### Health Checks
- Pipeline registry loaded successfully
- Storage directory accessible and writable
- Database connection healthy
- GPU availability (if configured)
- Queue system operational

### Rollback Triggers
- Critical bugs discovered in production
- Data loss or corruption detected
- Performance degradation > 50%
- Security vulnerability identified

### Rollback Procedure
1. Revert to v1.2.2 container/package
2. Restore database backup (schema compatible)
3. Re-enable optional authentication if needed
4. Notify users of temporary rollback

---

## 🤝 Stakeholder Communication

### Client Team Coordination
- **Weekly Sync**: Progress updates and blocker discussion
- **Beta Access**: Early v1.3.0-rc builds for testing
- **Feedback Loop**: Issue triage and priority adjustment

### User Communication
- **Deprecation Notices**: Advance warning of breaking changes
- **Migration Support**: Office hours for upgrade assistance
- **Security Advisories**: Clear communication of security improvements

---

## 🧩 Technical Debt Resolution (From v1.2.1)

### Package Layout Normalization
Interim absolute-import flattening was applied after v1.2.1 to fix runtime errors. v1.3.0 completes the full package restructuring.

**Planned Actions**:
1. Introduce canonical package directory `videoannotator/` (migrate modules incrementally)
2. Add deprecation shims for legacy top-level imports (one minor release grace)
3. Enforce import policy (no multi-level relative imports) via lint/CI script
4. Separate optional heavy ML deps into extras: `[ml]`, `[face]`, `[audio]`
5. Generate import graph; fail CI on new cycles
6. Confirm parity of editable vs wheel installs

**Success Criteria**:
- All public imports under `videoannotator.*`
- Wheel + editable produce identical module tree
- No runtime import errors in `videoannotator server` start
- Deprecation warnings emitted for legacy paths (removed in ≥ v1.4.0)

### Batch/Job Semantics
1. Clarify `success_rate` when total pipelines == 0 (return 0.0 consistently)
2. Add `started_at` distinct from `queued_at` timestamps
3. Expose configurable retry backoff (base delay, max delay, jitter)

### Test Media Fixtures
- Deterministic synthetic video generation (frames, duration, optional audio)
- Mock OpenCV capture to eliminate flaky minimal-file errors
- Reusable fixture library for different test scenarios

### Storage Lifecycle
- Consistent cleanup of temp job directories & orphaned artifacts
- Retention policy scaffold (dry-run logging; disabled by default)
- Storage monitoring and alerting

---

**Last Updated**: October 9, 2025  
**Target Release**: Q4 2025 (6-8 weeks)  
**Status**: Active Development - Critical fixes phase