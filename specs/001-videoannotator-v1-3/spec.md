# Feature Specification: VideoAnnotator v1.3.0 Production Reliability & Critical Fixes

**Feature Branch**: `001-videoannotator-v1-3`
**Created**: October 11, 2025
**Status**: Draft
**Input**: User description: "VideoAnnotator v1.3.0 Production Reliability & Critical Fixes Release"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Upload Video Without Data Loss (Priority: P1)

As a researcher, I need to upload a video file for processing and be confident that my video and results will persist even if the server restarts, so I don't lose hours of processing work and can retry failed jobs.

**Why this priority**: CRITICAL - Data loss blocks production use entirely. Researchers cannot trust the system if their work disappears on server restart.

**Independent Test**: Upload a video, restart the server, verify video and job metadata still accessible. Delivers core persistence guarantee.

**Acceptance Scenarios**:

1. **Given** I upload a 500MB video file, **When** the server restarts before processing completes, **Then** the video file is still available and job can resume or be retried
2. **Given** I have completed jobs with results, **When** the server restarts, **Then** all results remain downloadable
3. **Given** I configure a custom storage directory, **When** I upload videos, **Then** they are stored in my configured location (not /tmp)
4. **Given** the storage directory doesn't exist, **When** the server starts, **Then** it creates the directory structure automatically

---

### User Story 2 - Stop Runaway Jobs (Priority: P1)

As a lab administrator, I need to cancel running GPU jobs that are stuck or consuming too many resources, so I can prevent server crashes and free up resources for other researchers.

**Why this priority**: CRITICAL - Prevents GPU OOM crashes and enables resource management in shared lab environments.

**Independent Test**: Start a GPU job, cancel it mid-processing, verify GPU memory is released. Delivers core job control capability.

**Acceptance Scenarios**:

1. **Given** a job is running on GPU, **When** I cancel it via the API, **Then** processing stops within 5 seconds and GPU memory is released
2. **Given** I delete a job that is queued but not started, **When** the deletion completes, **Then** the job never starts processing
3. **Given** I cancel a job with multiple pipelines, **When** cancellation is requested, **Then** all pipeline processes are terminated
4. **Given** a cancelled job, **When** I check its status, **Then** it shows "CANCELLED" status with cancellation timestamp

---

### User Story 3 - Submit Jobs with Valid Configurations (Priority: P1)

As a researcher, I need to know immediately if my configuration is invalid before submitting a job, so I don't waste time waiting for a job to fail late in processing.

**Why this priority**: CRITICAL - Poor UX blocks adoption. Invalid configs currently pass validation and fail cryptically at runtime.

**Independent Test**: Submit configs with missing required fields, invalid values, and valid configs. Verify immediate, clear feedback. Delivers core validation capability.

**Acceptance Scenarios**:

1. **Given** I submit a config with an invalid pipeline name, **When** I submit the job, **Then** I receive immediate error with list of valid pipeline names
2. **Given** I submit a config missing required fields, **When** validation runs, **Then** I receive field-level errors specifying which fields are missing
3. **Given** I submit a config with invalid field values, **When** validation runs, **Then** I receive error with valid value ranges or options
4. **Given** I have a complex config, **When** I use the validation endpoint before submission, **Then** I receive detailed validation results without submitting the job

---

### User Story 4 - Work in Secure Lab Environment (Priority: P2)

As a lab administrator, I need authentication enabled by default and CORS restricted to known origins, so unauthorized users cannot access our research data or consume our GPU resources.

**Why this priority**: HIGH - Security is essential for shared lab environments and multi-user scenarios, but doesn't block single-user research workflows.

**Independent Test**: Try to access API without credentials, verify rejection. Configure allowed origins, verify CORS enforcement. Delivers core security posture.

**Acceptance Scenarios**:

1. **Given** a fresh server installation, **When** I start the server, **Then** authentication is required by default (no anonymous access)
2. **Given** CORS is configured for specific origins, **When** a request comes from unauthorized origin, **Then** the request is blocked
3. **Given** I need to disable auth for local development, **When** I set VIDEOANNOTATOR_REQUIRE_AUTH=false, **Then** server logs prominent security warning
4. **Given** I configure multiple allowed origins, **When** requests come from those origins, **Then** all are accepted with proper CORS headers

---

### User Story 5 - Understand API Errors Consistently (Priority: P2)

As a frontend developer integrating with VideoAnnotator, I need all API errors to follow the same format, so my error handling code works reliably across all endpoints.

**Why this priority**: HIGH - Enables reliable client integration. Currently forces clients to handle multiple error patterns, but doesn't block basic usage.

**Independent Test**: Trigger errors from different endpoints (404, 400, 500), verify all use standard envelope format. Delivers consistent error contract.

**Acceptance Scenarios**:

1. **Given** I request a non-existent job, **When** the API responds, **Then** the error follows standard envelope format with error code, message, and hint
2. **Given** I submit invalid data, **When** validation fails, **Then** the error includes field-level detail in standard format
3. **Given** the server encounters an internal error, **When** the error is returned, **Then** it includes error code, timestamp, and reference for support
4. **Given** I integrate with multiple endpoints, **When** errors occur, **Then** my single error parser handles all responses

---

### User Story 6 - Import Package with Standard Namespace (Priority: P3)

As a developer extending VideoAnnotator, I need to import from a standard package namespace (e.g., `from videoannotator.pipelines`), so my code follows Python conventions and doesn't break on updates.

**Why this priority**: MEDIUM - Technical debt that improves maintainability but doesn't affect end-user functionality. Can be deferred if timeline pressures exist.

**Independent Test**: Import from `videoannotator.*` namespace, verify old `src.*` imports show deprecation warnings. Delivers namespace normalization.

**Acceptance Scenarios**:

1. **Given** I install VideoAnnotator as a package, **When** I import `from videoannotator.pipelines`, **Then** the import succeeds without errors
2. **Given** I have legacy code using `from src.pipelines`, **When** I run it, **Then** it works but logs a deprecation warning
3. **Given** I install VideoAnnotator in editable mode, **When** I import the package, **Then** behavior matches wheel installation
4. **Given** I follow the migration guide, **When** I update my imports, **Then** migration completes in under 30 minutes

---

### User Story 7 - Submit Software for JOSS Publication (Priority: P1)

As a project maintainer, I need VideoAnnotator to meet JOSS review criteria (installation verification, comprehensive API docs, test coverage), so reviewers can successfully evaluate and approve our software paper submission.

**Why this priority**: CRITICAL - JOSS publication is release goal; reviewer friction blocks acceptance. All requirements must be verifiable by external reviewers.

**Independent Test**: External reviewer follows README to install, run tests, and execute sample pipeline. Completes JOSS checklist without clarification requests. Delivers publication-ready quality.

**Acceptance Scenarios**:

1. **Given** a reviewer runs `scripts/verify_installation.py`, **When** installation is correct, **Then** script validates dependencies and runs sample pipeline successfully
2. **Given** a reviewer examines API documentation, **When** checking endpoint documentation, **Then** all endpoints have docstrings with curl examples
3. **Given** a reviewer runs test suite, **When** executing `uv run pytest`, **Then** tests pass with >80% coverage for core pipelines
4. **Given** a reviewer follows troubleshooting guide, **When** encountering common issues, **Then** guide provides actionable solutions

---

### User Story 8 - Navigate Documentation as New Contributor (Priority: P2)

As an open-source contributor, I need clearly organized documentation that separates user guides, contributor guides, and troubleshooting, so I can quickly find relevant information without reading everything.

**Why this priority**: HIGH - Good documentation accelerates community contributions and reduces maintainer support burden, but doesn't block basic usage.

**Independent Test**: New contributor submits first PR within 2 hours using only CONTRIBUTING.md guidance. Delivers contributor-friendly documentation.

**Acceptance Scenarios**:

1. **Given** I want to fix a bug, **When** I read CONTRIBUTING.md, **Then** I can set up dev environment, run tests, and create PR without external help
2. **Given** I encounter an installation error, **When** I check troubleshooting guide, **Then** I find my specific error with clear resolution steps
3. **Given** I want to understand VideoAnnotator's architecture, **When** I browse docs/, **Then** I can navigate from high-level overview to detailed API references
4. **Given** I'm a JOSS reviewer, **When** I use "Getting Started for Reviewers" guide, **Then** I can verify all review checklist items in under 30 minutes

**Independent Test**: Install package, import from `videoannotator.*` namespace, verify all modules accessible. Delivers standard Python package structure.

**Acceptance Scenarios**:

1. **Given** I install VideoAnnotator as a package, **When** I import `from videoannotator.pipelines`, **Then** the import succeeds
2. **Given** I have code using old `src.pipelines` imports, **When** I upgrade to v1.3.0, **Then** I see deprecation warnings but code still works
3. **Given** I follow migration guide, **When** I update imports to new namespace, **Then** deprecation warnings disappear
4. **Given** I install via wheel or editable mode, **When** I import modules, **Then** behavior is identical

---

### Edge Cases

- What happens when storage directory runs out of disk space during upload?
- How does system handle cancellation request for already-completed job?
- What if pipeline name exists in registry but fails to load at startup?
- How does validation handle deeply nested configuration structures?
- What happens when concurrent job limit is reached and queue is full?
- How does system behave when GPU is available at startup but becomes unavailable during processing?
- What if configured CORS origins list is empty or malformed?
- How does error envelope handle errors that occur before request parsing completes?

## Requirements *(mandatory)*

### Functional Requirements

#### Pipeline Registry & Validation (P1)

- **FR-001**: System MUST validate at startup that all registered pipelines are loadable and functional
- **FR-002**: System MUST fail to start with clear error message if any registered pipeline cannot be loaded
- **FR-003**: `/api/v1/pipelines` endpoint MUST return only pipelines that are actually available and functional
- **FR-004**: System MUST provide audit script that compares registry metadata against actual pipeline implementations
- **FR-005**: System MUST log warning if pipeline implementation name doesn't match registry name
- **FR-006**: System MUST include pipeline availability status in health endpoint

#### Persistent Storage (P1)

- **FR-007**: System MUST store uploaded videos in persistent storage location (not /tmp)
- **FR-008**: System MUST support `VIDEOANNOTATOR_STORAGE_DIR` environment variable for configurable storage location (default: `./storage`)
- **FR-009**: System MUST create directory structure (`uploads/`, `results/`, `temp/`, `logs/`) if it doesn't exist
- **FR-010**: System MUST persist job metadata (status, config, timestamps) across server restarts
- **FR-011**: System MUST make completed job results downloadable after server restart
- **FR-012**: System MUST support configurable retention policy for completed jobs (default: 30 days, opt-in cleanup)
- **FR-013**: Health endpoint MUST include storage path, available disk space, and directory status

#### Job Cancellation & Concurrency (P1)

- **FR-014**: Database schema MUST include `CANCELLED` status in job state machine
- **FR-015**: System MUST provide `POST /api/v1/jobs/{id}/cancel` endpoint
- **FR-016**: System MUST immediately terminate all pipelines (in-progress and queued) within 5 seconds of cancellation request
- **FR-017**: System MUST release GPU memory within 5 seconds of job cancellation
- **FR-018**: System MUST support `MAX_CONCURRENT_GPU_JOBS` environment variable (default: 2)
- **FR-019**: System MUST queue jobs when concurrent limit is reached
- **FR-020**: Job status endpoint MUST show queue position for queued jobs
- **FR-021**: System MUST handle graceful shutdown on cancellation signals (SIGTERM, SIGINT)
- **FR-022**: System MUST prevent cancelled jobs from being restarted

#### Configuration Validation (P1)

- **FR-023**: System MUST validate job configurations against pipeline schema before job submission
- **FR-024**: System MUST provide `POST /api/v1/pipelines/{name}/validate` endpoint that accepts config and returns validation errors
- **FR-025**: Validation errors MUST include field path, error type, and valid value examples
- **FR-026**: System MUST reject invalid configurations at submission time (fail fast)
- **FR-027**: Validation errors MUST be human-readable with examples of correct values
- **FR-028**: System MUST use `config_schema` from pipeline registry metadata for validation

#### Security Configuration (P2)

- **FR-029**: System MUST require authentication by default. Local development can disable via `VIDEOANNOTATOR_REQUIRE_AUTH=false`
- **FR-030**: System MUST support `ALLOWED_ORIGINS` environment variable for CORS configuration (default: `http://localhost:19011`)
- **FR-031**: System MUST restrict CORS to configured origins only
- **FR-032**: System MUST log prominent security warning if authentication is disabled
- **FR-033**: System MUST log security configuration status at startup (auth enabled/disabled, CORS origins)
- **FR-034**: Documentation MUST include production security checklist and development setup guide

#### Standardized Error Responses (P2)

- **FR-035**: All API endpoints MUST return errors in standard envelope format
- **FR-036**: Error envelope MUST include: error code, message, detail, hint, field (optional), timestamp
- **FR-037**: System MUST define error code taxonomy (4xx client errors, 5xx server errors)
- **FR-038**: Error codes MUST be documented in API specification
- **FR-039**: System MUST provide custom FastAPI exception handler for consistent error formatting
- **FR-040**: OpenAPI schema MUST document error response formats

#### Package Namespace (P3)

- **FR-041**: Package MUST be importable as `videoannotator` (not `src`)
- **FR-042**: System MUST provide deprecation shims for old `src.*` imports (one release grace period)
- **FR-043**: Old imports MUST emit deprecation warnings when used
- **FR-044**: Package structure MUST work identically in wheel and editable install modes
- **FR-045**: Documentation MUST include migration guide for import path updates

#### Batch/Job Semantics Fixes (P3)

- **FR-046**: System MUST return success rate of 0.0 (not null) when total_pipelines equals zero
- **FR-047**: Job records MUST include distinct `queued_at` and `started_at` timestamps
- **FR-048**: System MUST expose retry configuration via environment variables (`RETRY_BASE_DELAY`, `RETRY_MAX_DELAY`, `RETRY_JITTER`)
- **FR-049**: Retry backoff MUST be configurable and documented

#### Test Infrastructure (P3)

- **FR-050**: Test suite MUST use synthetic video generation for deterministic tests
- **FR-051**: Unit tests MUST not perform file I/O (use mocked OpenCV capture)
- **FR-052**: System MUST provide reusable test fixture library for different video scenarios
- **FR-053**: Test suite MUST run reliably in CI without flaky failures

#### JOSS Publication Readiness (P1)

- **FR-054**: System MUST provide `scripts/verify_installation.py` that validates installation and runs sample pipeline
- **FR-055**: All API endpoints MUST have comprehensive docstrings with example curl commands
- **FR-056**: README MUST document test execution with `uv run pytest` and display CI status badge
- **FR-057**: API endpoint docstrings MUST include request/response examples with expected inputs/outputs
- **FR-058**: Test coverage for core pipelines MUST be verified and documented (target: >80% for critical paths)

#### Documentation Organization & Cleanup (P2)

- **FR-059**: Documentation MUST be reorganized for external users and OSS contributors clarity
- **FR-060**: Documentation MUST include troubleshooting guide for common installation/usage issues
- **FR-061**: Documentation MUST distinguish clearly between user guides, contributor guides, and API references
- **FR-062**: docs/ structure MUST follow consistent hierarchy: installation/ → usage/ → development/ → testing/ → deployment/
- **FR-063**: Documentation MUST include "Getting Started for Reviewers" guide for JOSS reviewers
- **FR-064**: All generated documentation (e.g., pipelines_spec.md) MUST include "DO NOT EDIT MANUALLY" warning and regeneration instructions

#### Script Consolidation & Diagnostic Tools (P2)

- **FR-065**: Essential testing scripts MUST be migrated to main test suite or removed if redundant
- **FR-066**: Outdated or dangerous scripts MUST be removed or archived with deprecation notices
- **FR-067**: System MUST provide diagnostic CLI commands: `videoannotator diagnose [system|gpu|storage|database]`
- **FR-068**: Diagnostic tools MUST validate installation, dependencies, GPU availability, and storage permissions
- **FR-069**: Scripts retained in scripts/ folder MUST have clear documentation headers explaining purpose and usage
- **FR-070**: Script documentation MUST indicate if script is for developers, users, or CI automation

### Key Entities

- **Job**: Represents a video processing request with status (pending, running, completed, failed, cancelled), timestamps (queued_at, started_at, completed_at), configuration, and results
- **Pipeline**: Represents a processing module (e.g., face detection, audio diarization) with name, category, configuration schema, and availability status
- **Storage Location**: Represents persistent file storage with directory structure (uploads/, results/, temp/, logs/), disk space monitoring, and retention policy
- **Validation Result**: Represents configuration validation outcome with field-level errors, error codes, and suggested corrections
- **Error Envelope**: Represents standardized API error response with code, message, detail, hint, field, and timestamp

## Success Criteria *(mandatory)*

### Measurable Outcomes

#### Data Integrity & Persistence (P1)

- **SC-001**: Zero job failures occur due to pipeline naming mismatches
- **SC-002**: Zero data loss occurs when server restarts (100% of uploaded videos and completed results persist)
- **SC-003**: 100% of uploaded videos survive server restart and remain accessible
- **SC-004**: Storage location is configurable via environment variable and documented

#### Job Management (P1)

- **SC-005**: All running jobs can be cancelled within 5 seconds via API
- **SC-006**: GPU memory is released within 5 seconds of job cancellation (verified via nvidia-smi)
- **SC-007**: Server remains stable when concurrent job limit prevents OOM (no crashes with multiple concurrent jobs)
- **SC-008**: Queue position is visible to users when jobs are waiting

#### Configuration Validation (P1)

- **SC-009**: 100% of invalid configurations are rejected at submission time (before job is queued)
- **SC-010**: Validation error messages include field-level detail and valid value examples
- **SC-011**: Pre-submission validation endpoint responds in under 200ms
- **SC-012**: Validation errors are human-readable and actionable (user can fix config without consulting docs)

#### Security Posture (P2)

- **SC-013**: Authentication is required by default on fresh installations (no configuration needed for secure default)
- **SC-014**: Unauthorized origin requests are blocked by CORS (100% enforcement of configured origins)
- **SC-015**: Security warnings are logged prominently when insecure modes are enabled
- **SC-016**: Production security checklist is documented and accessible

#### API Consistency (P2)

- **SC-017**: 100% of API endpoints return errors in standard envelope format
- **SC-018**: Error codes are consistent across all endpoints and documented
- **SC-019**: Client applications can parse errors reliably with single error handler
- **SC-020**: Error responses include timestamps for support correlation

#### Package Structure (P3)

- **SC-021**: Package is importable as `videoannotator.*` after installation
- **SC-022**: Old `src.*` imports work with deprecation warnings for one release
- **SC-023**: Wheel and editable installs produce identical import behavior
- **SC-024**: Migration guide enables developers to update imports in under 30 minutes

#### System Reliability (Overall)

- **SC-025**: Job submission latency remains under 200ms including validation
- **SC-026**: Test suite runs reliably in CI with zero flaky failures
- **SC-027**: Storage cleanup overhead consumes less than 1% CPU when enabled
- **SC-028**: All health checks pass (pipeline registry, storage, database, GPU availability, queue system)

#### JOSS Publication Quality (P1)

- **SC-029**: Installation verification script completes successfully on Ubuntu, macOS, and Windows
- **SC-030**: All core API endpoints have docstrings with curl examples (100% coverage for public endpoints)
- **SC-031**: Test coverage for core pipelines is verified at >80% for critical execution paths
- **SC-032**: Independent reviewer can install and run sample pipeline in under 15 minutes following README

#### Documentation Quality (P2)

- **SC-033**: Documentation structure receives positive feedback from 2+ external reviewers
- **SC-034**: Troubleshooting guide resolves 80% of common installation issues without support ticket
- **SC-035**: Contributors can navigate from "I want to contribute" to "Pull request submitted" using only CONTRIBUTING.md
- **SC-036**: JOSS reviewers can complete review checklist without external clarification requests

#### Script & Tool Quality (P2)

- **SC-037**: scripts/ folder contains only actively maintained, documented tools (no orphaned/confusing files)
- **SC-038**: Diagnostic CLI identifies common configuration issues with actionable fix suggestions
- **SC-039**: Essential test functionality is captured in main pytest suite (no critical coverage gaps from removed scripts)
- **SC-040**: All retained scripts have usage documentation and clear purpose statements

## Scope & Boundaries *(mandatory)*

### In Scope

- Critical bug fixes preventing production deployment
- Data loss prevention and persistence guarantees
- Job lifecycle management (cancellation, queuing, state transitions)
- Configuration validation with clear error messages
- Secure-by-default configuration
- Standardized error response format
- Package namespace normalization
- Test infrastructure improvements for reliability
- **JOSS publication readiness**: installation verification, API documentation, test coverage validation
- **Documentation spring cleaning**: reorganization for external users, troubleshooting guides, contributor clarity
- **Script consolidation**: migrate essential tests, remove outdated scripts, create diagnostic tools

### Out of Scope (Deferred to v1.4.0+)

- Advanced features: active learning, quality assessment, multi-modal correlation
- Plugin system architecture
- Real-time streaming capabilities
- GraphQL API
- Enhanced health endpoint (detailed GPU stats, performance metrics)
- Version info endpoint
- Structured logging (JSON format)
- Enterprise features (SSO, RBAC, multi-tenancy, audit logs)
- Cloud provider integration
- Microservice decomposition

## Assumptions *(include if applicable)*

1. **Infrastructure**: Server runs on Linux with GPU support optional but recommended for production
2. **Storage**: Persistent storage is on reliable filesystem (not network-mounted unless specifically configured for low latency)
3. **Concurrent Users**: Typical lab environment has 5-20 concurrent researchers, not thousands
4. **Job Duration**: Most jobs complete within 5-60 minutes (not multi-hour processing)
5. **Disk Space**: Storage directory has sufficient space for typical research workflows (100GB+ recommended)
6. **Authentication**: When enabled, assumes external authentication mechanism provides user tokens (JWT or API keys)
7. **CORS**: Production deployments use known frontend origins that can be configured at startup
8. **Migration**: Users upgrading from v1.2.x have 2-4 hour maintenance window for testing
9. **GPU Memory**: GPU jobs need 4-8GB VRAM; concurrent limit prevents exhaustion
10. **Python Version**: Python 3.10+ with uv package manager for dependency resolution

## Dependencies *(include if applicable)*

### Technical Dependencies

- Python 3.10+ runtime environment
- uv package manager for dependency resolution
- FastAPI framework (existing, version unchanged)
- SQLAlchemy for database operations (existing)
- GPU drivers (NVIDIA CUDA) for GPU-accelerated pipelines (optional)

### External Dependencies

- Client integration: Video Annotation Viewer v0.4.0+ (coordinated release)
- Documentation: Migration guide must be ready before release
- Testing: Client team QA sign-off required

### Breaking Changes

- Authentication now required by default (can be disabled via env var)
- Import paths change from `src.*` to `videoannotator.*` (deprecation shims provided)
- CORS restricted by default (must configure allowed origins)
- Error response format changes (standard envelope replaces ad-hoc formats)

## Risks & Mitigations *(include if applicable)*

### Risk 1: Storage Migration Breaks Existing Deployments

**Impact**: HIGH - Users with existing jobs could lose data during upgrade
**Likelihood**: MEDIUM - Migration path not yet tested with production data
**Mitigation**:
- Provide automated migration script that moves `/tmp` files to persistent storage
- Document backup procedure before upgrade
- Test migration with realistic data volumes
- Include rollback procedure in release notes

### Risk 2: Job Cancellation Doesn't Release GPU Memory

**Impact**: HIGH - GPU OOM crashes continue, defeats purpose of feature
**Likelihood**: MEDIUM - Complex signal handling and process lifecycle management
**Mitigation**:
- Test cancellation with all pipeline types
- Add monitoring for GPU memory usage before/after cancellation
- Include fallback to force-kill if graceful shutdown fails
- Document known limitations with specific pipelines

### Risk 3: Config Validation Performance Impact

**Impact**: MEDIUM - Job submission latency could exceed 200ms target
**Likelihood**: LOW - Validation is synchronous schema check
**Mitigation**:
- Benchmark validation with complex configs
- Cache schema validation rules
- Optimize validation logic for common patterns
- Consider async validation for very large configs

### Risk 4: Breaking Changes Block Urgent Upgrades

**Impact**: MEDIUM - Users with existing integrations must update code to upgrade
**Likelihood**: HIGH - Authentication and error format changes are breaking
**Mitigation**:
- Provide deprecation shims for import paths
- Document all breaking changes with migration examples
- Maintain v1.2.x security patches for 90 days
- Offer migration support office hours

### Risk 5: Timeline Slippage Due to Scope Creep

**Impact**: MEDIUM - Delayed release affects client team timeline
**Likelihood**: MEDIUM - Temptation to add "just one more fix"
**Mitigation**:
- Strict scope enforcement (only critical fixes)
- Weekly progress reviews with stakeholders
- Defer non-critical items to v1.4.0 backlog
- Maintain "deferred features" list in roadmap

## Clarifications Resolved

### Question 1: Storage Cleanup Strategy

**Decision**: **Option B - Automatic scheduled cleanup, disabled by default (opt-in via env var)**

**Rationale**: Conservative approach for research data - users maintain full control over when cleanup occurs. Can be enabled via `STORAGE_RETENTION_DAYS=30` when needed. Prevents accidental data loss while still providing cleanup capability for production deployments.

---

### Question 2: Job Cancellation During Multi-Pipeline Processing

**Decision**: **Option A - Immediate termination of all pipelines (in-progress and queued)**

**Rationale**: Simplest implementation with fastest response time (2-5 seconds). Primary goal is GPU memory release, which this achieves immediately. Users who need partial results can monitor job progress and cancel strategically. Reduces implementation complexity and ships faster.

---

### Question 3: Authentication Default Behavior

**Decision**: **Option A - Hard default: Auth required unless explicitly disabled (breaking change acceptable)**

**Rationale**: With minimal existing user base, can establish secure-by-default posture from the start. Sets correct precedent for future users. Development use remains easy with `VIDEOANNOTATOR_REQUIRE_AUTH=false`. Avoids "we'll fix security later" technical debt.

---

## Related Documents *(include if applicable)*

- [v1.3.0 Development Roadmap](../../docs/development/roadmap_v1.3.0.md) - Comprehensive release plan and timeline
- [JOSS Readiness Assessment](./JOSS_READINESS_ASSESSMENT.md) - Detailed analysis of JOSS review requirements and gaps
- [JOSS Paper Draft](../../docs/joss.md) - Software paper for Journal of Open Source Software submission
- [Client Team QA Report](../../docs/testing/) - Integration testing feedback from Video Annotation Viewer team
- [AGENTS.md](../../AGENTS.md) - Agent collaboration guidelines and technical standards
- [CHANGELOG.md](../../CHANGELOG.md) - Version history and release notes
- [CONTRIBUTING.md](../../CONTRIBUTING.md) - Contributor guide (to be enhanced in this release)

## Notes

- This specification covers a full minor release (v1.3.0) rather than a single feature, so scope is intentionally broad but strictly limited to critical fixes
- Timeline is aggressive (6-8 weeks) due to production deployment blocker issues AND coordinated JOSS publication
- Client team (Video Annotation Viewer) is coordinating their v0.5.0 release with our v1.3.0
- JOSS submission planned immediately after v1.3.0 release (targeting concurrent announcement)
- Documentation and script cleanup (FR-059 to FR-070) are "spring cleaning" tasks that remove technical debt and improve external contributor experience
- All advanced features (plugins, ML enhancements, enterprise capabilities) are explicitly deferred to v1.4.0+
- Test infrastructure improvements are included as P3 to prevent ongoing flaky test issues
- JOSS readiness requirements (FR-054 to FR-058) are P1 because they are release blockers for publication
