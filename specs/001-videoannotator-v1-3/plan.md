# Implementation Plan: VideoAnnotator v1.3.0 Production Reliability & Critical Fixes

**Branch**: `001-videoannotator-v1-3` | **Date**: 2025-10-11 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-videoannotator-v1-3/spec.md`

**Note**: This is a FULL MINOR RELEASE plan covering 70 functional requirements across 12 domains.

## Summary

VideoAnnotator v1.3.0 addresses critical production blockers preventing deployment: data persistence (no more /tmp data loss), job cancellation with GPU memory release, configuration validation at submission time, secure-by-default authentication, standardized error responses, and package namespace normalization. Additionally, this release prepares the project for JOSS (Journal of Open Source Software) publication by adding installation verification, comprehensive API documentation, and documentation reorganization for external contributors. The release establishes VideoAnnotator as a production-ready, publication-quality research software platform.

## Technical Context

**Language/Version**: Python 3.10+ (constraint from AGENTS.md, current dev: 3.12)
**Primary Dependencies**: FastAPI (web framework), SQLAlchemy (ORM), Pydantic (validation), uv (package manager), Typer (CLI), pytest (testing)
**Storage**: SQLite database (job metadata, status) + file storage (videos, results, logs); currently /tmp → must migrate to persistent storage
**Testing**: pytest with coverage reports, fixtures in tests/, integration tests required for cross-component behavior
**Target Platform**: Linux (Ubuntu 20.04+), macOS (Intel/Apple Silicon), Windows 10/11 (WSL2 + native); Docker containers for reproducibility; GPU optional (NVIDIA CUDA)
**Project Type**: Web API + CLI (existing structure: src/ with api/, pipelines/, database/, cli.py)
**Performance Goals**: Job submission <200ms (SC-025), job cancellation <5s (SC-005), GPU memory release <5s (SC-006), validation <200ms (SC-011)
**Constraints**: ASCII-safe console output (Windows compatibility - no emoji), minimal network calls during import, lazy model initialization, 0-1 existing users (breaking changes acceptable)
**Scale/Scope**: 5-20 concurrent researchers (lab environment), jobs 5-60 minutes, test suite 477 tests, ~4200 Python files, 6-8 week timeline, coordinated JOSS publication

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**VideoAnnotator Project Principles** (extracted from AGENTS.md):

### ✅ I. uv-First Dependency Management
All Python operations use `uv` for resolution, execution, and testing. Commands: `uv sync`, `uv run pytest`, `uv add <package>`. No manual pip/conda activation.

**Status**: ✅ COMPLIANT - v1.3.0 maintains uv-first approach, diagnostic tools will verify uv installation

### ✅ II. Minimal Friction Local Usability
No mandatory external services. Runs locally without cloud dependencies. Simple install: `uv sync` → `uv run python api_server.py`.

**Status**: ✅ COMPLIANT - Persistent storage remains local, authentication disabled by default for dev (`VIDEOANNOTATOR_REQUIRE_AUTH=false`)

### ✅ III. Standard Formats First (Non-Negotiable)
COCO, WebVTT, RTTM outputs. No bespoke schema proliferation. Registry-driven metadata.

**Status**: ✅ COMPLIANT - v1.3.0 does not introduce new output formats, error envelope follows REST standards

### ✅ IV. ASCII-Safe Console Output (Windows Compatibility)
No emoji, no fancy unicode in logs/console. Use `[ERROR]`, `[OK]`, `[START]` prefixes.

**Status**: ✅ COMPLIANT - Diagnostic tools and error messages will use plain ASCII, no emoji introduced

### ✅ V. Test-Driven Quality
Automated tests for core functionality. Target >80% coverage for critical paths. CI-validated.

**Status**: ⚠️ PARTIAL - v1.3.0 improves: installation verification script (FR-054), test coverage validation (FR-058), but adds significant untested code. **MITIGATION**: Phase 1 design includes test strategy for each domain.

### ✅ VI. Future Extensibility via Registry
Pipeline metadata in YAML registry, not hard-coded. Config validation via schema. Avoid premature plugin architecture.

**Status**: ✅ COMPLIANT - Config validation (FR-023-028) uses schema-driven approach, no architecture changes

### ⚠️ VII. Scope Discipline (GATE WARNING)
Defer advanced features (plugins, ML enhancements, GraphQL, streaming) to later releases. v1.3.0 scope: critical fixes only.

**Status**: ⚠️ SCOPE RISK - 70 FRs across 12 domains is ambitious for 6-8 weeks. JOSS+docs+scripts add significant scope beyond "critical fixes."
**MITIGATION**: Strict prioritization (P1 must complete, P2 can partial-defer, P3 can full-defer), weekly scope reviews

## Constitution Violations Requiring Justification

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| 70 FRs in one release | JOSS publication timeline (concurrent with v1.3.0) requires docs+installation verification. Production blockers (persistence, cancellation) cannot defer. | Splitting into v1.3.0 (core) + v1.3.1 (JOSS) delays publication by 2-3 months, missing research community announcement window. |
| Breaking changes (auth default, error format, imports) | 0-1 existing users means migration cost is minimal. Establishes correct defaults early. | Incremental migration complexity outweighs benefit with near-zero user base. Better to break once than maintain legacy paths. |

## Project Structure

### Documentation (this feature)

```
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (existing repository structure - modifications highlighted)

```
/workspaces/VideoAnnotator/
├── src/                              # Main package (becomes videoannotator.*)
│   ├── __init__.py                   # [MODIFY] Add namespace imports, deprecation warnings
│   ├── version.py                    # [EXISTING] Single source of truth for version
│   ├── cli.py                        # [MODIFY] Add 'diagnose' command (FR-067-068)
│   ├── main.py                       # [EXISTING] Entry point
│   ├── config.py                     # [MODIFY] Add validation logic (FR-023-028)
│   ├── api/
│   │   ├── v1/
│   │   │   ├── jobs.py               # [MODIFY] Add cancellation endpoint, enhance docs (FR-015, FR-055)
│   │   │   ├── pipelines.py          # [MODIFY] Add validation endpoint, enhance docs (FR-026, FR-055)
│   │   │   ├── system.py             # [MODIFY] Enrich health endpoint (FR-010)
│   │   │   └── errors.py             # [NEW] Standard error envelope (FR-035-043)
│   │   └── middleware/
│   │       ├── auth.py               # [MODIFY] Default-on authentication (FR-029-034)
│   │       └── cors.py               # [MODIFY] Restrict CORS by default (FR-031)
│   ├── database/
│   │   ├── models.py                 # [MODIFY] Add CANCELLED status, storage paths (FR-001, FR-014)
│   │   ├── migrations.py             # [NEW] Schema migration for v1.3.0
│   │   └── session.py                # [EXISTING] Database connection
│   ├── storage/
│   │   ├── backends.py               # [MODIFY] Persistent storage paths (FR-002-004)
│   │   ├── cleanup.py                # [NEW] Optional storage cleanup (FR-011-013)
│   │   └── paths.py                  # [NEW] Configurable storage locations
│   ├── worker/
│   │   ├── executor.py               # [MODIFY] Cancellation signal handling (FR-016-022)
│   │   └── queue.py                  # [MODIFY] Concurrent job limiting (FR-018-019)
│   ├── pipelines/                    # [EXISTING] No structural changes
│   ├── validation/                   # [NEW] Config validation module (FR-023-028)
│   │   ├── __init__.py
│   │   ├── config_validator.py       # Config schema validation
│   │   └── pipeline_validator.py    # Pipeline availability checks
│   └── diagnostics/                  # [NEW] Diagnostic tools (FR-067-068)
│       ├── __init__.py
│       ├── system.py                 # System diagnostic checks
│       ├── gpu.py                    # GPU availability checks
│       ├── storage.py                # Storage permission checks
│       └── database.py               # Database health checks
├── scripts/
│   ├── verify_installation.py        # [NEW] Installation verification (FR-054)
│   ├── [AUDIT NEEDED]                # [REVIEW] All existing scripts per FR-065-066
│   └── archive/                      # [NEW] Deprecated scripts moved here
├── tests/
│   ├── unit/                         # [EXPAND] Coverage for new modules
│   ├── integration/                  # [EXPAND] Cross-component tests
│   ├── pipelines/                    # [EXISTING] Pipeline-specific tests
│   └── fixtures/                     # [EXISTING] Shared test data
├── docs/
│   ├── installation/                 # [NEW] Reorganized structure (FR-059-062)
│   │   ├── README.md                 # Quick start
│   │   ├── requirements.md           # System requirements
│   │   ├── troubleshooting.md        # [NEW] Common issues (FR-060)
│   │   └── verification.md           # Using verify_installation.py
│   ├── usage/                        # [NEW] User-facing guides
│   │   ├── getting-started.md
│   │   ├── api-guide.md
│   │   └── cli-guide.md
│   ├── development/                  # [EXISTING] Contributor guides
│   │   ├── roadmap_v1.3.0.md         # [EXISTING]
│   │   ├── architecture.md
│   │   └── testing.md
│   ├── testing/                      # [NEW] Test documentation
│   │   └── coverage-report.md        # [NEW] Test coverage summary (FR-058)
│   ├── deployment/                   # [EXISTING] Production guides
│   └── joss.md                       # [MODIFY] Finalize for submission
├── specs/001-videoannotator-v1-3/    # [THIS PLANNING WORK]
│   ├── spec.md                       # Feature specification
│   ├── plan.md                       # This file
│   ├── research.md                   # [Phase 0] Research findings
│   ├── data-model.md                 # [Phase 1] Database schema
│   ├── quickstart.md                 # [Phase 1] Implementation guide
│   ├── contracts/                    # [Phase 1] API contracts
│   └── checklists/                   # Quality validation
└── [existing files unchanged]
```

**Structure Decision**: This is an existing Python web API + CLI project. No new top-level directories. Modifications focus on:
1. **src/**: Add validation/, diagnostics/ modules; modify api/, database/, storage/, worker/
2. **scripts/**: Add verify_installation.py, audit/clean existing scripts
3. **docs/**: Reorganize into installation/ → usage/ → development/ → testing/ → deployment/
4. **tests/**: Expand coverage for new functionality

## Phase 0: Research & Technical Decisions

*OUTPUT: `research.md` with decisions, rationale, and alternatives considered*

### Research Tasks

#### R1: Job Cancellation Patterns with GPU Memory Release
**Question**: How to reliably terminate Python processes running GPU workloads and ensure VRAM is freed within 5 seconds?
**Focus**: Signal handling (SIGTERM/SIGINT), process termination, CUDA context cleanup, zombie process prevention
**Success**: Decision on signal propagation + verification approach

#### R2: SQLAlchemy Schema Migration Strategy
**Question**: How to migrate existing jobs from schema v1.2.x to v1.3.0 (add CANCELLED status, storage paths)?
**Focus**: Alembic vs manual migration, backward compatibility, data preservation during upgrade
**Success**: Migration script outline + rollback plan

#### R3: Configuration Validation Framework Design
**Question**: How to validate pipeline configs at submission time without duplicating pipeline initialization logic?
**Focus**: Pydantic schema extraction from pipelines, validation caching, error message clarity
**Success**: Validation architecture that avoids runtime overhead

#### R4: FastAPI Error Envelope Pattern
**Question**: What's the idiomatic FastAPI pattern for consistent error responses across endpoints?
**Focus**: Exception handlers, middleware, Pydantic error models, OpenAPI schema integration
**Success**: Error envelope design + implementation strategy

#### R5: Python Package Namespace Migration
**Question**: How to transition from `src.*` imports to `videoannotator.*` with deprecation warnings?
**Focus**: `__init__.py` shims, `warnings.warn()` usage, impact on editable installs, wheel packaging
**Success**: Import strategy that works in both modes for one release

#### R6: Multi-Platform Installation Verification
**Question**: How to test installation success across Ubuntu, macOS, Windows with single verification script?
**Focus**: Platform detection, FFmpeg verification, GPU detection (optional), sample pipeline execution
**Success**: Cross-platform diagnostic approach

#### R7: Storage Cleanup Without Data Loss Risk
**Question**: How to implement optional cleanup that prevents accidental deletion of active/recent jobs?
**Focus**: Retention policy logic, job status checking, safeguards against premature deletion
**Success**: Cleanup algorithm with multiple safety checks

#### R8: Diagnostic CLI Design Patterns
**Question**: How should `videoannotator diagnose [system|gpu|storage|database]` provide actionable feedback?
**Focus**: Check sequencing, failure categorization, fix suggestions, machine-readable output (--json)
**Success**: Diagnostic command structure + output format

### Research Consolidation

Each research task produces:
- **Decision**: Chosen approach
- **Rationale**: Why this approach solves the problem
- **Alternatives Considered**: What else was evaluated and why rejected
- **Implementation Notes**: Key details for Phase 1 design

**GATE**: All "NEEDS CLARIFICATION" from Technical Context must be resolved in research.md before Phase 1.
## Phase 1: Design & Contracts

*PREREQUISITES: research.md complete with all technical decisions*
*OUTPUT: data-model.md, /contracts/*, quickstart.md, updated agent context*

### D1: Database Schema Design (data-model.md)

**Entities to Model**:

1. **Job** (modifications to existing model)
   - Add `status` enum: include `CANCELLED` state
   - Add `storage_path` field (string, directory where video/results stored)
   - Add `cancelled_at` timestamp (nullable)
   - Add `queue_position` computed property
   - Migration: existing jobs get default storage_path = /tmp/{job_id}

2. **ValidationResult** (new model, transient - not persisted)
   - `is_valid`: boolean
   - `errors`: list of FieldError
   - `warnings`: list of FieldWarning
   - FieldError: {field: string, message: string, code: string, hint: string}

3. **ErrorEnvelope** (Pydantic model for API responses)
   - `error`: {code: string, message: string, detail: optional, hint: optional, field: optional, timestamp: datetime}
   - HTTP status code mapping

**Relationships**:
- Job → Storage Path (one-to-one)
- Job → Pipeline Configs (one-to-many, existing)

**State Machine** (Job status transitions):
```
PENDING → RUNNING → COMPLETED
        ↓         ↓
     CANCELLED  FAILED
```

### D2: API Contracts (contracts/ directory)

Generate OpenAPI specs for new/modified endpoints:

**1. Job Cancellation** (`contracts/job-cancellation.yaml`)
```yaml
POST /api/v1/jobs/{job_id}/cancel
  Response 200: {status: "CANCELLED", cancelled_at: timestamp}
  Response 404: {error: {code: "JOB_NOT_FOUND", message: "...", hint: "..."}}
  Response 409: {error: {code: "JOB_ALREADY_COMPLETED", ...}}
```

**2. Config Validation** (`contracts/config-validation.yaml`)
```yaml
POST /api/v1/pipelines/validate
  Request: {config: object, pipelines: [string]}
  Response 200: {valid: true, warnings: [...]}
  Response 400: {valid: false, errors: [{field, message, code, hint}]}
```

**3. Health Endpoint Enhancement** (`contracts/health.yaml`)
```yaml
GET /api/v1/health?detailed=true
  Response 200: {
    status: "healthy",
    storage: {path, available_gb, writable},
    database: {connected, job_count},
    pipelines: {loaded_count, available: [...]},
    gpu: {available, devices: [...]} (optional)
  }
```

**4. Error Envelope Standard** (`contracts/error-envelope.yaml`)
```yaml
Error Response Schema (all 4xx/5xx):
  {
    error: {
      code: string (e.g., "PIPELINE_NOT_FOUND"),
      message: string (human-readable),
      detail: object (optional, field-level info),
      hint: string (optional, suggested action),
      field: string (optional, for validation errors),
      timestamp: datetime (ISO 8601)
    }
  }
```

### D3: CLI Command Design

**Diagnostic Commands** (contracts/cli-diagnose.yaml):
```bash
videoannotator diagnose system
  # Checks: uv installed, Python version, dependencies synced
  # Output: ✓ uv 0.5.0 found | ✗ FFmpeg not in PATH (install: apt install ffmpeg)

videoannotator diagnose gpu
  # Checks: NVIDIA drivers, CUDA version, available VRAM
  # Output: ✓ GPU 0: Tesla T4, 15 GB free | ⚠ CUDA 11.8 (12.0+ recommended)

videoannotator diagnose storage
  # Checks: Storage path writable, disk space, permissions
  # Output: ✓ /var/videoannotator/storage writable, 150 GB free

videoannotator diagnose database
  # Checks: Database accessible, schema version, job count
  # Output: ✓ Database OK, schema v1.3.0, 42 jobs

# --json flag for machine parsing
videoannotator diagnose system --json
  # Output: {"checks": [{"name": "uv", "status": "pass", "version": "0.5.0"}, ...]}
```

### D4: Import Shim Design

**videoannotator/__init__.py** (contracts/import-shim.md):
```python
import warnings

# New imports (preferred)
from videoannotator.pipelines import ...
from videoannotator.api import ...

# Deprecation shims (for backward compatibility)
def __getattr__(name):
    if name == "src":
        warnings.warn(
            "Importing from 'src' is deprecated. Use 'videoannotator' instead. "
            "See docs/migration-v1.3.0.md for details.",
            DeprecationWarning,
            stacklevel=2
        )
        import src
        return src
    raise AttributeError(f"module 'videoannotator' has no attribute '{name}'")
```

### D5: Quickstart Implementation Guide (quickstart.md)

Provide developers with implementation sequence:

1. **Phase 1A: Foundation (P1)**
   - Implement error envelope (src/api/v1/errors.py)
   - Database schema migration (src/database/migrations.py)
   - Persistent storage paths (src/storage/paths.py)

2. **Phase 1B: Core Features (P1)**
   - Job cancellation endpoint + signal handling (src/api/v1/jobs.py, src/worker/executor.py)
   - Config validation module (src/validation/)
   - Concurrent job limiting (src/worker/queue.py)

3. **Phase 1C: Security (P2)**
   - Auth default-on (src/api/middleware/auth.py)
   - CORS restriction (src/api/middleware/cors.py)
   - Security documentation

4. **Phase 1D: JOSS Readiness (P1)**
   - Installation verification script (scripts/verify_installation.py)
   - API docstring enhancement (all src/api/v1/*.py)
   - Test coverage validation

5. **Phase 1E: Cleanup (P2)**
   - Documentation reorganization (docs/)
   - Script audit and consolidation (scripts/)
   - Diagnostic CLI (src/diagnostics/, src/cli.py)

6. **Phase 1F: Polish (P3)**
   - Package namespace migration (src/__init__.py)
   - Test infrastructure improvements (tests/)
   - Final testing and integration

### D6: Agent Context Update

Run `.specify/scripts/bash/update-agent-context.sh copilot` to update GitHub Copilot context with:
- New technologies: None (using existing stack)
- New patterns: Error envelope, deprecation warnings, diagnostic CLI
- Architecture changes: validation/, diagnostics/ modules added

---

## Phase 2: Task Breakdown (NOT CREATED BY THIS COMMAND)

*NOTE: Phase 2 is handled by `/speckit.tasks` command, not `/speckit.plan`.*

The tasks.md file will break down each design component into:
- Specific implementation tasks
- Test coverage requirements
- Dependencies and sequencing
- Acceptance criteria per task

---

## Implementation Sequence Summary

### Week 1-2: Critical Foundation (P1)
- [ ] Research.md completion (Phase 0)
- [ ] Data model design (Phase 1)
- [ ] Error envelope implementation
- [ ] Database migration
- [ ] Persistent storage implementation
- [ ] Job cancellation endpoint + GPU cleanup

### Week 3-4: Core Features + JOSS (P1)
- [ ] Config validation module
- [ ] Concurrent job limiting
- [ ] Installation verification script
- [ ] API docstring enhancement (all endpoints)
- [ ] Test coverage measurement + documentation

### Week 5-6: Security + Documentation (P2)
- [ ] Auth default-on + security docs
- [ ] CORS restriction
- [ ] Documentation reorganization (docs/ structure)
- [ ] Troubleshooting guide
- [ ] JOSS reviewer guide

### Week 6-7: Cleanup + Diagnostics (P2)
- [ ] Script audit + consolidation
- [ ] Diagnostic CLI implementation
- [ ] Storage cleanup (optional feature)

### Week 7-8: Polish + Release (P3 if time permits)
- [ ] Package namespace migration
- [ ] Test infrastructure improvements
- [ ] Final integration testing
- [ ] External reviewer validation
- [ ] Release preparation

---

## Risk Mitigation Strategy

### R1: Scope Overload
- **Monitor**: Weekly scope review, track P1 vs P2 vs P3 completion
- **Trigger**: If P1 slips beyond Week 4, defer P2 items
- **Action**: Move docs reorganization, diagnostic CLI to v1.3.1

### R2: GPU Cancellation Complexity
- **Monitor**: Early prototype in Week 1, test with all pipeline types
- **Trigger**: If cancellation doesn't free GPU in <5s on any pipeline
- **Action**: Document known limitations, add force-kill fallback

### R3: JOSS Timeline Pressure
- **Monitor**: External reviewer availability by Week 3
- **Trigger**: If reviewers unavailable or feedback cycle >2 weeks
- **Action**: Adjust submission timeline or streamline docs

### R4: Breaking Change Blockers
- **Monitor**: User feedback channels, GitHub issues
- **Trigger**: Unexpected breaking change impacts (should be minimal with 0-1 users)
- **Action**: Provide migration scripts, extend v1.2.x support

---

## Success Criteria Validation

### Must-Have for Release (GATE)
- ✅ All P1 functional requirements complete (FR-001 to FR-058 excluding FR-044-049)
- ✅ Installation verification script passes on Ubuntu, macOS, Windows
- ✅ Test coverage >80% for core pipelines (measured + documented)
- ✅ External reviewer completes JOSS checklist without clarifications
- ✅ All API endpoints have comprehensive docstrings
- ✅ Zero data loss on server restart (persistence validated)
- ✅ Job cancellation <5s with GPU memory release (validated)
- ✅ Breaking changes documented with migration guide

### Nice-to-Have (Can Defer)
- ⚠️ Package namespace migration (P3 - can defer to v1.3.1)
- ⚠️ Complete script consolidation (P2 - can be partial)
- ⚠️ Test infrastructure improvements (P3 - can defer)

---

## Coordination Requirements

### Internal
- [ ] Batch A/B/C cleanup PR merged to master (baseline for v1.3.0 work)
- [ ] CI pipeline stable (test badge meaningful)
- [ ] Development team capacity: 1-2 developers full-time equivalent

### External
- [ ] External reviewers: 2+ for documentation validation (schedule Week 3-4)
- [ ] JOSS mock reviewer: 1 for checklist validation (schedule Week 5)
- [ ] Video Annotation Viewer team: coordinated release timeline (v0.5.0)

---

## Deliverables Checklist

### Planning Phase (This Document)
- [x] Technical context complete
- [x] Constitution check performed
- [x] Project structure documented
- [x] Research tasks defined (Phase 0)
- [x] Design tasks defined (Phase 1)
- [x] Implementation sequence outlined
- [x] Risk mitigation planned

### Next Steps
1. **Generate research.md** - Agent executes Phase 0 research tasks
2. **Generate data-model.md** - Agent designs database schema
3. **Generate contracts/** - Agent creates API contracts
4. **Generate quickstart.md** - Agent provides implementation guide
5. **Update agent context** - Run update script
6. **Proceed to /speckit.tasks** - Break down into granular tasks

---

**Plan Status**: ✅ READY FOR PHASE 0 RESEARCH
**Branch**: `001-videoannotator-v1-3`
**Estimated Timeline**: 6-8 weeks (aggressive, requires strict scope control)
**Risk Level**: MEDIUM (scope ambitious, but well-defined with clear prioritization)
