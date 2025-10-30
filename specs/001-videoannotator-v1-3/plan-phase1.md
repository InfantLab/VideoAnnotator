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
