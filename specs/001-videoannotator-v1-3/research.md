# Phase 0 Research: VideoAnnotator v1.3.0 Technical Decisions

**Date**: October 11, 2025
**Status**: Complete
**Purpose**: Resolve all technical unknowns before Phase 1 design

---

## R1: Job Cancellation Patterns with GPU Memory Release

### Research Question
How to reliably terminate Python processes running GPU workloads and ensure VRAM is freed within 5 seconds?

### Decision: Signal-Based Cancellation with Explicit GPU Cleanup

**Implementation Approach**:
1. Use `multiprocessing.Process` for job execution (not threading - need independent GIL)
2. Set cancellation flag in shared memory (`multiprocessing.Value`)
3. Worker checks flag periodically during pipeline execution
4. On cancellation signal:
   - Stop current pipeline step
   - Call explicit GPU cleanup: `torch.cuda.empty_cache()`, `tf.keras.backend.clear_session()`
   - Terminate process gracefully with timeout
   - If timeout exceeded (5s), force terminate with `Process.terminate()` then `Process.kill()`
5. Parent process monitors GPU memory via `nvidia-smi` or `pynvml` (optional validation)

**Code Pattern**:
```python
import multiprocessing
import signal
import time

def job_worker(job_id, cancel_flag):
    def signal_handler(signum, frame):
        cancel_flag.value = 1

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        for pipeline in pipelines:
            if cancel_flag.value:
                # Cleanup GPU resources
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                break
            pipeline.run()
    finally:
        # Ensure cleanup even on exception
        cleanup_resources()

# In cancellation endpoint
cancel_flag.value = 1
process.join(timeout=5.0)
if process.is_alive():
    process.terminate()
    process.join(timeout=1.0)
    if process.is_alive():
        process.kill()  # Force kill as last resort
```

### Rationale
- **Graceful first, force second**: Allows pipelines to clean up properly but guarantees termination
- **Explicit GPU cleanup**: Python GC may not immediately free CUDA memory; explicit calls ensure release
- **Timeout-based escalation**: 5s graceful → 1s terminate → kill ensures SC-005 and SC-006
- **Signal propagation**: Standard Unix signals work across platforms (WSL, Linux, macOS)

### Alternatives Considered

**Alternative A: Threading with Event**
- **Rejected**: Python GIL prevents true parallelism, doesn't isolate GPU memory per thread
- **Reason**: GPU pipelines need process isolation for memory safety

**Alternative B: Subprocess with Shell Scripts**
- **Rejected**: Harder to propagate cancellation, no direct GPU cleanup control
- **Reason**: Need Python API access for torch.cuda.empty_cache()

**Alternative C: Kubernetes Job Cancellation**
- **Rejected**: Adds infrastructure complexity, violates "minimal friction local usability"
- **Reason**: Must work on researcher laptops, not just K8s clusters

### Implementation Notes
- Add `is_cancellation_requested()` helper checked between pipeline stages
- Log GPU memory before/after cancellation for validation
- Test with all pipeline types (CPU-only, GPU-required, hybrid)
- Document known limitation: Pipelines blocking on I/O may delay cancellation slightly

---

## R2: SQLAlchemy Schema Migration Strategy

### Research Question
How to migrate existing jobs from schema v1.2.x to v1.3.0 (add CANCELLED status, storage paths)?

### Decision: Inline Migration Script (Not Alembic)

**Implementation Approach**:
1. Detect schema version from database metadata table
2. On first v1.3.0 startup, run inline migration:
   - Add `status` column if missing (default: `PENDING`)
   - Add `storage_path` column (default: `/tmp/{job_id}` for existing jobs)
   - Add `cancelled_at` column (nullable timestamp)
   - Update status enum to include `CANCELLED`
3. Migration runs automatically, logs progress
4. No downgrade support (0-1 users, breaking changes acceptable)

**Code Pattern**:
```python
# src/database/migrations.py
from sqlalchemy import inspect, text

def migrate_to_v1_3_0(engine):
    inspector = inspect(engine)
    columns = [col['name'] for col in inspector.get_columns('jobs')]

    with engine.begin() as conn:
        if 'storage_path' not in columns:
            conn.execute(text("ALTER TABLE jobs ADD COLUMN storage_path TEXT"))
            conn.execute(text("UPDATE jobs SET storage_path = '/tmp/' || job_id WHERE storage_path IS NULL"))

        if 'cancelled_at' not in columns:
            conn.execute(text("ALTER TABLE jobs ADD COLUMN cancelled_at TIMESTAMP"))

        # Update existing jobs to have explicit status if needed
        conn.execute(text("UPDATE jobs SET status = 'PENDING' WHERE status IS NULL"))

    print("[OK] Database migrated to v1.3.0 schema")

# In api_server.py startup
from src.database.migrations import migrate_to_v1_3_0
migrate_to_v1_3_0(engine)
```

### Rationale
- **Simplicity**: Only 2 new columns + 1 enum value, inline migration is sufficient
- **Automatic**: Runs on server start, no manual DBA intervention needed
- **Backward compatible data**: Existing jobs get sensible defaults
- **No Alembic overhead**: Alembic is powerful but overkill for simple schema evolution

### Alternatives Considered

**Alternative A: Alembic Migrations**
- **Rejected**: Adds alembic dependency, requires separate migration command
- **Reason**: VideoAnnotator targets researchers, not DBAs; automatic migration better UX

**Alternative B: Manual SQL Scripts**
- **Rejected**: Users must run scripts manually, error-prone
- **Reason**: Violates "minimal friction" principle; auto-migration safer

**Alternative C: Drop and Recreate Database**
- **Rejected**: Loses existing job data
- **Reason**: Users may have completed jobs they want to retain

### Implementation Notes
- Add `schema_version` metadata table to track version
- Log migration steps verbosely for debugging
- Test migration with realistic v1.2.x database snapshot
- Document rollback: restore v1.2.x database backup, downgrade code

---

## R3: Configuration Validation Framework Design

### Research Question
How to validate pipeline configs at submission time without duplicating pipeline initialization logic?

### Decision: Registry-Driven Pydantic Schema Validation

**Implementation Approach**:
1. Extend pipeline registry metadata to include Pydantic config schemas
2. Load schemas from registry at server startup (cached)
3. Validation endpoint uses cached schemas to validate config
4. For pipelines not in registry, fall back to basic type checking
5. Return field-level errors with hints

**Code Pattern**:
```python
# src/validation/config_validator.py
from pydantic import BaseModel, ValidationError
from src.registry import load_pipeline_registry

class ConfigValidator:
    def __init__(self):
        self.schemas = {}  # Cache loaded schemas
        registry = load_pipeline_registry()
        for pipeline in registry:
            if 'config_schema' in pipeline:
                # Dynamically create Pydantic model from schema
                self.schemas[pipeline['name']] = create_model_from_schema(pipeline['config_schema'])

    def validate(self, pipeline_name: str, config: dict) -> ValidationResult:
        if pipeline_name not in self.schemas:
            return ValidationResult(valid=True, warnings=["No schema available for " + pipeline_name])

        try:
            self.schemas[pipeline_name](**config)
            return ValidationResult(valid=True)
        except ValidationError as e:
            errors = [
                FieldError(
                    field=err['loc'][0],
                    message=err['msg'],
                    code='VALIDATION_ERROR',
                    hint=f"Valid values: {err.get('ctx', {}).get('enum_values', 'see docs')}"
                )
                for err in e.errors()
            ]
            return ValidationResult(valid=False, errors=errors)

# In API endpoint
@app.post("/api/v1/pipelines/validate")
async def validate_config(request: ValidateRequest):
    validator = ConfigValidator()  # Or inject singleton
    result = validator.validate(request.pipeline, request.config)
    if not result.valid:
        raise HTTPException(status_code=400, detail=result.errors)
    return result
```

### Rationale
- **DRY**: Single source of truth (registry metadata) for config schemas
- **Fast**: Schema validation is pure Pydantic (microseconds), no pipeline initialization
- **Field-level errors**: Pydantic provides detailed error locations and messages
- **Extensible**: New pipelines just add schema to registry metadata

### Alternatives Considered

**Alternative A: Initialize Pipeline for Validation**
- **Rejected**: Slow (model loading), resource-intensive (GPU allocation)
- **Reason**: Validation should be <200ms (SC-011), initialization can be seconds

**Alternative B: JSON Schema Validation**
- **Rejected**: Pydantic is already project standard, better Python integration
- **Reason**: Consistency with FastAPI patterns

**Alternative C: No Pre-Validation**
- **Rejected**: Jobs fail late (after queueing), poor UX
- **Reason**: Immediate feedback critical for researcher experience (User Story 3)

### Implementation Notes
- Add `config_schema` field to registry metadata YAML files
- Use Pydantic's `create_model()` for dynamic model generation
- Cache validator instance as singleton for performance
- Add schema evolution version in registry for future changes

---

## R4: FastAPI Error Envelope Pattern

### Research Question
What's the idiomatic FastAPI pattern for consistent error responses across endpoints?

### Decision: Custom Exception Classes + Global Exception Handlers

**Implementation Approach**:
1. Define custom exception hierarchy: `VideoAnnotatorException` base class
2. Subclasses for specific errors: `JobNotFoundException`, `InvalidConfigException`, `PipelineNotFoundException`
3. Register global exception handlers in FastAPI app
4. All exceptions include error code, message, optional detail/hint
5. Pydantic model for error envelope ensures OpenAPI documentation

**Code Pattern**:
```python
# src/api/v1/errors.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime

class ErrorDetail(BaseModel):
    code: str
    message: str
    detail: dict | None = None
    hint: str | None = None
    field: str | None = None
    timestamp: datetime

class ErrorEnvelope(BaseModel):
    error: ErrorDetail

class VideoAnnotatorException(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400,
                 detail: dict = None, hint: str = None, field: str = None):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.detail = detail
        self.hint = hint
        self.field = field

class JobNotFoundException(VideoAnnotatorException):
    def __init__(self, job_id: str):
        super().__init__(
            code="JOB_NOT_FOUND",
            message=f"Job {job_id} not found",
            status_code=404,
            hint="Check job ID or use GET /api/v1/jobs to list available jobs"
        )

class InvalidConfigException(VideoAnnotatorException):
    def __init__(self, errors: list):
        super().__init__(
            code="INVALID_CONFIG",
            message="Configuration validation failed",
            status_code=400,
            detail={"errors": errors},
            hint="Fix validation errors and resubmit"
        )

def register_exception_handlers(app: FastAPI):
    @app.exception_handler(VideoAnnotatorException)
    async def videoannotator_exception_handler(request: Request, exc: VideoAnnotatorException):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorEnvelope(error=ErrorDetail(
                code=exc.code,
                message=exc.message,
                detail=exc.detail,
                hint=exc.hint,
                field=exc.field,
                timestamp=datetime.utcnow()
            )).dict()
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        # Catch-all for unexpected errors
        return JSONResponse(
            status_code=500,
            content=ErrorEnvelope(error=ErrorDetail(
                code="INTERNAL_SERVER_ERROR",
                message="An unexpected error occurred",
                hint="Contact support with timestamp",
                timestamp=datetime.utcnow()
            )).dict()
        )

# In api_server.py
from src.api.v1.errors import register_exception_handlers
register_exception_handlers(app)
```

### Rationale
- **Consistency**: All errors follow same envelope format (FR-035-043)
- **FastAPI idiomatic**: Exception handlers are standard FastAPI pattern
- **OpenAPI integration**: Pydantic models generate OpenAPI schema automatically
- **Actionable**: Hint field provides user guidance (SC-019)
- **Debuggable**: Timestamp enables support correlation (SC-020)

### Alternatives Considered

**Alternative A: Middleware-Based Error Transformation**
- **Rejected**: Middleware harder to customize per error type
- **Reason**: Exception handlers more granular, easier to test

**Alternative B: Response Wrapper Functions**
- **Rejected**: Must remember to wrap every response, error-prone
- **Reason**: Exception handlers are automatic, can't be forgotten

**Alternative C: HTTPException with custom detail**
- **Rejected**: FastAPI HTTPException detail is freeform, no schema enforcement
- **Reason**: Need Pydantic validation for error responses

### Implementation Notes
- Migrate existing endpoints incrementally (start with new endpoints)
- Test all error paths to ensure envelope consistency
- Add error code registry documentation
- Consider `error_code_hint()` function for standardized hints

---

## R5: Python Package Namespace Migration

### Research Question
How to transition from `src.*` imports to `videoannotator.*` with deprecation warnings?

### Decision: `__getattr__` Shim with One-Release Deprecation Period

**Implementation Approach**:
1. Keep `src/` directory structure unchanged initially
2. Create `videoannotator/` as symlink or copy during package build
3. Add `__getattr__` hook in root `__init__.py` to intercept `src.*` imports
4. Emit `DeprecationWarning` on old imports
5. v1.3.0: Both work (with warnings), v1.4.0: Remove shim

**Code Pattern**:
```python
# videoannotator/__init__.py (root package)
import warnings
import sys

# New imports (preferred)
from videoannotator.version import __version__

def __getattr__(name):
    """
    Provide backward compatibility for src.* imports.
    Deprecated in v1.3.0, will be removed in v1.4.0.
    """
    if name == "src":
        warnings.warn(
            "Importing from 'src' is deprecated as of v1.3.0 and will be removed in v1.4.0. "
            "Use 'videoannotator' instead. Example: 'from videoannotator.pipelines import ...' "
            "See https://github.com/InfantLab/VideoAnnotator/blob/master/docs/migration-v1.3.0.md",
            DeprecationWarning,
            stacklevel=2
        )
        # Import and return the src module
        import src
        return src
    raise AttributeError(f"module 'videoannotator' has no attribute '{name}'")

# pyproject.toml changes
# [project]
# name = "videoannotator"  # Was implicit, now explicit
#
# [tool.setuptools]
# packages = ["videoannotator", "videoannotator.api", "videoannotator.pipelines", ...]
```

### Rationale
- **Graceful transition**: Existing code works with warnings, users have time to update
- **Clear messaging**: Warning includes migration guide link
- **One release window**: v1.3.0 warns, v1.4.0 removes (6-12 months notice)
- **Both modes work**: Editable install and wheel behave identically

### Alternatives Considered

**Alternative A: Hard Break, Immediate Rename**
- **Rejected**: Breaks all existing user code (even with 0-1 users, breaks examples/docs)
- **Reason**: Deprecation is best practice, enables smoother adoption

**Alternative B: Dual Namespace Forever**
- **Rejected**: Maintenance burden, confusing for new users
- **Reason**: Clean cutover better long-term

**Alternative C: Import Hook with `sys.meta_path`**
- **Rejected**: More complex, harder to debug
- **Reason**: `__getattr__` is simpler and sufficient for top-level redirect

### Implementation Notes
- Add migration guide: `docs/migration-v1.3.0.md`
- Update all internal imports to use `videoannotator.*`
- Update examples, README, documentation
- Test with both `pip install -e .` and `pip install videoannotator`
- Set warnings filter in tests to ensure tests catch deprecations

---

## R6: Multi-Platform Installation Verification

### Research Question
How to test installation success across Ubuntu, macOS, Windows with single verification script?

### Decision: Progressive Check Script with Platform-Specific Logic

**Implementation Approach**:
1. Single Python script: `scripts/verify_installation.py`
2. Progressive checks: dependencies → FFmpeg → GPU (optional) → sample pipeline
3. Platform detection using `platform.system()`
4. Clear pass/fail output with fix suggestions
5. Exit code: 0 = success, 1 = failure (for CI integration)

**Code Pattern**:
```python
# scripts/verify_installation.py
import platform
import sys
import subprocess
import shutil
from pathlib import Path

class Colors:
    """ASCII-safe output (no emoji, Windows compatible)"""
    PASS = "[OK]"
    FAIL = "[ERROR]"
    WARN = "[WARNING]"
    INFO = "[INFO]"

def check_uv():
    """Check uv package manager installed"""
    if shutil.which("uv"):
        version = subprocess.check_output(["uv", "--version"], text=True).strip()
        print(f"{Colors.PASS} uv found: {version}")
        return True
    else:
        print(f"{Colors.FAIL} uv not found")
        print(f"{Colors.INFO} Install: curl -LsSf https://astral.sh/uv/install.sh | sh")
        return False

def check_ffmpeg():
    """Check FFmpeg installed"""
    if shutil.which("ffmpeg"):
        version = subprocess.check_output(["ffmpeg", "-version"], text=True).split('\n')[0]
        print(f"{Colors.PASS} FFmpeg found: {version}")
        return True
    else:
        print(f"{Colors.FAIL} FFmpeg not found")
        system = platform.system()
        if system == "Linux":
            print(f"{Colors.INFO} Install: sudo apt install ffmpeg")
        elif system == "Darwin":
            print(f"{Colors.INFO} Install: brew install ffmpeg")
        elif system == "Windows":
            print(f"{Colors.INFO} Install: winget install ffmpeg or download from ffmpeg.org")
        return False

def check_gpu():
    """Check GPU availability (optional, non-fatal)"""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"{Colors.PASS} GPU detected (nvidia-smi available)")
            return True
        else:
            print(f"{Colors.WARN} GPU not detected (optional, CPU-only mode will work)")
            return False
    except FileNotFoundError:
        print(f"{Colors.WARN} nvidia-smi not found (GPU optional, CPU mode will work)")
        return False

def check_dependencies():
    """Check Python dependencies via uv"""
    try:
        subprocess.run(["uv", "sync", "--check"], check=True, capture_output=True)
        print(f"{Colors.PASS} Dependencies synced")
        return True
    except subprocess.CalledProcessError:
        print(f"{Colors.FAIL} Dependencies not synced")
        print(f"{Colors.INFO} Run: uv sync")
        return False

def run_sample_pipeline():
    """Run minimal pipeline to verify functionality"""
    print(f"{Colors.INFO} Running sample pipeline test...")
    try:
        # Import videoannotator to verify package works
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from videoannotator.version import __version__
        print(f"{Colors.PASS} VideoAnnotator {__version__} importable")
        return True
    except ImportError as e:
        print(f"{Colors.FAIL} Failed to import videoannotator: {e}")
        return False

def main():
    print("VideoAnnotator Installation Verification")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    print("-" * 60)

    checks = [
        ("uv package manager", check_uv),
        ("FFmpeg", check_ffmpeg),
        ("Python dependencies", check_dependencies),
        ("GPU (optional)", check_gpu),
        ("VideoAnnotator import", run_sample_pipeline),
    ]

    results = []
    for name, check_func in checks:
        try:
            results.append(check_func())
        except Exception as e:
            print(f"{Colors.FAIL} {name} check failed: {e}")
            results.append(False)
        print()

    # GPU check is optional, exclude from pass/fail
    critical_checks = results[:-2] + [results[-1]]  # Skip GPU check

    print("-" * 60)
    if all(critical_checks):
        print(f"{Colors.PASS} All critical checks passed!")
        print(f"{Colors.INFO} VideoAnnotator is ready to use.")
        sys.exit(0)
    else:
        print(f"{Colors.FAIL} Some checks failed. Fix the issues above and re-run.")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### Rationale
- **Single script**: Portable across platforms, no bash/powershell split
- **Progressive checks**: Each check independent, clear which step failed
- **Actionable**: Each failure includes fix instructions for that platform
- **CI-friendly**: Exit code enables automated testing
- **ASCII-safe**: No emoji, works in Windows PowerShell (AGENTS.md constraint)

### Alternatives Considered

**Alternative A: Separate Scripts Per Platform**
- **Rejected**: Maintenance burden, users must pick correct script
- **Reason**: Python platform detection makes single script feasible

**Alternative B: Makefile Targets**
- **Rejected**: Make not standard on Windows, adds dependency
- **Reason**: Pure Python more portable

**Alternative C: Docker Health Check**
- **Rejected**: Only works in Docker, doesn't verify local install
- **Reason**: Need to verify laptop installs for researchers

### Implementation Notes
- Test on Ubuntu 20.04, 22.04, 24.04, macOS Intel/ARM, Windows 10/11
- Add to CI pipeline as smoke test
- Document in README: "Verify installation: `uv run python scripts/verify_installation.py`"
- Consider adding `--verbose` flag for debugging

---

## R7: Storage Cleanup Without Data Loss Risk

### Research Question
How to implement optional cleanup that prevents accidental deletion of active/recent jobs?

### Decision: Conservative Opt-In Cleanup with Multiple Safeguards

**Implementation Approach**:
1. Cleanup disabled by default (FR-012)
2. Enable via `STORAGE_RETENTION_DAYS=30` environment variable
3. Run cleanup on schedule (daily cron or background thread)
4. Safety checks before deletion:
   - Job status must be COMPLETED or FAILED (never PENDING/RUNNING/CANCELLED)
   - Job completed_at must be older than retention days
   - Storage path must be within configured storage directory (prevent deletion outside videoannotator)
   - Dry-run mode for testing
5. Log all deletions for audit trail

**Code Pattern**:
```python
# src/storage/cleanup.py
from datetime import datetime, timedelta
from pathlib import Path
import logging
from src.database.models import Job, JobStatus

logger = logging.getLogger(__name__)

class StorageCleanup:
    def __init__(self, retention_days: int, storage_root: Path, dry_run: bool = False):
        self.retention_days = retention_days
        self.storage_root = storage_root.resolve()
        self.dry_run = dry_run

    def is_safe_to_delete(self, job: Job) -> tuple[bool, str]:
        """Multiple safety checks before deletion"""
        # Check 1: Status must be terminal
        if job.status not in [JobStatus.COMPLETED, JobStatus.FAILED]:
            return False, f"Status {job.status} not terminal"

        # Check 2: Must have completion time
        if not job.completed_at:
            return False, "No completion timestamp"

        # Check 3: Must be older than retention period
        cutoff = datetime.utcnow() - timedelta(days=self.retention_days)
        if job.completed_at > cutoff:
            return False, f"Completed {(datetime.utcnow() - job.completed_at).days} days ago (retention: {self.retention_days})"

        # Check 4: Storage path must be within videoannotator storage
        job_path = Path(job.storage_path).resolve()
        if not str(job_path).startswith(str(self.storage_root)):
            return False, f"Storage path {job_path} outside storage root {self.storage_root}"

        return True, "Safe to delete"

    def cleanup(self, session) -> dict:
        """Execute cleanup with safety checks"""
        results = {"checked": 0, "deleted": 0, "skipped": 0, "errors": 0}

        jobs = session.query(Job).filter(
            Job.status.in_([JobStatus.COMPLETED, JobStatus.FAILED])
        ).all()

        for job in jobs:
            results["checked"] += 1
            safe, reason = self.is_safe_to_delete(job)

            if not safe:
                logger.debug(f"Skipping job {job.job_id}: {reason}")
                results["skipped"] += 1
                continue

            try:
                if self.dry_run:
                    logger.info(f"[DRY RUN] Would delete job {job.job_id} ({Path(job.storage_path).stat().st_size // 1024 // 1024} MB)")
                else:
                    job_path = Path(job.storage_path)
                    if job_path.exists():
                        shutil.rmtree(job_path)
                        logger.info(f"Deleted job {job.job_id} storage ({Path(job.storage_path).stat().st_size // 1024 // 1024} MB)")
                    session.delete(job)
                    results["deleted"] += 1
            except Exception as e:
                logger.error(f"Error deleting job {job.job_id}: {e}")
                results["errors"] += 1

        session.commit()
        return results

# In background scheduler or CLI
import os
retention_days = os.getenv("STORAGE_RETENTION_DAYS")
if retention_days:
    cleanup = StorageCleanup(retention_days=int(retention_days), storage_root=STORAGE_PATH)
    results = cleanup.cleanup(session)
    logger.info(f"Cleanup: {results}")
```

### Rationale
- **Opt-in**: Disabled by default prevents accidental data loss (FR-011)
- **Multiple safeguards**: Layered checks reduce risk of premature deletion
- **Audit trail**: Logging enables investigation if something goes wrong
- **Dry-run mode**: Allows testing cleanup without actual deletion
- **Conservative**: Only deletes terminal states, checks age and location

### Alternatives Considered

**Alternative A: Automatic Cleanup Enabled by Default**
- **Rejected**: Too risky for research data, accidental deletion unacceptable
- **Reason**: FR-011 specifies disabled by default, user must opt-in

**Alternative B: Move to Archive Instead of Delete**
- **Rejected**: Still consumes storage, users can implement archive externally if needed
- **Reason**: Cleanup purpose is to free space, moving doesn't achieve that

**Alternative C: Prompt User Before Deletion**
- **Rejected**: Doesn't work for automated/scheduled cleanup
- **Reason**: Cleanup must run unattended in production

### Implementation Notes
- Add CLI command: `videoannotator storage cleanup --dry-run`
- Log cleanup stats to monitoring system (future)
- Document in production deployment guide
- Consider adding `--force` flag that bypasses some checks (use with caution)

---

## R8: Diagnostic CLI Design Patterns

### Research Question
How should `videoannotator diagnose [system|gpu|storage|database]` provide actionable feedback?

### Decision: Typer Commands with Structured Output + Optional JSON

**Implementation Approach**:
1. Use Typer for CLI framework (already used in videoannotator)
2. Each diagnostic category is a subcommand: `system`, `gpu`, `storage`, `database`
3. Plain text output by default (ASCII-safe for Windows)
4. `--json` flag for machine-readable output
5. Exit codes: 0 = all checks pass, 1 = some checks fail, 2 = critical failure
6. Each check returns: status (pass/warn/fail), message, fix suggestion

**Code Pattern**:
```python
# src/diagnostics/system.py
from dataclasses import dataclass
from typing import Literal
import subprocess
import shutil

@dataclass
class DiagnosticCheck:
    name: str
    status: Literal["pass", "warn", "fail"]
    message: str
    fix_hint: str | None = None
    details: dict | None = None

def check_uv() -> DiagnosticCheck:
    if shutil.which("uv"):
        version = subprocess.check_output(["uv", "--version"], text=True).strip()
        return DiagnosticCheck(
            name="uv",
            status="pass",
            message=f"uv {version} installed",
        )
    else:
        return DiagnosticCheck(
            name="uv",
            status="fail",
            message="uv not found in PATH",
            fix_hint="Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
        )

def check_python_version() -> DiagnosticCheck:
    import sys
    version = sys.version_info
    if version >= (3, 10):
        return DiagnosticCheck(
            name="python",
            status="pass",
            message=f"Python {version.major}.{version.minor}.{version.micro}",
        )
    else:
        return DiagnosticCheck(
            name="python",
            status="fail",
            message=f"Python {version.major}.{version.minor} too old",
            fix_hint="Requires Python 3.10+. Install from python.org"
        )

def diagnose_system() -> list[DiagnosticCheck]:
    """Run all system diagnostics"""
    return [
        check_uv(),
        check_python_version(),
        check_ffmpeg(),
        check_dependencies_synced(),
    ]

# src/cli.py
import typer
import json
from src.diagnostics.system import diagnose_system
from src.diagnostics.gpu import diagnose_gpu
from src.diagnostics.storage import diagnose_storage
from src.diagnostics.database import diagnose_database

diagnose_app = typer.Typer()

@diagnose_app.command("system")
def diagnose_system_cmd(json_output: bool = typer.Option(False, "--json")):
    """Diagnose system dependencies and configuration"""
    checks = diagnose_system()

    if json_output:
        output = [{"name": c.name, "status": c.status, "message": c.message,
                   "fix_hint": c.fix_hint, "details": c.details} for c in checks]
        print(json.dumps({"category": "system", "checks": output}, indent=2))
    else:
        print("System Diagnostics")
        print("-" * 60)
        for check in checks:
            icon = {"pass": "[OK]", "warn": "[WARNING]", "fail": "[ERROR]"}[check.status]
            print(f"{icon} {check.name}: {check.message}")
            if check.fix_hint:
                print(f"  Fix: {check.fix_hint}")

    # Exit code based on worst status
    if any(c.status == "fail" for c in checks):
        raise typer.Exit(code=1)
    elif any(c.status == "warn" for c in checks):
        raise typer.Exit(code=0)  # Warnings don't fail
    else:
        raise typer.Exit(code=0)

@diagnose_app.command("gpu")
def diagnose_gpu_cmd(json_output: bool = typer.Option(False, "--json")):
    """Diagnose GPU availability and configuration"""
    checks = diagnose_gpu()
    # Similar output formatting as system
    ...

# In main CLI app
app.add_typer(diagnose_app, name="diagnose")
```

### Rationale
- **Structured checks**: DiagnosticCheck dataclass makes checks testable and composable
- **Actionable feedback**: Every failure includes fix hint
- **Machine-readable**: --json flag enables scripting and CI integration
- **ASCII-safe**: Plain text output works in Windows PowerShell
- **Exit codes**: Enables shell scripting: `if videoannotator diagnose system; then ...`

### Alternatives Considered

**Alternative A: Rich Library with Fancy Formatting**
- **Rejected**: Rich uses unicode box characters, violates ASCII-safe constraint
- **Reason**: AGENTS.md specifies no fancy unicode for Windows compatibility

**Alternative B: YAML Output Instead of JSON**
- **Rejected**: JSON more widely supported for machine parsing
- **Reason**: JSON standard for APIs and automation tools

**Alternative C: Single `diagnose` Command, No Subcommands**
- **Rejected**: Running all diagnostics may be slow (GPU checks, storage scans)
- **Reason**: Targeted diagnostics faster for debugging specific issues

### Implementation Notes
- Add diagnostics for each subsystem: system, gpu, storage, database
- GPU diagnostics should be optional (not all systems have GPU)
- Storage diagnostics check: path writable, disk space, permissions
- Database diagnostics check: connection, schema version, job counts
- Consider adding `--verbose` flag for detailed output

---

## Research Summary

All 8 technical decisions resolved with clear implementation approaches:

| Research Task | Decision | Complexity | Risk |
|---------------|----------|------------|------|
| R1: Job Cancellation | Signal-based + explicit GPU cleanup | Medium | Medium (GPU state management) |
| R2: Schema Migration | Inline migration on startup | Low | Low (simple schema) |
| R3: Config Validation | Registry-driven Pydantic schemas | Medium | Low (Pydantic mature) |
| R4: Error Envelope | Custom exceptions + global handlers | Low | Low (FastAPI standard) |
| R5: Namespace Migration | __getattr__ shim with deprecation | Low | Low (standard pattern) |
| R6: Installation Verification | Progressive checks script | Low | Low (straightforward) |
| R7: Storage Cleanup | Conservative opt-in with safeguards | Medium | Low (disabled by default) |
| R8: Diagnostic CLI | Typer commands + structured output | Low | Low (Typer is project standard) |

**Total Risk Assessment**: LOW-MEDIUM
**Estimated Implementation Time**: 4-6 weeks for all components
**Dependencies**: None critical; can implement in parallel

**GATE PASSED**: All technical unknowns resolved. Ready for Phase 1 design.

---

**Next Steps**: Proceed to Phase 1 to generate:
- `data-model.md` - Database schema design
- `contracts/` - API contracts (OpenAPI specs)
- `quickstart.md` - Implementation guide
- Update agent context with research findings
