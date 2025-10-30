# Data Model Design: VideoAnnotator v1.3.0

**Date**: October 11, 2025
**Status**: Design Phase
**Source**: Research decisions from research.md

---

## Overview

This document defines the data models and schema changes for v1.3.0, focusing on:
- Database schema modifications for job persistence and cancellation
- Validation result structures for config validation
- Error envelope format for standardized API responses

---

## Database Schema Changes

### Entity 1: Job (Modified)

**Purpose**: Represents a video processing request with lifecycle tracking

**Schema Changes for v1.3.0**:

```sql
-- Existing columns (v1.2.x)
CREATE TABLE jobs (
    job_id TEXT PRIMARY KEY,
    video_path TEXT NOT NULL,
    config TEXT,  -- JSON blob
    status TEXT NOT NULL,  -- PENDING, RUNNING, COMPLETED, FAILED
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT
);

-- NEW columns for v1.3.0
ALTER TABLE jobs ADD COLUMN storage_path TEXT NOT NULL DEFAULT '/tmp/';
ALTER TABLE jobs ADD COLUMN cancelled_at TIMESTAMP NULL;

-- Status enum now includes: PENDING, RUNNING, COMPLETED, FAILED, CANCELLED
```

**Python Model** (SQLAlchemy):

```python
# src/database/models.py
from sqlalchemy import Column, String, DateTime, Text, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from enum import Enum

Base = declarative_base()

class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"  # NEW in v1.3.0

class Job(Base):
    __tablename__ = "jobs"

    # Existing fields
    job_id = Column(String, primary_key=True)
    video_path = Column(Text, nullable=False)
    config = Column(Text)  # JSON string
    status = Column(SQLEnum(JobStatus), nullable=False, default=JobStatus.PENDING)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)

    # NEW fields for v1.3.0
    storage_path = Column(Text, nullable=False)  # Persistent storage location
    cancelled_at = Column(DateTime, nullable=True)  # When cancellation occurred

    @property
    def queue_position(self) -> int | None:
        """
        Computed property: position in queue for PENDING jobs.
        Returns None if not pending.
        """
        if self.status != JobStatus.PENDING:
            return None
        # Implementation: count PENDING jobs created before this one
        # (Implemented in repository/service layer, not ORM)
        return None  # Placeholder

    @property
    def is_terminal(self) -> bool:
        """Check if job is in a terminal state"""
        return self.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]

    def to_dict(self) -> dict:
        """Serialize to dict for API responses"""
        return {
            "job_id": self.job_id,
            "video_path": self.video_path,
            "config": self.config,
            "status": self.status.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "cancelled_at": self.cancelled_at.isoformat() if self.cancelled_at else None,
            "storage_path": self.storage_path,
            "error_message": self.error_message,
            "queue_position": self.queue_position,
        }
```

**State Transitions**:

```
Initial State: PENDING

PENDING → RUNNING      (job picked up by worker)
PENDING → CANCELLED    (cancelled before starting)
RUNNING → COMPLETED    (successful completion)
RUNNING → FAILED       (error during processing)
RUNNING → CANCELLED    (cancelled during processing)

Terminal States: COMPLETED, FAILED, CANCELLED
```

**Migration Script**:

```python
# src/database/migrations.py
from sqlalchemy import inspect, text, MetaData, Table, Column, Text, DateTime
from sqlalchemy.engine import Engine
import logging

logger = logging.getLogger(__name__)

def get_schema_version(engine: Engine) -> str:
    """Get current schema version from metadata table"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version FROM schema_metadata LIMIT 1"))
            row = result.fetchone()
            return row[0] if row else "1.2.0"
    except:
        # Table doesn't exist, assume v1.2.0
        return "1.2.0"

def migrate_to_v1_3_0(engine: Engine):
    """Migrate database schema from v1.2.x to v1.3.0"""
    current_version = get_schema_version(engine)

    if current_version >= "1.3.0":
        logger.info(f"[OK] Database already at v{current_version}")
        return

    logger.info(f"[START] Migrating database from v{current_version} to v1.3.0")

    inspector = inspect(engine)

    # Check if jobs table exists
    if 'jobs' not in inspector.get_table_names():
        logger.warning("[WARNING] Jobs table doesn't exist, creating fresh schema")
        Base.metadata.create_all(engine)
        return

    columns = {col['name']: col for col in inspector.get_columns('jobs')}

    with engine.begin() as conn:
        # Add storage_path column if missing
        if 'storage_path' not in columns:
            logger.info("[MIGRATE] Adding storage_path column")
            conn.execute(text("ALTER TABLE jobs ADD COLUMN storage_path TEXT"))
            # Set default for existing jobs
            conn.execute(text("UPDATE jobs SET storage_path = '/tmp/' || job_id WHERE storage_path IS NULL"))
            conn.execute(text("ALTER TABLE jobs ALTER COLUMN storage_path SET NOT NULL"))

        # Add cancelled_at column if missing
        if 'cancelled_at' not in columns:
            logger.info("[MIGRATE] Adding cancelled_at column")
            conn.execute(text("ALTER TABLE jobs ADD COLUMN cancelled_at TIMESTAMP"))

        # Ensure status column supports CANCELLED
        # (SQLite doesn't enforce enums, so this is just documentation)
        logger.info("[MIGRATE] Status enum now includes CANCELLED")

        # Create/update schema_metadata table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS schema_metadata (
                version TEXT NOT NULL,
                migrated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("DELETE FROM schema_metadata"))  # Clear old version
        conn.execute(text("INSERT INTO schema_metadata (version) VALUES ('1.3.0')"))

    logger.info("[OK] Database migrated to v1.3.0")
```

---

## Transient Models (Not Persisted)

### Entity 2: ValidationResult

**Purpose**: Represents the outcome of configuration validation

**Pydantic Model**:

```python
# src/validation/models.py
from pydantic import BaseModel
from typing import Literal

class FieldError(BaseModel):
    """Single field validation error"""
    field: str
    message: str
    code: str
    hint: str | None = None

    class Config:
        json_schema_extra = {
            "example": {
                "field": "pipeline",
                "message": "Pipeline 'invalid_pipeline' not found",
                "code": "PIPELINE_NOT_FOUND",
                "hint": "Available pipelines: person, face, audio, scene"
            }
        }

class FieldWarning(BaseModel):
    """Single field validation warning (non-blocking)"""
    field: str
    message: str
    suggestion: str | None = None

class ValidationResult(BaseModel):
    """Result of configuration validation"""
    valid: bool
    errors: list[FieldError] = []
    warnings: list[FieldWarning] = []

    class Config:
        json_schema_extra = {
            "example": {
                "valid": False,
                "errors": [
                    {
                        "field": "config.detection.confidence_threshold",
                        "message": "Value must be between 0.0 and 1.0",
                        "code": "VALUE_OUT_OF_RANGE",
                        "hint": "Use a value like 0.5 or 0.7"
                    }
                ],
                "warnings": [
                    {
                        "field": "config.batch_size",
                        "message": "Batch size 64 may exceed GPU memory on some devices",
                        "suggestion": "Consider batch_size=32 for 8GB GPUs"
                    }
                ]
            }
        }
```

**Usage Pattern**:

```python
# In validation endpoint
validator = ConfigValidator()
result = validator.validate(pipeline_name="face", config=request_config)

if not result.valid:
    raise InvalidConfigException(errors=result.errors)

return result  # 200 OK with warnings
```

---

## API Response Models

### Entity 3: ErrorEnvelope

**Purpose**: Standardized error response format for all API errors

**Pydantic Model**:

```python
# src/api/v1/errors.py
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Any

class ErrorDetail(BaseModel):
    """Detailed error information"""
    code: str = Field(..., description="Machine-readable error code (e.g., 'JOB_NOT_FOUND')")
    message: str = Field(..., description="Human-readable error message")
    detail: dict[str, Any] | None = Field(None, description="Additional context (field-level errors, etc.)")
    hint: str | None = Field(None, description="Suggested action to fix the error")
    field: str | None = Field(None, description="Field name for validation errors")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When error occurred (UTC)")

    class Config:
        json_schema_extra = {
            "example": {
                "code": "INVALID_CONFIG",
                "message": "Configuration validation failed",
                "detail": {
                    "errors": [
                        {"field": "pipeline", "message": "Pipeline 'xyz' not found"}
                    ]
                },
                "hint": "Use GET /api/v1/pipelines to list available pipelines",
                "field": None,
                "timestamp": "2025-10-11T10:30:00Z"
            }
        }

class ErrorEnvelope(BaseModel):
    """Standard error response wrapper"""
    error: ErrorDetail

    class Config:
        json_schema_extra = {
            "example": {
                "error": {
                    "code": "JOB_NOT_FOUND",
                    "message": "Job abc123 not found",
                    "hint": "Check job ID or use GET /api/v1/jobs to list jobs",
                    "timestamp": "2025-10-11T10:30:00Z"
                }
            }
        }
```

**Error Code Registry**:

| Code | HTTP Status | Meaning | Hint Template |
|------|-------------|---------|---------------|
| `JOB_NOT_FOUND` | 404 | Job ID doesn't exist | "Check job ID or use GET /api/v1/jobs" |
| `PIPELINE_NOT_FOUND` | 404 | Pipeline name invalid | "Available pipelines: {pipeline_list}" |
| `INVALID_CONFIG` | 400 | Config validation failed | "Fix validation errors: {error_summary}" |
| `JOB_ALREADY_COMPLETED` | 409 | Can't cancel completed job | "Job already in terminal state: {status}" |
| `STORAGE_FULL` | 507 | Insufficient disk space | "Free up {required_gb}GB or configure larger storage" |
| `GPU_OUT_OF_MEMORY` | 507 | GPU VRAM exhausted | "Reduce batch_size or wait for running jobs to finish" |
| `UNAUTHORIZED` | 401 | Missing/invalid auth token | "Provide valid API key or JWT token" |
| `FORBIDDEN` | 403 | Insufficient permissions | "Contact admin for access to this resource" |
| `INTERNAL_SERVER_ERROR` | 500 | Unexpected server error | "Contact support with timestamp: {timestamp}" |

---

## Relationships & Dependencies

### Job ↔ Storage
- One-to-one: Each job has exactly one storage directory
- Storage path format: `{STORAGE_ROOT}/{job_id}/`
- Contents: `video.mp4`, `results/*.json`, `logs/*.log`

### Job ↔ Pipelines
- One-to-many: Job references multiple pipeline names in config
- No explicit foreign key (pipelines are registry-driven, not DB entities)

### ValidationResult ↔ Registry
- Validation schemas loaded from pipeline registry metadata
- No persistence; validation is request-scoped

### ErrorEnvelope ↔ All Endpoints
- Universal: All HTTP errors (4xx, 5xx) use ErrorEnvelope format
- Registered via FastAPI exception handlers

---

## Indexing Strategy

### Performance Considerations

```sql
-- Existing indexes (assumed)
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_created_at ON jobs(created_at);

-- NEW indexes for v1.3.0 (optional, for performance)
CREATE INDEX idx_jobs_completed_at ON jobs(completed_at) WHERE completed_at IS NOT NULL;
CREATE INDEX idx_jobs_storage_path ON jobs(storage_path);  -- For cleanup queries
```

**Rationale**:
- `idx_jobs_status`: Fast queue position calculation (count PENDING)
- `idx_jobs_completed_at`: Fast storage cleanup queries (WHERE completed_at < cutoff)
- `idx_jobs_storage_path`: Verify storage integrity checks

---

## Data Validation Rules

### Job
- `job_id`: UUID v4 format, immutable once created
- `video_path`: Must be absolute path, file must exist at submission
- `storage_path`: Must be within `STORAGE_ROOT` (security check)
- `status`: Must follow valid state transitions (enforced in service layer)
- Timestamps: `created_at` < `started_at` < `completed_at` (if not null)

### ValidationResult
- If `valid=False`, `errors` list must not be empty
- Each `FieldError` must have non-empty `field`, `message`, `code`
- `hint` should provide actionable guidance (not just "invalid value")

### ErrorEnvelope
- `code` must match error code registry (uppercase, snake_case)
- `message` must be human-readable (not stack traces)
- `timestamp` always UTC (for cross-timezone support)

---

## Testing Strategy

### Unit Tests
- [ ] Job model state transitions (PENDING → RUNNING → COMPLETED, etc.)
- [ ] Job.to_dict() serialization includes all new fields
- [ ] Migration script handles missing columns gracefully
- [ ] ValidationResult with multiple errors serializes correctly
- [ ] ErrorEnvelope JSON matches OpenAPI schema

### Integration Tests
- [ ] Create job with custom storage_path, restart server, verify persistence
- [ ] Cancel job, verify cancelled_at timestamp set
- [ ] Invalid config returns ValidationResult with field-level errors
- [ ] All API endpoints return ErrorEnvelope on failure

### Migration Tests
- [ ] Migrate v1.2.0 database, verify all existing jobs get default storage_path
- [ ] Re-run migration (idempotent), no duplicate columns
- [ ] Schema version stored correctly in metadata table

---

## Backward Compatibility

### Breaking Changes
- **Status enum**: Existing code checking `status in [COMPLETED, FAILED]` must add `CANCELLED`
- **API responses**: Error format changes (clients must handle ErrorEnvelope)

### Non-Breaking Changes
- **New fields**: `storage_path`, `cancelled_at` added to Job (existing clients ignore unknown fields)
- **New endpoints**: `/cancel`, `/validate` are additive

### Migration Support
- Migration runs automatically on first v1.3.0 startup
- Existing jobs get sensible defaults (`storage_path = /tmp/{job_id}`)
- No manual intervention required

---

## Open Questions

1. **Storage path uniqueness**: Should we enforce unique storage_path in DB?
   - **Answer**: No, allow multiple jobs to share storage (e.g., batch processing)

2. **Job history retention**: How long to keep CANCELLED jobs in database?
   - **Answer**: Same as COMPLETED/FAILED (controlled by storage cleanup policy)

3. **Queue position caching**: Compute queue_position on-demand or cache?
   - **Answer**: Compute on-demand for now (simple COUNT query), optimize later if slow

---

**Status**: Design complete, ready for contract generation
**Next**: Generate OpenAPI contracts in contracts/ directory
