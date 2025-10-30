# API Contracts Directory

**Purpose**: OpenAPI specifications for new/modified endpoints in v1.3.0

## Files

### [`job-cancellation.yaml`](./job-cancellation.yaml)
**Endpoint**: `POST /api/v1/jobs/{job_id}/cancel`
**Purpose**: Cancel a running or pending job with proper GPU cleanup
**Key Features**:
- Idempotent (safe to call multiple times)
- Returns 200 even if job already completed (graceful handling)
- Includes CancellationResponse schema with status + timestamp
- Documents error cases (404 for not found, 500 for cancellation failure)

### [`config-validation.yaml`](./config-validation.yaml)
**Endpoint**: `POST /api/v1/pipelines/validate`
**Purpose**: Validate pipeline configuration without submitting job
**Key Features**:
- Returns 200 OK for both valid and invalid configs (check `valid` field)
- Field-level errors with hints (e.g., "Value must be between 0.0 and 1.0")
- Registry-driven validation (Pydantic schemas from pipeline metadata)
- Target performance: <200ms validation time

### [`health.yaml`](./health.yaml)
**Endpoint**: `GET /api/v1/health?detailed=true`
**Purpose**: Enhanced health check with optional system diagnostics
**Key Features**:
- Basic mode: Fast liveness check (200 OK)
- Detailed mode: Database, storage, GPU, registry status
- Returns 503 if unhealthy (detailed mode only)
- Used by load balancers (basic) and monitoring systems (detailed)

### [`error-envelope.yaml`](./error-envelope.yaml)
**Schema**: `ErrorEnvelope` (used by all endpoints)
**Purpose**: Standardized error response format for 4xx/5xx
**Key Features**:
- Consistent structure across all endpoints
- Machine-readable error codes (e.g., `JOB_NOT_FOUND`, `INVALID_CONFIG`)
- Human-readable messages + actionable hints
- Optional `detail` object for additional context (e.g., field-level validation errors)
- ISO 8601 UTC timestamp for error tracking

## Error Code Registry

All endpoints use standardized error codes defined in [`error-envelope.yaml`](./error-envelope.yaml):

| Code | HTTP Status | Meaning |
|------|-------------|---------|
| `JOB_NOT_FOUND` | 404 | Job ID doesn't exist |
| `PIPELINE_NOT_FOUND` | 404 | Pipeline name not in registry |
| `INVALID_CONFIG` | 400 | Config validation failed |
| `JOB_ALREADY_COMPLETED` | 409 | Cannot cancel terminal job |
| `STORAGE_FULL` | 507 | Insufficient disk space |
| `GPU_OUT_OF_MEMORY` | 507 | GPU VRAM exhausted |
| `UNAUTHORIZED` | 401 | Missing/invalid auth |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `CANCELLATION_FAILED` | 500 | Worker termination failed |
| `INTERNAL_SERVER_ERROR` | 500 | Unexpected server error |

## Usage

These contracts serve as:
1. **Implementation Reference**: Detailed specs for FastAPI endpoint implementation
2. **Testing Guide**: Expected request/response formats for integration tests
3. **Documentation Source**: Can be imported into API documentation tools (Swagger, Redoc)

## Validation

To validate OpenAPI specs:

```bash
# Install openapi-spec-validator
uv add --dev openapi-spec-validator

# Validate each contract
openapi-spec-validator contracts/job-cancellation.yaml
openapi-spec-validator contracts/config-validation.yaml
openapi-spec-validator contracts/health.yaml
openapi-spec-validator contracts/error-envelope.yaml
```

## Integration with FastAPI

FastAPI will auto-generate OpenAPI specs from code. These contracts ensure:
- Code matches design intent
- Documentation is comprehensive
- Examples are realistic

Example implementation pattern:

```python
# src/api/v1/jobs.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()

class CancellationResponse(BaseModel):
    """Matches schema in job-cancellation.yaml"""
    job_id: str
    status: str
    cancelled_at: datetime | None
    message: str

@router.post(
    "/jobs/{job_id}/cancel",
    response_model=CancellationResponse,
    responses={
        404: {"model": ErrorEnvelope, "description": "Job not found"},
        500: {"model": ErrorEnvelope, "description": "Cancellation failed"},
    },
    summary="Cancel a job",
    description="Cancel a pending or running job. See contract: contracts/job-cancellation.yaml",
)
async def cancel_job(job_id: str) -> CancellationResponse:
    # Implementation here
    pass
```

## Related Files

- **Data Model**: [`../data-model.md`](../data-model.md) - Database schema and entity definitions
- **Implementation Guide**: [`../quickstart.md`](../quickstart.md) - Phase-by-phase implementation sequence
- **Research Decisions**: [`../research.md`](../research.md) - Technical decisions and rationale

---

**Status**: Design phase complete
**Next**: Implement endpoints per quickstart.md Phase 1A-1B
