"""Batch/group endpoints for VideoAnnotator API (spec 008).

A "batch" is not a separately-created resource -- it's simply the set of
jobs currently carrying a given client-supplied `batch_id` (see
`POST /api/v1/jobs`'s optional `batch_id` form field). These endpoints
aggregate over that set: a computed-on-read summary, and a bulk retry that
reuses spec 006's single-job retry semantics per job.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, Path
from pydantic import BaseModel

from ...batch.batch_summary import compute_batch_summary
from ...storage.base import StorageBackend
from ..database import get_storage_backend
from ..errors import APIError
from ..middleware.auth import validate_api_key
from .exceptions import JobNotFoundException, JobNotRetryableException
from .jobs import retry_job

logger = logging.getLogger("videoannotator.api")

router = APIRouter()


def get_storage() -> StorageBackend:
    """Get storage backend for batch aggregation (mirrors jobs.py)."""
    return get_storage_backend()


class BatchStatusCounts(BaseModel):
    pending: int
    running: int
    completed: int
    failed: int
    cancelled: int


class BatchSummaryResponse(BaseModel):
    batch_id: str
    total: int
    by_status: BatchStatusCounts
    completion_percentage: float
    estimated_seconds_remaining: float | None = None


class BatchRetrySkipped(BaseModel):
    job_id: str
    reason: str


class BatchRetryResponse(BaseModel):
    batch_id: str
    retried: list[str]
    skipped: list[BatchRetrySkipped]


def _load_batch_jobs(storage: StorageBackend, batch_id: str) -> list[Any]:
    job_ids = storage.list_jobs_by_batch(batch_id)
    jobs = []
    for job_id in job_ids:
        job = storage.load_job_metadata(job_id)
        if job is not None:
            jobs.append(job)
    return jobs


@router.get(
    "/{batch_id}",
    response_model=BatchSummaryResponse,
    summary="Get aggregate status for a submission batch",
    description="""
Aggregates every job currently tagged with `batch_id` (via `POST /api/v1/jobs`'s
`batch_id` form field) into one summary: counts by state, overall completion
percentage, and (once at least one job has completed) a time-remaining estimate
computed from real observed per-job processing time (FR-002/FR-003).

A batch that no job currently references returns an empty summary (`total: 0`)
rather than 404 -- a batch isn't a resource that can be "not found," it's just
whatever the query over `jobs` currently returns (see spec's Key Entities).
""",
)
async def get_batch_summary(
    batch_id: str = Path(..., description="The client-supplied batch identifier"),
    storage: StorageBackend = Depends(get_storage),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> BatchSummaryResponse:
    """Get aggregate status for all jobs sharing a batch identifier."""
    jobs = _load_batch_jobs(storage, batch_id)
    summary = compute_batch_summary(batch_id, jobs)
    return BatchSummaryResponse(**summary.to_dict())


@router.post(
    "/{batch_id}/retry",
    response_model=BatchRetryResponse,
    summary="Retry every retryable job in a batch",
    description="""
Retries every job tagged with `batch_id` that is currently in a retryable
terminal state (failed/cancelled with its original video still available),
applying spec 006's single-job retry semantics to each (FR-004). Jobs that
are still running, already completed, or otherwise not retryable are left
untouched and reported in `skipped` with why -- this never fails the whole
request because of jobs that simply don't need retrying (FR-005).
""",
)
async def retry_batch(
    batch_id: str = Path(..., description="The client-supplied batch identifier"),
    storage: StorageBackend = Depends(get_storage),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> BatchRetryResponse:
    """Retry every currently-retryable job in a batch."""
    try:
        job_ids = storage.list_jobs_by_batch(batch_id)
        retried: list[str] = []
        skipped: list[BatchRetrySkipped] = []

        for job_id in job_ids:
            try:
                retry_job(job_id, storage)
                retried.append(job_id)
            except JobNotRetryableException as e:
                skipped.append(
                    BatchRetrySkipped(job_id=job_id, reason=e.hint or e.message)
                )
            except JobNotFoundException:
                # Listed by list_jobs_by_batch a moment ago but gone now
                # (e.g. deleted concurrently) -- skip, don't fail the batch.
                skipped.append(
                    BatchRetrySkipped(job_id=job_id, reason="job no longer exists")
                )

        return BatchRetryResponse(batch_id=batch_id, retried=retried, skipped=skipped)

    except APIError:
        raise
    except Exception as e:
        logger.error(f"[ERROR] Failed to retry batch {batch_id}: {e}", exc_info=True)
        raise APIError(
            status_code=500,
            code="BATCH_RETRY_FAILED",
            message=f"Failed to retry batch: {e!s}",
            hint="Check server logs for details",
        ) from e
