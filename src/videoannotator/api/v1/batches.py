"""Batch/group endpoints for VideoAnnotator API (spec 008).

A "batch" is not a separately-created resource -- it's simply the set of
jobs currently carrying a given client-supplied `batch_id` (see
`POST /api/v1/jobs`'s optional `batch_id` form field). These endpoints
aggregate over that set: a computed-on-read summary, and a bulk retry that
reuses spec 006's single-job retry semantics per job.
"""

import logging
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Path, Query
from pydantic import BaseModel, Field

from ...batch.batch_summary import compute_batch_summary
from ...results_folder import folder_ref, run_folder_of
from ...storage.base import StorageBackend
from ..database import get_storage_backend
from ..errors import APIError
from ..middleware.auth import validate_api_key
from .exceptions import (
    JobAlreadyCompletedException,
    JobNotFoundException,
    JobNotRetryableException,
)
from .jobs import (
    FolderRef,
    RerunNotPossibleException,
    RerunRequest,
    cancel_job_by_id,
    create_rerun,
    retry_job,
    validate_pipeline_selection,
)

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
    batch_name: str | None = None
    dataset_id: str | None = None
    created_at: datetime | None = None
    total: int
    by_status: BatchStatusCounts
    completion_percentage: float
    estimated_seconds_remaining: float | None = None
    results_folder: FolderRef | None = Field(
        default=None,
        description="The run's results folder (spec 022); null for runs made "
        "before results folders existed",
    )


def _summary_response(batch_id: str, jobs: list[Any]) -> BatchSummaryResponse:
    summary = compute_batch_summary(batch_id, jobs)
    run_folder = next((f for f in map(run_folder_of, jobs) if f is not None), None)
    return BatchSummaryResponse(
        **summary.to_dict(), results_folder=folder_ref(run_folder)
    )


class BatchListResponse(BaseModel):
    batches: list[BatchSummaryResponse]
    total: int
    page: int
    per_page: int


class BatchJobSkipped(BaseModel):
    job_id: str
    reason: str


# Retained under its original name: the batch-retry response has documented
# this shape since 008 shipped, and clients type against it.
BatchRetrySkipped = BatchJobSkipped


class BatchRetryResponse(BaseModel):
    batch_id: str
    retried: list[str]
    skipped: list[BatchJobSkipped]


class BatchCancelResponse(BaseModel):
    batch_id: str
    cancelled: list[str]
    skipped: list[BatchJobSkipped]


def _load_batch_jobs(storage: StorageBackend, batch_id: str) -> list[Any]:
    job_ids = storage.list_jobs_by_batch(batch_id)
    jobs = []
    for job_id in job_ids:
        job = storage.load_job_metadata(job_id)
        if job is not None:
            jobs.append(job)
    return jobs


@router.get("", include_in_schema=False)
@router.get(
    "/",
    response_model=BatchListResponse,
    summary="List submission batches",
    description="""
Lists every batch the server currently knows about -- that is, every distinct
`batch_id` carried by at least one job -- most recently submitted first, with
the same aggregate summary `GET /api/v1/batches/{batch_id}` returns for one.

This is what lets a client show N videos submitted together as a single unit
without having to remember, client-side, which jobs it submitted together: the
grouping lives on the server and survives a page reload, a different browser,
or a different machine.

A batch has no independent existence, so this list is derived, not stored: a
batch appears here as soon as one job carries its id, and disappears once its
last member job is deleted. Jobs with no `batch_id` (standalone submissions)
are not represented here at all -- list those through `GET /api/v1/jobs`.
""",
)
async def list_batches(
    page: int = Query(1, ge=1, description="1-based page number"),
    per_page: int = Query(20, ge=1, le=100, description="Batches per page"),
    storage: StorageBackend = Depends(get_storage),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> BatchListResponse:
    """List batch summaries, newest submission first."""
    try:
        batch_ids = storage.list_batches()
        total = len(batch_ids)

        start = (page - 1) * per_page
        page_batch_ids = batch_ids[start : start + per_page]

        # Summaries are computed only for the requested page, so the cost of
        # this endpoint tracks per_page rather than the server's whole history.
        batches = []
        for member_batch_id in page_batch_ids:
            jobs = _load_batch_jobs(storage, member_batch_id)
            if not jobs:
                # Every member job was deleted between the id listing and here.
                continue
            batches.append(_summary_response(member_batch_id, jobs))

        return BatchListResponse(
            batches=batches, total=total, page=page, per_page=per_page
        )

    except APIError:
        raise
    except Exception as e:
        logger.error(f"[ERROR] Failed to list batches: {e}", exc_info=True)
        raise APIError(
            status_code=500,
            code="BATCH_LIST_FAILED",
            message=f"Failed to list batches: {e!s}",
            hint="Check server logs for details",
        ) from e


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
    return _summary_response(batch_id, _load_batch_jobs(storage, batch_id))


@router.post(
    "/{batch_id}/cancel",
    response_model=BatchCancelResponse,
    summary="Cancel every cancellable job in a batch",
    description="""
Cancels every job tagged with `batch_id` that is still pending or running,
applying the same semantics as `POST /api/v1/jobs/{job_id}/cancel` to each.
Jobs that have already completed or failed are left untouched and reported in
`skipped` with why -- cancelling a batch that is half-finished is a normal
thing to want, so this never fails the whole request because some of its jobs
are past the point of cancelling.

Jobs already cancelled are reported in `cancelled` (the operation is
idempotent per job), so calling this twice is safe.
""",
)
async def cancel_batch(
    batch_id: str = Path(..., description="The client-supplied batch identifier"),
    storage: StorageBackend = Depends(get_storage),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> BatchCancelResponse:
    """Cancel every still-cancellable job in a batch."""
    try:
        job_ids = storage.list_jobs_by_batch(batch_id)
        cancelled: list[str] = []
        skipped: list[BatchJobSkipped] = []

        for job_id in job_ids:
            try:
                cancel_job_by_id(job_id, storage)
                cancelled.append(job_id)
            except JobAlreadyCompletedException as e:
                skipped.append(BatchJobSkipped(job_id=job_id, reason=e.message))
            except JobNotFoundException:
                # Listed by list_jobs_by_batch a moment ago but gone now
                # (e.g. deleted concurrently) -- skip, don't fail the batch.
                skipped.append(
                    BatchJobSkipped(job_id=job_id, reason="job no longer exists")
                )

        return BatchCancelResponse(
            batch_id=batch_id, cancelled=cancelled, skipped=skipped
        )

    except APIError:
        raise
    except Exception as e:
        logger.error(f"[ERROR] Failed to cancel batch {batch_id}: {e}", exc_info=True)
        raise APIError(
            status_code=500,
            code="BATCH_CANCEL_FAILED",
            message=f"Failed to cancel batch: {e!s}",
            hint="Check server logs for details",
        ) from e


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
        skipped: list[BatchJobSkipped] = []

        for job_id in job_ids:
            try:
                retry_job(job_id, storage)
                retried.append(job_id)
            except JobNotRetryableException as e:
                skipped.append(
                    BatchJobSkipped(job_id=job_id, reason=e.hint or e.message)
                )
            except JobNotFoundException:
                # Listed by list_jobs_by_batch a moment ago but gone now
                # (e.g. deleted concurrently) -- skip, don't fail the batch.
                skipped.append(
                    BatchJobSkipped(job_id=job_id, reason="job no longer exists")
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


class BatchRerunResponse(BaseModel):
    batch_id: str = Field(description="The new batch")
    rerun_of_batch: str
    created: list[str]
    skipped: list[BatchJobSkipped]


@router.post(
    "/{batch_id}/rerun",
    response_model=BatchRerunResponse,
    status_code=201,
    summary="Run a batch again",
    description="""
Runs every finished job of a batch again (spec 019) as a new batch named
"<name> (rerun)", each job linked to its original by `rerun_of`. The original
batch is not changed. Optional `selected_pipelines` / `config` replace every
job's ("Edit and run again"); they are checked before any job is created. Jobs
still running, or whose video is gone, are reported in `skipped`.
""",
)
async def rerun_batch(
    request: RerunRequest | None = None,
    batch_id: str = Path(..., description="The batch to run again"),
    storage: StorageBackend = Depends(get_storage),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> BatchRerunResponse:
    """Run a batch's finished jobs again as a new batch."""
    jobs = _load_batch_jobs(storage, batch_id)
    if not jobs:
        raise APIError(
            status_code=404,
            code="BATCH_NOT_FOUND",
            message=f"Batch '{batch_id}' not found",
            hint="Check the batch ID, or use GET /api/v1/batches.",
        )
    request = request or RerunRequest()
    for job in jobs:
        validate_pipeline_selection(
            request.selected_pipelines
            if request.selected_pipelines is not None
            else job.selected_pipelines,
            request.config if request.config is not None else job.config,
        )

    new_batch_id = str(uuid.uuid4())
    name = next((j.batch_name for j in jobs if j.batch_name), None)
    new_name = f"{name} (rerun)" if name else "Rerun"
    created: list[str] = []
    skipped: list[BatchJobSkipped] = []
    for job in jobs:
        try:
            new = create_rerun(job, storage, request, new_batch_id, new_name)
            created.append(new.job_id)
        except RerunNotPossibleException as e:
            skipped.append(BatchJobSkipped(job_id=job.job_id, reason=e.message))
    return BatchRerunResponse(
        batch_id=new_batch_id, rerun_of_batch=batch_id, created=created, skipped=skipped
    )
