"""Stateless aggregation over jobs sharing a submission-batch identifier
(spec 008).

Deliberately not `ProgressTracker` (progress_tracker.py): that class tracks
one *live, in-process* run via start_job()/complete_job() callbacks, which
never fire for a batch queried cold from storage (e.g. after a server
restart, or from a request handler that never ran the jobs itself). Batch
summary here is computed entirely from each job's own persisted
started_at/completed_at -- the single source of truth per FR-002.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import BatchJob, JobStatus

_TERMINAL_STATUSES = (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)


@dataclass
class BatchSummary:
    """Aggregate, computed-on-read view over a batch's jobs."""

    batch_id: str
    total: int
    by_status: dict[str, int]
    completion_percentage: float
    estimated_seconds_remaining: float | None

    def to_dict(self) -> dict:
        """Convert to the API contract's exact response shape."""
        return {
            "batch_id": self.batch_id,
            "total": self.total,
            "by_status": self.by_status,
            "completion_percentage": self.completion_percentage,
            "estimated_seconds_remaining": self.estimated_seconds_remaining,
        }


def compute_batch_summary(batch_id: str, jobs: list[BatchJob]) -> BatchSummary:
    """Aggregate `jobs` (all jobs currently tagged with `batch_id`) into a
    `BatchSummary`.

    ETA (FR-003): average wall-clock duration (completed_at - started_at)
    across every COMPLETED job in the batch so far, times the number of
    jobs not yet in a terminal state. Recomputed fresh from every completed
    job each call, not just the first one or two, so it keeps reflecting
    real performance as more of the batch finishes (see spec's Edge Cases).
    `None` until at least one job has completed (FR-003) or once nothing
    remains to estimate.
    """
    by_status = {
        "pending": 0,
        "running": 0,
        "completed": 0,
        "failed": 0,
        "cancelled": 0,
    }
    for job in jobs:
        by_status[job.status.value] += 1

    total = len(jobs)
    terminal = sum(by_status[s.value] for s in _TERMINAL_STATUSES)
    completion_percentage = (terminal / total * 100.0) if total else 0.0

    completed_durations = [
        (job.completed_at - job.started_at).total_seconds()
        for job in jobs
        if job.status == JobStatus.COMPLETED
        and job.started_at is not None
        and job.completed_at is not None
    ]

    remaining = total - terminal
    estimated_seconds_remaining: float | None = None
    if completed_durations and remaining > 0:
        avg_duration = sum(completed_durations) / len(completed_durations)
        estimated_seconds_remaining = avg_duration * remaining

    return BatchSummary(
        batch_id=batch_id,
        total=total,
        by_status=by_status,
        completion_percentage=completion_percentage,
        estimated_seconds_remaining=estimated_seconds_remaining,
    )
