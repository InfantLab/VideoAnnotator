"""Unit tests for the stateless batch-summary aggregation (spec 008)."""

from datetime import datetime, timedelta

from videoannotator.batch.batch_summary import compute_batch_summary
from videoannotator.batch.types import BatchJob, JobStatus


def _job(status: JobStatus, started_at=None, completed_at=None) -> BatchJob:
    return BatchJob(status=status, started_at=started_at, completed_at=completed_at)


class TestComputeBatchSummary:
    def test_counts_by_status_match_true_job_states(self):
        """US1 acceptance scenario 1."""
        jobs = [
            _job(JobStatus.PENDING),
            _job(JobStatus.RUNNING),
            _job(JobStatus.RUNNING),
            _job(JobStatus.COMPLETED),
            _job(JobStatus.FAILED),
            _job(JobStatus.CANCELLED),
        ]
        summary = compute_batch_summary("b1", jobs)
        assert summary.total == 6
        assert summary.by_status == {
            "pending": 1,
            "running": 2,
            "completed": 1,
            "failed": 1,
            "cancelled": 1,
        }

    def test_no_completed_jobs_yet_gives_null_eta(self):
        """FR-003: ETA is null until at least one job has completed."""
        jobs = [_job(JobStatus.PENDING), _job(JobStatus.RUNNING)]
        summary = compute_batch_summary("b1", jobs)
        assert summary.estimated_seconds_remaining is None

    def test_eta_uses_real_observed_processing_time(self):
        """FR-003/SC-002: ETA reflects real observed per-job duration."""
        now = datetime.now()
        completed = _job(
            JobStatus.COMPLETED,
            started_at=now - timedelta(seconds=20),
            completed_at=now,
        )
        pending = [_job(JobStatus.PENDING), _job(JobStatus.PENDING)]
        summary = compute_batch_summary("b1", [completed, *pending])
        # avg duration (20s) * 2 remaining pending jobs
        assert summary.estimated_seconds_remaining == 40.0

    def test_eta_averages_across_all_completed_jobs_not_just_first(self):
        """Edge case: ETA stays meaningful as more jobs complete, not a
        fixed extrapolation from the first one or two results."""
        now = datetime.now()
        completed_a = _job(
            JobStatus.COMPLETED,
            started_at=now - timedelta(seconds=10),
            completed_at=now,
        )
        completed_b = _job(
            JobStatus.COMPLETED,
            started_at=now - timedelta(seconds=30),
            completed_at=now,
        )
        pending = _job(JobStatus.PENDING)
        summary = compute_batch_summary("b1", [completed_a, completed_b, pending])
        # avg duration (10+30)/2 = 20s * 1 remaining
        assert summary.estimated_seconds_remaining == 20.0

    def test_fully_terminal_batch_reports_complete_with_no_eta(self):
        """US1 acceptance scenario 3."""
        now = datetime.now()
        jobs = [
            _job(
                JobStatus.COMPLETED,
                started_at=now - timedelta(seconds=5),
                completed_at=now,
            ),
            _job(JobStatus.FAILED),
            _job(JobStatus.CANCELLED),
        ]
        summary = compute_batch_summary("b1", jobs)
        assert summary.completion_percentage == 100.0
        assert summary.estimated_seconds_remaining is None

    def test_empty_batch_does_not_error(self):
        summary = compute_batch_summary("nonexistent", [])
        assert summary.total == 0
        assert summary.completion_percentage == 0.0
        assert summary.estimated_seconds_remaining is None

    def test_completion_percentage_counts_all_terminal_states(self):
        jobs = [
            _job(
                JobStatus.COMPLETED,
                started_at=datetime.now(),
                completed_at=datetime.now(),
            ),
            _job(JobStatus.FAILED),
            _job(JobStatus.CANCELLED),
            _job(JobStatus.PENDING),
        ]
        summary = compute_batch_summary("b1", jobs)
        assert summary.completion_percentage == 75.0
