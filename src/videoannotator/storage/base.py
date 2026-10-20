"""Base storage backend interface for VideoAnnotator batch processing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from ..batch.types import BatchJob, BatchReport


class PipelineSummary(NamedTuple):
    status: str
    error: str | None
    sha256: str | None


@dataclass
class JobSummary:
    """What spec 029's video list needs of a job, without its results."""

    job_id: str
    video_path: str | None
    status: str
    created_at: datetime | None
    completed_at: datetime | None
    batch_id: str | None = None
    batch_name: str | None = None
    dataset_id: str | None = None
    rerun_of: str | None = None
    storage_path: str | None = None
    output_dir: str | None = None
    selected_pipelines: list[str] | None = None
    config: dict[str, Any] | None = None
    started_at: datetime | None = None
    pipelines: dict[str, PipelineSummary] = field(default_factory=dict)


def provenance_sha256(record: Any) -> str | None:
    """The input video's sha256 a provenance record holds (spec 017), if any."""
    if not isinstance(record, dict):
        return None
    sha = (record.get("input") or {}).get("sha256")
    return sha if isinstance(sha, str) and sha else None


def _value(status: Any) -> str:
    return str(getattr(status, "value", status))


def summary_of(job: BatchJob) -> JobSummary:
    return JobSummary(
        job_id=job.job_id,
        video_path=str(job.video_path) if job.video_path else None,
        status=_value(job.status),
        created_at=job.created_at,
        completed_at=job.completed_at,
        batch_id=job.batch_id,
        batch_name=job.batch_name,
        dataset_id=job.dataset_id,
        rerun_of=job.rerun_of,
        storage_path=str(job.storage_path) if job.storage_path else None,
        output_dir=str(job.output_dir) if job.output_dir else None,
        selected_pipelines=job.selected_pipelines,
        config=job.config,
        started_at=job.started_at,
        pipelines={
            name: PipelineSummary(
                _value(result.status),
                result.error_message,
                provenance_sha256(result.provenance),
            )
            for name, result in (job.pipeline_results or {}).items()
        },
    )


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def save_annotations(
        self, job_id: str, pipeline: str, annotations: list[dict[str, Any]]
    ) -> str:
        """Save pipeline annotations for a job.

        Args:
            job_id: Unique job identifier
            pipeline: Pipeline name (e.g., 'scene_detection', 'person_tracking')
            annotations: List of annotation dictionaries

        Returns:
            Path/identifier where annotations were saved
        """
        pass

    @abstractmethod
    def load_annotations(self, job_id: str, pipeline: str) -> list[dict[str, Any]]:
        """Load pipeline annotations for a job.

        Args:
            job_id: Unique job identifier
            pipeline: Pipeline name

        Returns:
            List of annotation dictionaries

        Raises:
            FileNotFoundError: If annotations don't exist
        """
        pass

    @abstractmethod
    def annotation_exists(self, job_id: str, pipeline: str) -> bool:
        """Check if annotations exist for a job and pipeline.

        Args:
            job_id: Unique job identifier
            pipeline: Pipeline name

        Returns:
            True if annotations exist, False otherwise
        """
        pass

    @abstractmethod
    def save_job_metadata(self, job: BatchJob) -> None:
        """Save job metadata.

        Args:
            job: BatchJob instance with metadata
        """
        pass

    @abstractmethod
    def load_job_metadata(self, job_id: str) -> BatchJob | None:
        """Load job metadata.

        Args:
            job_id: Unique job identifier

        Returns:
            BatchJob instance

        Raises:
            FileNotFoundError: If job metadata doesn't exist
        """
        pass

    @abstractmethod
    def list_jobs(self, status_filter: str | None = None) -> list[str]:
        """List all job IDs, optionally filtered by status.

        Args:
            status_filter: Optional status to filter by

        Returns:
            List of job IDs
        """
        pass

    def has_video_path(self, video_path: str) -> bool:
        """Whether some job read its video from `video_path` (spec 027)."""
        for job_id in self.list_jobs():
            try:
                job = self.load_job_metadata(job_id)
            except Exception:
                continue
            if job is not None and str(job.video_path) == video_path:
                return True
        return False

    def job_summaries(self) -> list[JobSummary]:
        """Every job, oldest first, without loading result files (spec 029).
        Backends with a query language override this per-job load."""
        summaries = []
        for job_id in self.list_jobs():
            try:
                job = self.load_job_metadata(job_id)
            except Exception:
                continue
            if job is not None:
                summaries.append(summary_of(job))
        return sorted(
            summaries, key=lambda s: s.created_at.timestamp() if s.created_at else 0
        )

    def jobs_signature(self) -> tuple[Any, ...]:
        """Changes whenever a job is added, removed or moves on; spec 029's
        video index rebuilds only then. The default always differs."""
        return (object(),)

    @abstractmethod
    def list_jobs_by_batch(self, batch_id: str) -> list[str]:
        """List job IDs sharing a given batch identifier (spec 008).

        Args:
            batch_id: The client-supplied batch identifier jobs were
                submitted with.

        Returns:
            List of job IDs currently tagged with this batch_id.
        """
        pass

    def list_jobs_by_dataset(self, dataset_id: str) -> list[str]:
        """Job IDs run from a saved dataset, oldest first (spec 025). Backends
        with a query language override this scan."""
        jobs = [self.load_job_metadata(job_id) for job_id in self.list_jobs()]
        matching = [j for j in jobs if j is not None and j.dataset_id == dataset_id]
        return [j.job_id for j in sorted(matching, key=lambda j: j.created_at)]

    def list_reruns(self, job_id: str) -> list[str]:
        """Jobs that run `job_id` again, oldest first (spec 019). Backends
        with an index override this; the default reads every job."""
        reruns = []
        for other_id in self.list_jobs():
            other = self.load_job_metadata(other_id)
            if other is not None and other.rerun_of == job_id:
                reruns.append(other)
        return [j.job_id for j in sorted(reruns, key=lambda j: j.created_at)]

    @abstractmethod
    def list_unbatched_jobs(self) -> list[str]:
        """List job IDs carrying no batch identifier.

        These are standalone submissions -- from the CLI, or from a client that
        predates batch tagging. They belong to no batch, so a batch listing
        never surfaces them.

        Returns:
            Job IDs with no batch_id.
        """
        pass

    @abstractmethod
    def list_batches(self) -> list[str]:
        """List every distinct batch identifier currently carried by a job.

        A batch has no independent existence -- this is a group-by over jobs,
        so a batch disappears once its last member job is deleted.

        Returns:
            Batch identifiers, most recently submitted first.
        """
        pass

    @abstractmethod
    def delete_job(self, job_id: str) -> bool:
        """Delete all data for a job.

        Args:
            job_id: Unique job identifier

        Returns:
            True if the job existed and was deleted, False if it was not found.
        """
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Get storage statistics.

        Returns:
            Dictionary with storage stats
        """
        pass

    @abstractmethod
    def save_report(self, report: BatchReport) -> None:
        """Save batch report.

        Args:
            report: BatchReport instance
        """
        pass

    @abstractmethod
    def load_report(self, batch_id: str) -> BatchReport | None:
        """Load batch report.

        Args:
            batch_id: Unique batch identifier

        Returns:
            BatchReport instance

        Raises:
            FileNotFoundError: If report doesn't exist
        """
        pass

    @abstractmethod
    def list_reports(self) -> list[BatchReport]:
        """List all batch report IDs.

        Returns:
            List of batch IDs
        """
        pass

    def close(self) -> None:  # noqa: B027 — intentional no-op default hook
        """Release any held database connections or file handles.

        Subclasses that own a connection pool (e.g. SQLAlchemy engine) must
        override this and call engine.dispose() so that the underlying file
        can be deleted on Windows without a WinError 32.
        """

    def cleanup_old_files(self, max_age_days: int) -> tuple[int, int]:
        """Clean up old files (optional implementation).

        Args:
            max_age_days: Maximum age in days

        Returns:
            Tuple of (deleted_jobs, deleted_reports)
        """
        return (0, 0)
