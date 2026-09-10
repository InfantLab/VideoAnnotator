"""Base storage backend interface for VideoAnnotator batch processing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..batch.types import BatchJob, BatchReport


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
