"""
Base storage backend interface for VideoAnnotator batch processing.
"""

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
        """
        Save pipeline annotations for a job.

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
        """
        Load pipeline annotations for a job.

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
        """
        Check if annotations exist for a job and pipeline.

        Args:
            job_id: Unique job identifier
            pipeline: Pipeline name

        Returns:
            True if annotations exist, False otherwise
        """
        pass

    @abstractmethod
    def save_job_metadata(self, job: "BatchJob") -> None:
        """
        Save job metadata.

        Args:
            job: BatchJob instance with metadata
        """
        pass

    @abstractmethod
    def load_job_metadata(self, job_id: str) -> "BatchJob":
        """
        Load job metadata.

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
        """
        List all job IDs, optionally filtered by status.

        Args:
            status_filter: Optional status to filter by

        Returns:
            List of job IDs
        """
        pass

    @abstractmethod
    def delete_job(self, job_id: str) -> None:
        """
        Delete all data for a job.

        Args:
            job_id: Unique job identifier
        """
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dictionary with storage stats
        """
        pass

    @abstractmethod
    def save_report(self, report: "BatchReport") -> None:
        """
        Save batch report.

        Args:
            report: BatchReport instance
        """
        pass

    @abstractmethod
    def load_report(self, batch_id: str) -> "BatchReport":
        """
        Load batch report.

        Args:
            batch_id: Unique batch identifier

        Returns:
            BatchReport instance

        Raises:
            FileNotFoundError: If report doesn't exist
        """
        pass

    @abstractmethod
    def list_reports(self) -> list[str]:
        """
        List all batch report IDs.

        Returns:
            List of batch IDs
        """
        pass

    def cleanup_old_files(self, max_age_days: int) -> tuple[int, int]:
        """
        Clean up old files (optional implementation).

        Args:
            max_age_days: Maximum age in days

        Returns:
            Tuple of (deleted_jobs, deleted_reports)
        """
        return (0, 0)
