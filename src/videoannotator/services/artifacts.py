"""Service for managing job artifacts."""

from collections.abc import Iterator

from videoannotator.storage.manager import get_storage_provider
from videoannotator.storage.providers.base import ArtifactType, JobArtifact


def list_job_artifacts(
    job_id: str, include_types: list[ArtifactType] | None = None
) -> Iterator[JobArtifact]:
    """List artifacts for a specific job.

    Args:
        job_id: The unique identifier of the job.
        include_types: Optional list of artifact types to include.
                       If None, all types are included.

    Returns:
        Iterator[JobArtifact]: Iterator of artifact objects.
    """
    provider = get_storage_provider()

    for artifact in provider.list_files(job_id):
        if include_types is None or artifact.artifact_type in include_types:
            yield artifact


def get_annotation_artifacts(job_id: str) -> list[JobArtifact]:
    """Get all relevant artifacts for a job, including source video.

    Args:
        job_id: The unique identifier of the job.

    Returns:
        list[JobArtifact]: List of artifacts including video, annotations, reports, and logs.
    """
    return list(
        list_job_artifacts(
            job_id,
            include_types=[
                ArtifactType.ANNOTATION,
                ArtifactType.REPORT,
                ArtifactType.LOG,
                ArtifactType.VIDEO,
            ],
        )
    )
