from unittest.mock import Mock, patch
from videoannotator.services.artifacts import get_annotation_artifacts
from videoannotator.storage.providers.base import ArtifactType, JobArtifact


def test_get_annotation_artifacts_includes_video():
    """Test that get_annotation_artifacts includes video files."""
    job_id = "test-job-123"

    # Mock artifacts
    mock_artifacts = [
        JobArtifact(job_id, "video.mp4", "video.mp4", 1000, ArtifactType.VIDEO),
        JobArtifact(job_id, "output.json", "output.json", 100, ArtifactType.ANNOTATION),
        JobArtifact(job_id, "report.md", "report.md", 50, ArtifactType.REPORT),
        JobArtifact(job_id, "process.log", "process.log", 10, ArtifactType.LOG),
        JobArtifact(job_id, "temp.tmp", "temp.tmp", 5, ArtifactType.OTHER),
    ]

    # Mock provider
    mock_provider = Mock()
    mock_provider.list_files.return_value = iter(mock_artifacts)

    # Patch get_storage_provider
    with patch(
        "videoannotator.services.artifacts.get_storage_provider",
        return_value=mock_provider,
    ):
        artifacts = get_annotation_artifacts(job_id)

        # Verify all expected types are present
        types = [a.artifact_type for a in artifacts]
        assert ArtifactType.VIDEO in types
        assert ArtifactType.ANNOTATION in types
        assert ArtifactType.REPORT in types
        assert ArtifactType.LOG in types
        assert ArtifactType.OTHER not in types

        # Verify count (4 expected: video, annotation, report, log)
        assert len(artifacts) == 4
