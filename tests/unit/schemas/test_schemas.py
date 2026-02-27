"""Unit tests for schema modules (industry_standards + storage)."""

import pytest

from videoannotator.schemas.industry_standards import (
    IndustryStandardsPlaceholder,
    get_coco_format_for_pipeline,
    get_webvtt_format_for_audio,
)
from videoannotator.schemas.storage import (
    JobArtifact,
    StorageConfig,
    StorageProviderType,
)
from videoannotator.storage.providers.base import ArtifactType

# ---------------------------------------------------------------------------
# industry_standards.py
# ---------------------------------------------------------------------------


class TestIndustryStandardsPlaceholder:
    """Tests for the migrated industry_standards module."""

    def test_placeholder_class_exists(self):
        """Placeholder class can be instantiated for backward compatibility."""
        obj = IndustryStandardsPlaceholder()
        assert obj is not None

    def test_get_coco_format_returns_none(self):
        """Legacy COCO stub returns None for any pipeline type."""
        assert get_coco_format_for_pipeline("face_analysis") is None
        assert get_coco_format_for_pipeline("person_tracking") is None
        assert get_coco_format_for_pipeline("") is None

    def test_get_webvtt_format_returns_none(self):
        """Legacy WebVTT stub returns None."""
        assert get_webvtt_format_for_audio() is None


# ---------------------------------------------------------------------------
# storage.py — StorageProviderType enum
# ---------------------------------------------------------------------------


class TestStorageProviderType:
    """Tests for the StorageProviderType enum."""

    def test_local_value(self):
        assert StorageProviderType.LOCAL == "local"
        assert StorageProviderType.LOCAL.value == "local"

    def test_is_string_enum(self):
        assert isinstance(StorageProviderType.LOCAL, str)


# ---------------------------------------------------------------------------
# storage.py — JobArtifact model
# ---------------------------------------------------------------------------


class TestJobArtifact:
    """Tests for the JobArtifact Pydantic model."""

    def test_valid_artifact(self):
        artifact = JobArtifact(
            job_id="job-123",
            path="results/output.json",
            name="output.json",
            size_bytes=1024,
            artifact_type=ArtifactType.ANNOTATION,
        )
        assert artifact.job_id == "job-123"
        assert artifact.path == "results/output.json"
        assert artifact.name == "output.json"
        assert artifact.size_bytes == 1024
        assert artifact.artifact_type == ArtifactType.ANNOTATION

    def test_all_artifact_types_accepted(self):
        for atype in ArtifactType:
            artifact = JobArtifact(
                job_id="j1",
                path="p",
                name="n",
                size_bytes=0,
                artifact_type=atype,
            )
            assert artifact.artifact_type == atype

    def test_missing_required_field_raises(self):
        with pytest.raises(ValueError):
            JobArtifact(job_id="j1", path="p", name="n", size_bytes=10)

    def test_serialization_round_trip(self):
        artifact = JobArtifact(
            job_id="j1",
            path="a/b.txt",
            name="b.txt",
            size_bytes=42,
            artifact_type=ArtifactType.LOG,
        )
        data = artifact.model_dump()
        restored = JobArtifact(**data)
        assert restored == artifact


# ---------------------------------------------------------------------------
# storage.py — StorageConfig model
# ---------------------------------------------------------------------------


class TestStorageConfig:
    """Tests for the StorageConfig Pydantic model."""

    def test_defaults(self):
        config = StorageConfig()
        assert config.provider == StorageProviderType.LOCAL
        assert config.root_path == "./storage/jobs"
        assert config.options == {}

    def test_custom_values(self):
        config = StorageConfig(
            provider=StorageProviderType.LOCAL,
            root_path="/data/jobs",
            options={"max_size": 1024},
        )
        assert config.root_path == "/data/jobs"
        assert config.options["max_size"] == 1024

    def test_serialization_round_trip(self):
        config = StorageConfig(root_path="/tmp/test", options={"a": 1})
        data = config.model_dump()
        restored = StorageConfig(**data)
        assert restored == config
