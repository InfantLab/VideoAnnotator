"""Storage configuration schemas."""

from enum import Enum

from pydantic import BaseModel, Field

from ..storage.providers.base import ArtifactType


class JobArtifact(BaseModel):
    """Represents a file stored in the job storage."""

    job_id: str
    path: str
    name: str
    size_bytes: int
    artifact_type: ArtifactType


class StorageProviderType(str, Enum):
    """Supported storage provider types."""

    LOCAL = "local"
    # Future: S3 = "s3"
    # Future: ONEDRIVE = "onedrive"


class StorageConfig(BaseModel):
    """Configuration for the storage subsystem."""

    provider: StorageProviderType = Field(
        default=StorageProviderType.LOCAL,
        description="Type of storage provider to use",
    )
    root_path: str = Field(
        default="./storage/jobs",
        description="Root directory for job storage",
    )
    options: dict = Field(
        default_factory=dict,
        description="Provider-specific options",
    )
