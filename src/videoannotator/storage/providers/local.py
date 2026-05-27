"""Local filesystem storage provider implementation."""

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

from videoannotator.storage.providers.base import (
    ArtifactType,
    JobArtifact,
    StorageProvider,
)
from videoannotator.utils.logging_config import get_logger

logger = get_logger("storage.local")


class LocalStorageProvider(StorageProvider):
    """Storage provider that uses the local filesystem."""

    def __init__(self, root_path: str | Path, create_dirs: bool = True):
        """Initialize the local storage provider.

        Args:
            root_path: Root directory for storage.
            create_dirs: Whether to create the root directory if it doesn't exist.
        """
        self.root_path = Path(root_path).resolve()
        self.create_dirs = create_dirs

    def initialize(self) -> None:
        """Initialize the storage provider."""
        if self.create_dirs:
            self.root_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initialized local storage at {self.root_path}")
        elif not self.root_path.exists():
            logger.warning(f"Storage root {self.root_path} does not exist")

    def _get_job_path(self, job_id: str) -> Path:
        """Get the absolute path for a job directory."""
        return self.root_path / job_id

    def create_job_dir(self, job_id: str) -> None:
        """Create a directory for a specific job."""
        job_path = self._get_job_path(job_id)
        job_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created job directory: {job_path}")

    def save_file(self, job_id: str, relative_path: str, content: BinaryIO) -> str:
        """Save a file to the job storage."""
        full_path = self._get_job_path(job_id) / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        with open(full_path, "wb") as f:
            shutil.copyfileobj(content, f)

        logger.debug(f"Saved file: {full_path}")
        return str(full_path)

    def get_file(self, job_id: str, relative_path: str) -> BinaryIO:
        """Open a file for reading."""
        full_path = self._get_job_path(job_id) / relative_path
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {full_path}")
        return open(full_path, "rb")

    def list_files(self, job_id: str) -> Iterator[JobArtifact]:
        """List all files for a specific job."""
        job_path = self._get_job_path(job_id)
        if not job_path.exists():
            return

        for file_path in job_path.rglob("*"):
            if file_path.is_file():
                relative_path = str(file_path.relative_to(job_path))
                yield JobArtifact(
                    job_id=job_id,
                    path=relative_path,
                    name=file_path.name,
                    size_bytes=file_path.stat().st_size,
                    artifact_type=self._determine_artifact_type(file_path.name),
                )

    def exists(self, job_id: str, relative_path: str) -> bool:
        """Check if a file exists."""
        full_path = self._get_job_path(job_id) / relative_path
        return full_path.exists()

    def get_absolute_path(self, job_id: str, relative_path: str) -> Path:
        """Get the absolute local path of a file."""
        return self._get_job_path(job_id) / relative_path

    def _determine_artifact_type(self, filename: str) -> ArtifactType:
        """Determine artifact type based on filename extension."""
        ext = filename.lower().split(".")[-1] if "." in filename else ""
        if ext in ["mp4", "avi", "mov", "mkv"]:
            return ArtifactType.VIDEO
        elif ext in ["json", "csv", "srt", "vtt", "rttm"]:
            return ArtifactType.ANNOTATION
        elif ext in ["md", "txt", "pdf"]:
            return ArtifactType.REPORT
        elif ext in ["log"]:
            return ArtifactType.LOG
        return ArtifactType.OTHER
