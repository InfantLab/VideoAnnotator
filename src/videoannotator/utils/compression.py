"""Compression utilities for VideoAnnotator."""

import tempfile
import zipfile
from collections.abc import Generator
from pathlib import Path

from videoannotator.storage.manager import get_storage_provider
from videoannotator.storage.providers.base import JobArtifact
from videoannotator.utils.logging_config import get_logger

logger = get_logger("utils.compression")


def create_job_zip_archive(
    job_id: str, artifacts: list[JobArtifact]
) -> Generator[bytes, None, None]:
    """Create a ZIP archive of job artifacts and stream it.

    Note: This implementation creates a temporary file on disk to ensure
    ZIP integrity and compatibility, then streams that file.

    Args:
        job_id: The unique identifier of the job.
        artifacts: List of artifacts to include in the ZIP.

    Yields:
        bytes: Chunks of the ZIP file.
    """
    provider = get_storage_provider()

    # Create a temporary file for the ZIP archive
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=True) as temp_zip:
        temp_zip_path = Path(temp_zip.name)
        logger.debug(f"Creating temporary ZIP at {temp_zip_path}")

        try:
            # Write artifacts to the ZIP file
            with zipfile.ZipFile(temp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                for artifact in artifacts:
                    try:
                        # Get the absolute path if possible (efficient for local)
                        # For cloud, we might need to read content and write
                        try:
                            file_path = provider.get_absolute_path(
                                job_id, artifact.path
                            )
                            zf.write(file_path, arcname=artifact.path)
                        except NotImplementedError:
                            # Fallback for providers without local paths (read into memory)
                            # Note: This might be memory intensive for large files
                            with provider.get_file(job_id, artifact.path) as f:
                                zf.writestr(artifact.path, f.read())

                    except Exception as e:
                        logger.error(
                            f"Failed to add artifact {artifact.path} to ZIP: {e}"
                        )

            # Flush to ensure all data is written
            temp_zip.flush()

            # Rewind and stream the file
            temp_zip.seek(0)
            while True:
                chunk = temp_zip.read(8192)  # 8KB chunks
                if not chunk:
                    break
                yield chunk

        except Exception as e:
            logger.error(f"Error creating ZIP archive: {e}")
            raise
