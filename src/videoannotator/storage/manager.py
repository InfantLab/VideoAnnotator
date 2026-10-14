"""Storage manager factory.

This module provides a factory for obtaining the configured storage provider.
"""

from functools import lru_cache
from pathlib import Path

from videoannotator.storage.config import get_storage_root
from videoannotator.storage.providers.base import StorageProvider
from videoannotator.storage.providers.local import LocalStorageProvider
from videoannotator.utils.logging_config import get_logger

logger = get_logger("storage.manager")


@lru_cache
def get_storage_provider() -> StorageProvider:
    """Get the configured storage provider instance.

    Returns:
        StorageProvider: The singleton storage provider instance.
    """
    # In the future, we will read the provider type from config.
    # For now, we default to LocalStorageProvider.

    root_path = get_storage_root()
    logger.info(f"Initializing storage provider with root: {root_path}")

    provider = LocalStorageProvider(root_path=root_path)
    provider.initialize()

    # Validate write permissions
    try:
        test_file = root_path / ".write_test"
        test_file.touch()
        test_file.unlink()
    except Exception as e:
        logger.warning(f"Storage root {root_path} is not writable: {e}")
        # We don't raise here to allow read-only scenarios if intended,
        # but for a job processor this is likely fatal.

    return provider


def provider_for_job(
    job_id: str, recorded_folder: str | Path | None
) -> StorageProvider:
    """The provider holding `job_id`'s files.

    Usually the configured one. A job made before v1.6.0 recorded a folder
    under `./storage/jobs`, outside today's storage root; it is read from
    there, so upgrading doesn't strand old results.
    """
    provider = get_storage_provider()
    if recorded_folder:
        folder = Path(recorded_folder)
        if (
            folder.name == job_id
            and folder.is_dir()
            and folder != provider.get_absolute_path(job_id, "")
        ):
            return LocalStorageProvider(root_path=folder.parent, create_dirs=False)
    return provider
