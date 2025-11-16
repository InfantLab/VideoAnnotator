"""Clean up orphaned storage directories from deleted jobs.

This script removes storage directories for jobs that no longer exist in the database.
Useful for cleaning up after tests or manual database cleanup.
"""

import shutil

from videoannotator.api.database import get_storage_backend
from videoannotator.storage.config import get_storage_root


def cleanup_orphaned_storage():
    """Remove storage directories for jobs not in database."""
    storage = get_storage_backend()
    storage_root = get_storage_root()

    if not storage_root.exists():
        print(f"Storage root does not exist: {storage_root}")
        return

    # Get all job IDs from database
    try:
        db_job_ids = set(storage.list_jobs())
        print(f"Found {len(db_job_ids)} jobs in database")
    except Exception as e:
        print(f"Error loading jobs from database: {e}")
        return

    # Get all storage directories
    storage_dirs = [d for d in storage_root.iterdir() if d.is_dir()]
    print(f"Found {len(storage_dirs)} storage directories")

    # Find orphaned directories
    orphaned = []
    for storage_dir in storage_dirs:
        job_id = storage_dir.name
        if job_id not in db_job_ids:
            orphaned.append(storage_dir)

    if not orphaned:
        print("No orphaned storage directories found!")
        return

    print(f"\nFound {len(orphaned)} orphaned storage directories:")
    for storage_dir in orphaned:
        # Calculate directory size
        total_size = sum(
            f.stat().st_size for f in storage_dir.rglob("*") if f.is_file()
        )
        size_mb = total_size / (1024 * 1024)
        print(f"  - {storage_dir.name} ({size_mb:.2f} MB)")

    # Ask for confirmation
    response = input(f"\nDelete {len(orphaned)} orphaned directories? (y/N): ")
    if response.lower() != "y":
        print("Cancelled")
        return

    # Delete orphaned directories
    deleted_count = 0
    for storage_dir in orphaned:
        try:
            shutil.rmtree(storage_dir)
            print(f"Deleted: {storage_dir.name}")
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {storage_dir.name}: {e}")

    print(f"\nDeleted {deleted_count}/{len(orphaned)} directories")


if __name__ == "__main__":
    cleanup_orphaned_storage()
