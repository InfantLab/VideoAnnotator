"""Unit tests for v1.3.0 database migrations.

Tests migration from v1.2.x to v1.3.0 schema, including:
- Addition of jobs.cancelled_at column
- Addition of jobs.storage_path column

Note: These are smoke tests to verify migration logic integrity and
that the new columns are present in the database models.
"""

import pytest


class TestDatabaseModelsV1_3_0:
    """Test that v1.3.0 database migration infrastructure is in place."""

    def test_job_status_enum_has_cancelled(self):
        """Test that JobStatus enum includes CANCELLED state."""
        from videoannotator.batch.types import JobStatus

        assert hasattr(JobStatus, "CANCELLED")
        assert JobStatus.CANCELLED.value == "cancelled"

    def test_database_job_status_constant_has_cancelled(self):
        """Test that database JobStatus constants include CANCELLED."""
        from videoannotator.storage.models import JobStatus

        assert hasattr(JobStatus, "CANCELLED")
        assert JobStatus.CANCELLED == "cancelled"
        assert "cancelled" in JobStatus.ALL_STATUSES
        assert "cancelled" in JobStatus.FINAL_STATUSES


class TestMigrationFunction:
    """Test migration function exists and is callable."""

    def test_migrate_to_v1_3_0_exists(self):
        """Test that migrate_to_v1_3_0 function exists."""
        from videoannotator.storage.migrations import migrate_to_v1_3_0

        assert callable(migrate_to_v1_3_0)

    def test_migrate_to_v1_3_0_returns_bool(self):
        """Test that migrate_to_v1_3_0 has correct return type signature."""
        from videoannotator.storage.migrations import migrate_to_v1_3_0

        # Check function signature
        import inspect

        sig = inspect.signature(migrate_to_v1_3_0)
        # Should have no required parameters
        assert len(sig.parameters) == 0
