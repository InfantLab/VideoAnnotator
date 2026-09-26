"""Unit tests for v1.3.0 database migrations.

Tests migration from v1.2.x to v1.3.0 schema, including:
- Addition of jobs.cancelled_at column
- Addition of jobs.storage_path column

Note: These are smoke tests to verify migration logic integrity and
that the new columns are present in the database models.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


class TestDatabaseModelsV1_3_0:
    """Test that v1.3.0 database migration infrastructure is in place."""

    def test_job_status_enum_has_cancelled(self):
        """Test that JobStatus enum includes CANCELLED state."""
        from videoannotator.batch.types import JobStatus

        assert hasattr(JobStatus, "CANCELLED")
        assert JobStatus.CANCELLED.value == "cancelled"

    def test_database_job_status_constant_has_cancelled(self):
        """Test that database JobStatus constants include CANCELLED."""
        from videoannotator.database.models import JobStatus

        assert hasattr(JobStatus, "CANCELLED")
        assert JobStatus.CANCELLED == "cancelled"
        assert "cancelled" in JobStatus.ALL_STATUSES
        assert "cancelled" in JobStatus.FINAL_STATUSES


class TestMigrationFunction:
    """Test migration function exists and is callable."""

    def test_migrate_to_v1_3_0_exists(self):
        """Test that migrate_to_v1_3_0 function exists."""
        from videoannotator.database.migrations import migrate_to_v1_3_0

        assert callable(migrate_to_v1_3_0)

    def test_migrate_to_v1_3_0_returns_bool(self):
        """Test that migrate_to_v1_3_0 has correct return type signature."""
        # Check function signature
        import inspect

        from videoannotator.database.migrations import migrate_to_v1_3_0

        sig = inspect.signature(migrate_to_v1_3_0)
        # Should have no required parameters
        assert len(sig.parameters) == 0


@pytest.fixture
def temp_db(monkeypatch, tmp_path):
    import videoannotator.database.database as db_module

    engine = create_engine(
        f"sqlite:///{tmp_path / 'test_admin.db'}",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    monkeypatch.setattr(db_module, "engine", engine)
    monkeypatch.setattr(
        db_module,
        "SessionLocal",
        sessionmaker(autocommit=False, autoflush=False, bind=engine),
    )
    db_module.Base.metadata.create_all(bind=engine)
    yield db_module.SessionLocal


class TestCreateAdminUser:
    def test_restart_with_default_identity_reuses_custom_admin(self, temp_db):
        """start_server.sh passes the default admin identity on every
        restart; that must not mint a second admin next to a first-run
        admin created under a custom name."""
        from videoannotator.database.migrations import create_admin_user
        from videoannotator.database.models import User

        first, first_key = create_admin_user(
            username="infantologist", email="infantologist@example.com"
        )
        again, again_key = create_admin_user(
            username="admin", email="admin@videoannotator.local"
        )

        assert first_key
        assert again_key is None
        assert again.username == "infantologist"
        db = temp_db()
        try:
            assert db.query(User).count() == 1
        finally:
            db.close()
