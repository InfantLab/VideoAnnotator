"""Unit tests for database/crud.py using in-memory SQLite.

Tests all three CRUD classes: UserCRUD, APIKeyCRUD, JobCRUD.
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from videoannotator.database.crud import APIKeyCRUD, JobCRUD, UserCRUD
from videoannotator.database.database import Base
from videoannotator.database.models import JobStatus


@pytest.fixture
def db():
    """Create a fresh in-memory SQLite database for each test."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def sample_user(db):
    """Create a sample user and return it."""
    return UserCRUD.create(db, email="test@example.com", username="testuser")


# ---------------------------------------------------------------------------
# UserCRUD
# ---------------------------------------------------------------------------


class TestUserCRUD:
    def test_create_user(self, db):
        user = UserCRUD.create(db, email="a@b.com", username="alice")
        assert user.email == "a@b.com"
        assert user.username == "alice"
        assert user.id is not None

    def test_create_user_with_full_name(self, db):
        user = UserCRUD.create(
            db, email="b@b.com", username="bob", full_name="Bob Builder"
        )
        assert user.full_name == "Bob Builder"

    def test_get_by_id(self, db, sample_user):
        found = UserCRUD.get_by_id(db, str(sample_user.id))
        assert found is not None
        assert found.username == "testuser"

    def test_get_by_id_not_found(self, db):
        assert UserCRUD.get_by_id(db, "nonexistent-uuid") is None

    def test_get_by_email(self, db, sample_user):
        found = UserCRUD.get_by_email(db, "test@example.com")
        assert found is not None
        assert found.username == "testuser"

    def test_get_by_email_not_found(self, db):
        assert UserCRUD.get_by_email(db, "nobody@example.com") is None

    def test_get_by_username(self, db, sample_user):
        found = UserCRUD.get_by_username(db, "testuser")
        assert found is not None
        assert found.email == "test@example.com"

    def test_get_by_username_not_found(self, db):
        assert UserCRUD.get_by_username(db, "ghost") is None

    def test_update_user(self, db, sample_user):
        updated = UserCRUD.update(db, str(sample_user.id), full_name="Updated Name")
        assert updated is not None
        assert updated.full_name == "Updated Name"

    def test_update_nonexistent_user(self, db):
        result = UserCRUD.update(db, "no-such-id", full_name="X")
        assert result is None

    def test_update_ignores_unknown_fields(self, db, sample_user):
        updated = UserCRUD.update(db, str(sample_user.id), nonexistent_field="value")
        assert updated is not None  # should still succeed

    def test_delete_user(self, db, sample_user):
        assert UserCRUD.delete(db, str(sample_user.id)) is True
        assert UserCRUD.get_by_id(db, str(sample_user.id)) is None

    def test_delete_nonexistent_user(self, db):
        assert UserCRUD.delete(db, "no-such-id") is False


# ---------------------------------------------------------------------------
# APIKeyCRUD
# ---------------------------------------------------------------------------


class TestAPIKeyCRUD:
    def test_generate_api_key_format(self):
        raw, key_hash, prefix = APIKeyCRUD.generate_api_key()
        assert raw.startswith("va_")
        assert len(key_hash) == 64  # SHA256 hex digest
        assert prefix == raw[3:11]

    def test_hash_api_key_deterministic(self):
        h1 = APIKeyCRUD.hash_api_key("va_test123")
        h2 = APIKeyCRUD.hash_api_key("va_test123")
        assert h1 == h2

    def test_hash_api_key_different_inputs(self):
        h1 = APIKeyCRUD.hash_api_key("va_key1")
        h2 = APIKeyCRUD.hash_api_key("va_key2")
        assert h1 != h2

    def test_create_api_key(self, db, sample_user):
        api_key, raw = APIKeyCRUD.create(
            db, user_id=str(sample_user.id), key_name="Test Key"
        )
        assert api_key.key_name == "Test Key"
        assert api_key.user_id == sample_user.id
        assert api_key.is_active is True
        assert api_key.expires_at is None
        assert raw.startswith("va_")

    def test_create_api_key_with_expiry(self, db, sample_user):
        api_key, _ = APIKeyCRUD.create(
            db,
            user_id=str(sample_user.id),
            key_name="Expiring Key",
            expires_days=30,
        )
        assert api_key.expires_at is not None

    def test_get_by_token(self, db, sample_user):
        _, raw = APIKeyCRUD.create(
            db, user_id=str(sample_user.id), key_name="Lookup Key"
        )
        found = APIKeyCRUD.get_by_token(db, raw)
        assert found is not None
        assert found.key_name == "Lookup Key"

    def test_get_by_token_not_found(self, db):
        assert APIKeyCRUD.get_by_token(db, "va_nonexistent") is None

    def test_get_by_hash(self, db, sample_user):
        api_key, raw = APIKeyCRUD.create(
            db, user_id=str(sample_user.id), key_name="Hash Key"
        )
        key_hash = APIKeyCRUD.hash_api_key(raw)
        found = APIKeyCRUD.get_by_hash(db, key_hash)
        assert found is not None
        assert found.id == api_key.id

    def test_get_by_hash_inactive_not_found(self, db, sample_user):
        api_key, raw = APIKeyCRUD.create(
            db, user_id=str(sample_user.id), key_name="Inactive Key"
        )
        APIKeyCRUD.revoke(db, str(api_key.id))
        key_hash = APIKeyCRUD.hash_api_key(raw)
        found = APIKeyCRUD.get_by_hash(db, key_hash)
        assert found is None

    def test_get_by_prefix(self, db, sample_user):
        api_key, raw = APIKeyCRUD.create(
            db, user_id=str(sample_user.id), key_name="Prefix Key"
        )
        results = APIKeyCRUD.get_by_prefix(db, api_key.key_prefix)
        assert len(results) >= 1
        assert any(k.id == api_key.id for k in results)

    def test_authenticate_valid_key(self, db, sample_user):
        _, raw = APIKeyCRUD.create(db, user_id=str(sample_user.id), key_name="Auth Key")
        user = APIKeyCRUD.authenticate(db, raw)
        assert user is not None
        assert user.id == sample_user.id

    def test_authenticate_invalid_key(self, db):
        user = APIKeyCRUD.authenticate(db, "va_invalid")
        assert user is None

    def test_authenticate_expired_key(self, db, sample_user):
        api_key, raw = APIKeyCRUD.create(
            db,
            user_id=str(sample_user.id),
            key_name="Expired Auth",
            expires_days=1,
        )
        # Force expiry into the past
        api_key.expires_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=1)
        db.commit()

        user = APIKeyCRUD.authenticate(db, raw)
        assert user is None

    def test_authenticate_updates_last_used(self, db, sample_user):
        _, raw = APIKeyCRUD.create(
            db, user_id=str(sample_user.id), key_name="Last Used Key"
        )
        APIKeyCRUD.authenticate(db, raw)
        found = APIKeyCRUD.get_by_token(db, raw)
        assert found.last_used is not None

    def test_revoke(self, db, sample_user):
        api_key, _ = APIKeyCRUD.create(
            db, user_id=str(sample_user.id), key_name="Revoke Key"
        )
        assert APIKeyCRUD.revoke(db, str(api_key.id)) is True
        db.expire_all()
        assert api_key.is_active is False

    def test_revoke_nonexistent(self, db):
        assert APIKeyCRUD.revoke(db, "no-such-id") is False


# ---------------------------------------------------------------------------
# JobCRUD
# ---------------------------------------------------------------------------


class TestJobCRUD:
    def test_create_job(self, db, sample_user):
        job = JobCRUD.create(
            db,
            video_path="/videos/test.mp4",
            user_id=str(sample_user.id),
            video_filename="test.mp4",
        )
        assert job.video_path == "/videos/test.mp4"
        assert job.status == "pending"
        assert job.progress_percentage == 0

    def test_create_job_with_pipelines_and_config(self, db):
        job = JobCRUD.create(
            db,
            video_path="/videos/v.mp4",
            selected_pipelines=["face_analysis", "audio"],
            config={"threshold": 0.5},
            job_metadata={"source": "test"},
        )
        assert job.selected_pipelines == ["face_analysis", "audio"]
        assert job.config["threshold"] == 0.5
        assert job.job_metadata["source"] == "test"

    def test_create_job_without_user(self, db):
        job = JobCRUD.create(db, video_path="/videos/anon.mp4")
        assert job.user_id is None

    def test_get_by_id(self, db):
        job = JobCRUD.create(db, video_path="/videos/x.mp4")
        found = JobCRUD.get_by_id(db, str(job.id))
        assert found is not None
        assert found.video_path == "/videos/x.mp4"

    def test_get_by_id_not_found(self, db):
        assert JobCRUD.get_by_id(db, "nonexistent-id") is None

    def test_get_by_user(self, db, sample_user):
        JobCRUD.create(db, video_path="/v1.mp4", user_id=str(sample_user.id))
        JobCRUD.create(db, video_path="/v2.mp4", user_id=str(sample_user.id))
        jobs = JobCRUD.get_by_user(db, str(sample_user.id))
        assert len(jobs) == 2

    def test_get_by_user_with_limit(self, db, sample_user):
        for i in range(5):
            JobCRUD.create(db, video_path=f"/v{i}.mp4", user_id=str(sample_user.id))
        jobs = JobCRUD.get_by_user(db, str(sample_user.id), limit=3)
        assert len(jobs) == 3

    def test_get_all(self, db):
        JobCRUD.create(db, video_path="/a.mp4")
        JobCRUD.create(db, video_path="/b.mp4")
        jobs = JobCRUD.get_all(db)
        assert len(jobs) == 2

    def test_get_all_with_limit_and_offset(self, db):
        for i in range(5):
            JobCRUD.create(db, video_path=f"/v{i}.mp4")
        jobs = JobCRUD.get_all(db, limit=2, offset=1)
        assert len(jobs) == 2

    def test_get_by_status(self, db):
        j1 = JobCRUD.create(db, video_path="/a.mp4")
        JobCRUD.create(db, video_path="/b.mp4")
        JobCRUD.update_status(db, str(j1.id), JobStatus.RUNNING)

        pending = JobCRUD.get_by_status(db, JobStatus.PENDING)
        running = JobCRUD.get_by_status(db, JobStatus.RUNNING)
        assert len(pending) == 1
        assert len(running) == 1

    def test_get_active_jobs(self, db):
        j1 = JobCRUD.create(db, video_path="/a.mp4")
        j2 = JobCRUD.create(db, video_path="/b.mp4")
        JobCRUD.update_status(db, str(j2.id), JobStatus.COMPLETED)

        active = JobCRUD.get_active_jobs(db)
        assert len(active) == 1
        assert active[0].id == j1.id

    def test_update_status_to_running(self, db):
        job = JobCRUD.create(db, video_path="/a.mp4")
        updated = JobCRUD.update_status(db, str(job.id), JobStatus.RUNNING)
        assert updated.status == JobStatus.RUNNING
        assert updated.started_at is not None

    def test_update_status_to_completed(self, db):
        job = JobCRUD.create(db, video_path="/a.mp4")
        JobCRUD.update_status(db, str(job.id), JobStatus.RUNNING)
        updated = JobCRUD.update_status(db, str(job.id), JobStatus.COMPLETED)
        assert updated.status == JobStatus.COMPLETED
        assert updated.completed_at is not None

    def test_update_status_with_error(self, db):
        job = JobCRUD.create(db, video_path="/a.mp4")
        updated = JobCRUD.update_status(
            db,
            str(job.id),
            JobStatus.FAILED,
            error_message="Out of memory",
        )
        assert updated.error_message == "Out of memory"

    def test_update_status_with_progress(self, db):
        job = JobCRUD.create(db, video_path="/a.mp4")
        updated = JobCRUD.update_status(
            db, str(job.id), JobStatus.RUNNING, progress_percentage=50
        )
        assert updated.progress_percentage == 50

    def test_update_status_nonexistent(self, db):
        assert JobCRUD.update_status(db, "no-id", JobStatus.RUNNING) is None

    def test_update_results(self, db):
        job = JobCRUD.create(db, video_path="/a.mp4")
        updated = JobCRUD.update_results(
            db,
            str(job.id),
            result_path="/results/output.json",
            job_metadata={"frames": 100},
        )
        assert updated.result_path == "/results/output.json"
        assert updated.job_metadata["frames"] == 100

    def test_update_results_merges_metadata(self, db):
        job = JobCRUD.create(db, video_path="/a.mp4", job_metadata={"source": "test"})
        updated = JobCRUD.update_results(
            db,
            str(job.id),
            result_path="/results/out.json",
            job_metadata={"frames": 50},
        )
        assert updated.job_metadata["source"] == "test"
        assert updated.job_metadata["frames"] == 50

    def test_update_results_nonexistent(self, db):
        assert JobCRUD.update_results(db, "no-id", "/path") is None

    def test_delete_job(self, db):
        job = JobCRUD.create(db, video_path="/a.mp4")
        assert JobCRUD.delete(db, str(job.id)) is True
        assert JobCRUD.get_by_id(db, str(job.id)) is None

    def test_delete_nonexistent_job(self, db):
        assert JobCRUD.delete(db, "no-such-id") is False

    def test_cleanup_old_jobs(self, db):
        old_job = JobCRUD.create(db, video_path="/old.mp4")
        JobCRUD.update_status(db, str(old_job.id), JobStatus.COMPLETED)
        # Force the created_at to be old
        old_job.created_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=60)
        db.commit()

        recent_job = JobCRUD.create(db, video_path="/recent.mp4")
        JobCRUD.update_status(db, str(recent_job.id), JobStatus.COMPLETED)

        deleted = JobCRUD.cleanup_old_jobs(db, days_old=30)
        assert deleted == 1
        assert JobCRUD.get_by_id(db, str(old_job.id)) is None
        assert JobCRUD.get_by_id(db, str(recent_job.id)) is not None

    def test_cleanup_skips_active_jobs(self, db):
        job = JobCRUD.create(db, video_path="/active.mp4")
        job.created_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=60)
        db.commit()

        deleted = JobCRUD.cleanup_old_jobs(db, days_old=30)
        assert deleted == 0  # still pending, not in FINAL_STATUSES

    def test_cleanup_no_old_jobs(self, db):
        JobCRUD.create(db, video_path="/new.mp4")
        deleted = JobCRUD.cleanup_old_jobs(db, days_old=30)
        assert deleted == 0
