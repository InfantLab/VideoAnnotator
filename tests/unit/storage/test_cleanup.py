"""Tests for storage cleanup functionality.

Tests cover:
- Dry-run mode (no actual deletion)
- Safety checks (prevent deleting active jobs)
- Audit logging
- Retention calculation
- Terminal state verification
"""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from videoannotator.database.models import Job, JobStatus
from videoannotator.storage.cleanup import (
    TERMINAL_STATES,
    CleanupResult,
    cleanup_job_storage,
    cleanup_old_jobs,
    find_old_jobs,
    get_job_storage_size,
    is_cleanup_enabled,
    verify_job_safe_to_delete,
)
from videoannotator.storage.file_backend import FileStorageBackend


class TestCleanupResult:
    """Test CleanupResult tracking."""

    def test_initial_state(self):
        """Test initial result state."""
        result = CleanupResult()
        assert result.jobs_found == 0
        assert result.jobs_deleted == 0
        assert result.jobs_skipped == 0
        assert result.bytes_freed == 0
        assert result.errors == []
        assert result.deleted_jobs == []
        assert result.skipped_jobs == []

    def test_add_deleted(self):
        """Test recording deleted jobs."""
        result = CleanupResult()
        result.add_deleted("job-1", 1000)
        result.add_deleted("job-2", 2000)

        assert result.jobs_deleted == 2
        assert result.bytes_freed == 3000
        assert result.deleted_jobs == ["job-1", "job-2"]

    def test_add_skipped(self):
        """Test recording skipped jobs."""
        result = CleanupResult()
        result.add_skipped("job-1", "Not in terminal state")
        result.add_skipped("job-2", "No completion timestamp")

        assert result.jobs_skipped == 2
        assert len(result.skipped_jobs) == 2
        assert result.skipped_jobs[0]["job_id"] == "job-1"
        assert result.skipped_jobs[0]["reason"] == "Not in terminal state"

    def test_add_error(self):
        """Test recording errors."""
        result = CleanupResult()
        result.add_error("job-1", "Permission denied")

        assert len(result.errors) == 1
        assert "job-1" in result.errors[0]
        assert "Permission denied" in result.errors[0]

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = CleanupResult()
        result.jobs_found = 5
        result.add_deleted("job-1", 1024 * 1024)  # 1 MB
        result.add_skipped("job-2", "Test")
        result.add_error("job-3", "Test error")

        data = result.to_dict()

        assert data["jobs_found"] == 5
        assert data["jobs_deleted"] == 1
        assert data["jobs_skipped"] == 1
        assert data["bytes_freed"] == 1024 * 1024
        assert data["bytes_freed_mb"] == 1.0
        assert len(data["deleted_jobs"]) == 1
        assert len(data["skipped_jobs"]) == 1
        assert len(data["errors"]) == 1


class TestCleanupEnabled:
    """Test cleanup enabled check."""

    @patch("videoannotator.storage.cleanup.STORAGE_RETENTION_DAYS", None)
    def test_disabled_when_none(self):
        """Test cleanup disabled when retention days is None."""
        assert not is_cleanup_enabled()

    @patch("videoannotator.storage.cleanup.STORAGE_RETENTION_DAYS", 0)
    def test_disabled_when_zero(self):
        """Test cleanup disabled when retention days is 0."""
        assert not is_cleanup_enabled()

    @patch("videoannotator.storage.cleanup.STORAGE_RETENTION_DAYS", -1)
    def test_disabled_when_negative(self):
        """Test cleanup disabled when retention days is negative."""
        assert not is_cleanup_enabled()

    @patch("videoannotator.storage.cleanup.STORAGE_RETENTION_DAYS", 30)
    def test_enabled_when_positive(self):
        """Test cleanup enabled when retention days is positive."""
        assert is_cleanup_enabled()


class TestVerifyJobSafeToDelete:
    """Test job safety verification."""

    def test_job_in_terminal_state_is_safe(self):
        """Test job in terminal state passes checks."""
        job = MagicMock(spec=Job)
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now() - timedelta(days=1)

        is_safe, reason = verify_job_safe_to_delete(job)
        assert is_safe
        assert reason == ""

    def test_running_job_not_safe(self):
        """Test RUNNING job is not safe to delete."""
        job = MagicMock(spec=Job)
        job.status = JobStatus.RUNNING
        job.completed_at = None

        is_safe, reason = verify_job_safe_to_delete(job)
        assert not is_safe
        assert "not in terminal state" in reason.lower()
        assert "running" in reason.lower()

    def test_pending_job_not_safe(self):
        """Test PENDING job is not safe to delete."""
        job = MagicMock(spec=Job)
        job.status = JobStatus.PENDING
        job.completed_at = None

        is_safe, reason = verify_job_safe_to_delete(job)
        assert not is_safe
        assert "not in terminal state" in reason.lower()

    def test_job_without_completion_not_safe(self):
        """Test job without completion timestamp is not safe."""
        job = MagicMock(spec=Job)
        job.status = JobStatus.COMPLETED
        job.completed_at = None

        is_safe, reason = verify_job_safe_to_delete(job)
        assert not is_safe
        assert "no completion timestamp" in reason.lower()

    def test_job_with_future_completion_not_safe(self):
        """Test job with future completion timestamp is not safe."""
        job = MagicMock(spec=Job)
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now() + timedelta(days=1)

        is_safe, reason = verify_job_safe_to_delete(job)
        assert not is_safe
        assert "future" in reason.lower()

    @pytest.mark.parametrize("status", TERMINAL_STATES)
    def test_all_terminal_states_safe(self, status):
        """Test all terminal states are considered safe."""
        job = MagicMock(spec=Job)
        job.status = status
        job.completed_at = datetime.now() - timedelta(days=1)

        is_safe, reason = verify_job_safe_to_delete(job)
        assert is_safe
        assert reason == ""


class TestGetJobStorageSize:
    """Test storage size calculation."""

    def test_nonexistent_job_returns_zero(self, tmp_path):
        """Test nonexistent job directory returns 0 size."""
        storage = FileStorageBackend(base_dir=tmp_path)
        size = get_job_storage_size("nonexistent-job", storage)
        assert size == 0

    def test_empty_directory_returns_zero(self, tmp_path):
        """Test empty job directory returns 0 size."""
        storage = FileStorageBackend(base_dir=tmp_path)
        job_dir = storage._get_job_dir("test-job")
        job_dir.mkdir(parents=True)

        size = get_job_storage_size("test-job", storage)
        assert size == 0

    def test_calculates_total_size(self, tmp_path):
        """Test total size calculation for job with files."""
        storage = FileStorageBackend(base_dir=tmp_path)
        job_dir = storage._get_job_dir("test-job")
        job_dir.mkdir(parents=True)

        # Create test files
        (job_dir / "file1.txt").write_text("a" * 1000)
        (job_dir / "file2.txt").write_text("b" * 2000)
        subdir = job_dir / "subdir"
        subdir.mkdir()
        (subdir / "file3.txt").write_text("c" * 3000)

        size = get_job_storage_size("test-job", storage)
        assert size == 6000  # 1000 + 2000 + 3000


class TestCleanupJobStorage:
    """Test individual job cleanup."""

    def test_dry_run_does_not_delete(self, tmp_path):
        """Test dry-run mode doesn't actually delete files."""
        storage = FileStorageBackend(base_dir=tmp_path)
        job_dir = storage._get_job_dir("test-job")
        job_dir.mkdir(parents=True)
        test_file = job_dir / "test.txt"
        test_file.write_text("test content")

        job = MagicMock(spec=Job)
        job.id = "test-job"
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now() - timedelta(days=1)

        size, error = cleanup_job_storage(job, storage, dry_run=True)

        # File should still exist
        assert test_file.exists()
        assert size > 0
        assert error is None

    def test_force_deletes_files(self, tmp_path):
        """Test force mode actually deletes files."""
        storage = FileStorageBackend(base_dir=tmp_path)
        job_dir = storage._get_job_dir("test-job")
        job_dir.mkdir(parents=True)
        test_file = job_dir / "test.txt"
        test_file.write_text("test content")

        job = MagicMock(spec=Job)
        job.id = "test-job"
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now() - timedelta(days=1)

        size, error = cleanup_job_storage(job, storage, dry_run=False)

        # Directory should be deleted
        assert not job_dir.exists()
        assert not test_file.exists()
        assert size > 0
        assert error is None

    def test_unsafe_job_not_deleted(self, tmp_path):
        """Test unsafe job is not deleted even in force mode."""
        storage = FileStorageBackend(base_dir=tmp_path)
        job_dir = storage._get_job_dir("test-job")
        job_dir.mkdir(parents=True)

        # Job is RUNNING (not safe)
        job = MagicMock(spec=Job)
        job.id = "test-job"
        job.status = JobStatus.RUNNING
        job.completed_at = None

        size, error = cleanup_job_storage(job, storage, dry_run=False)

        # Directory should still exist
        assert job_dir.exists()
        assert size == 0
        assert error is not None
        assert "safety check failed" in error.lower()

    def test_nonexistent_directory_handled(self, tmp_path):
        """Test cleanup handles nonexistent directory gracefully."""
        storage = FileStorageBackend(base_dir=tmp_path)

        job = MagicMock(spec=Job)
        job.id = "test-job"
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now() - timedelta(days=1)

        size, error = cleanup_job_storage(job, storage, dry_run=False)

        assert size == 0
        assert error is not None
        assert "not found" in error.lower()


class TestFindOldJobs:
    """Test finding jobs eligible for cleanup."""

    @patch("videoannotator.storage.cleanup.STORAGE_RETENTION_DAYS", None)
    def test_raises_when_disabled(self):
        """Test raises ValueError when cleanup disabled."""
        with pytest.raises(ValueError, match="disabled"):
            find_old_jobs()

    @patch("videoannotator.storage.cleanup.STORAGE_RETENTION_DAYS", 0)
    def test_raises_when_zero(self):
        """Test raises ValueError when retention is 0."""
        with pytest.raises(ValueError, match="disabled"):
            find_old_jobs()

    def test_accepts_override(self):
        """Test accepts retention_days override parameter."""
        # Should not raise even if config is None
        with patch("videoannotator.storage.cleanup.STORAGE_RETENTION_DAYS", None):
            # This would query database, but we just test it doesn't raise
            try:
                find_old_jobs(retention_days=30)
            except Exception as e:
                # Database errors are okay for this test
                assert "disabled" not in str(e).lower()


class TestCleanupOldJobs:
    """Test full cleanup operation."""

    @patch("videoannotator.storage.cleanup.STORAGE_RETENTION_DAYS", None)
    def test_returns_error_when_disabled(self, tmp_path):
        """Test returns error result when cleanup disabled."""
        storage = FileStorageBackend(base_dir=tmp_path)
        result = cleanup_old_jobs(storage=storage)

        assert result.jobs_found == 0
        assert len(result.errors) > 0
        assert "disabled" in result.errors[0].lower()

    def test_accepts_override_parameter(self, tmp_path):
        """Test accepts retention_days override."""
        storage = FileStorageBackend(base_dir=tmp_path)

        # Mock find_old_jobs to return empty list
        with patch(
            "videoannotator.storage.cleanup.find_old_jobs", return_value=[]
        ):
            result = cleanup_old_jobs(
                retention_days=30, dry_run=True, storage=storage
            )

            assert result.jobs_found == 0
            assert len(result.errors) == 0

    def test_dry_run_default_behavior(self, tmp_path):
        """Test dry-run is the default behavior."""
        storage = FileStorageBackend(base_dir=tmp_path)

        # Create a mock job
        job_dir = storage._get_job_dir("test-job")
        job_dir.mkdir(parents=True)
        (job_dir / "test.txt").write_text("test")

        mock_job = MagicMock(spec=Job)
        mock_job.id = "test-job"
        mock_job.status = JobStatus.COMPLETED
        mock_job.completed_at = datetime.now() - timedelta(days=40)

        with patch(
            "videoannotator.storage.cleanup.find_old_jobs", return_value=[mock_job]
        ):
            # Don't pass dry_run parameter (should default to True)
            result = cleanup_old_jobs(retention_days=30, storage=storage)

            # File should still exist (dry-run)
            assert (job_dir / "test.txt").exists()
            assert result.jobs_found == 1
            assert result.jobs_skipped == 1  # Skipped in dry-run

    def test_force_mode_deletes(self, tmp_path):
        """Test force mode actually deletes files."""
        storage = FileStorageBackend(base_dir=tmp_path)

        # Create a mock job
        job_dir = storage._get_job_dir("test-job")
        job_dir.mkdir(parents=True)
        (job_dir / "test.txt").write_text("test")

        mock_job = MagicMock(spec=Job)
        mock_job.id = "test-job"
        mock_job.status = JobStatus.COMPLETED
        mock_job.completed_at = datetime.now() - timedelta(days=40)

        with patch(
            "videoannotator.storage.cleanup.find_old_jobs", return_value=[mock_job]
        ):
            result = cleanup_old_jobs(
                retention_days=30, dry_run=False, storage=storage
            )

            # Directory should be deleted
            assert not job_dir.exists()
            assert result.jobs_found == 1
            assert result.jobs_deleted == 1
            assert result.bytes_freed > 0

    def test_skips_unsafe_jobs(self, tmp_path):
        """Test cleanup skips jobs that fail safety checks."""
        storage = FileStorageBackend(base_dir=tmp_path)

        # Create unsafe job (RUNNING)
        mock_job = MagicMock(spec=Job)
        mock_job.id = "test-job"
        mock_job.status = JobStatus.RUNNING
        mock_job.completed_at = None

        with patch(
            "videoannotator.storage.cleanup.find_old_jobs", return_value=[mock_job]
        ):
            result = cleanup_old_jobs(
                retention_days=30, dry_run=False, storage=storage
            )

            assert result.jobs_found == 1
            assert result.jobs_deleted == 0
            assert result.jobs_skipped == 1
            assert len(result.errors) > 0
