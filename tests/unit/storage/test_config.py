"""Unit tests for storage configuration and path management.

Tests the storage path utilities that manage persistent job storage directories.
"""

import os
import tempfile
from pathlib import Path

from videoannotator.storage.config import (
    get_job_storage_path,
    get_storage_root,
)


class TestGetStorageRoot:
    """Test get_storage_root() function."""

    def test_default_storage_root(self, monkeypatch):
        """Test default storage root when STORAGE_ROOT not set."""
        monkeypatch.delenv("STORAGE_ROOT", raising=False)
        root = get_storage_root()

        assert isinstance(root, Path)
        assert root.is_absolute()
        # Should resolve to absolute path based on ./storage/jobs
        assert root.name == "jobs" or "storage" in str(root)

    def test_custom_storage_root_from_env(self, monkeypatch):
        """Test custom storage root from environment variable."""
        custom_path = "/tmp/custom_storage"
        monkeypatch.setenv("STORAGE_ROOT", custom_path)

        root = get_storage_root()

        assert isinstance(root, Path)
        assert root.is_absolute()
        assert str(root) == custom_path

    def test_storage_root_resolves_relative_paths(self, monkeypatch):
        """Test that relative paths are resolved to absolute."""
        monkeypatch.setenv("STORAGE_ROOT", "../test_storage")

        root = get_storage_root()

        assert root.is_absolute()
        assert "test_storage" in str(root)

    def test_storage_root_with_tilde_expansion(self, monkeypatch):
        """Test that tilde in path is expanded."""
        monkeypatch.setenv("STORAGE_ROOT", "~/videoannotator_storage")

        root = get_storage_root()

        assert root.is_absolute()
        # Tilde should be expanded by Path.resolve()
        assert "~" not in str(root)


class TestGetJobStoragePath:
    """Test get_job_storage_path() function."""

    def test_job_storage_path_structure(self, monkeypatch):
        """Test that job path is under storage root."""
        storage_root = "/tmp/test_storage"
        monkeypatch.setenv("STORAGE_ROOT", storage_root)

        job_id = "test-job-123"
        path = get_job_storage_path(job_id)

        assert isinstance(path, Path)
        assert path.is_absolute()
        assert path.parent == Path(storage_root)
        assert path.name == job_id

    def test_job_storage_path_consistency(self):
        """Test that same job_id always returns same path."""
        job_id = "consistent-job"

        path1 = get_job_storage_path(job_id)
        path2 = get_job_storage_path(job_id)

        assert path1 == path2

    def test_different_jobs_different_paths(self):
        """Test that different job IDs get different paths."""
        path1 = get_job_storage_path("job-1")
        path2 = get_job_storage_path("job-2")

        assert path1 != path2
        assert path1.parent == path2.parent  # Same root
        assert path1.name == "job-1"
        assert path2.name == "job-2"

    def test_job_storage_path_with_special_characters(self):
        """Test job IDs with special characters."""
        # UUID-like job IDs with hyphens
        job_id = "abc-123-def-456"
        path = get_job_storage_path(job_id)

        assert path.name == job_id
        assert path.is_absolute()

    def test_job_storage_path_does_not_create_directory(self):
        """Test that get_job_storage_path doesn't create the directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            job_id = "uncreated-job"
            # Use custom storage root

            original_env = os.environ.get("STORAGE_ROOT")
            try:
                os.environ["STORAGE_ROOT"] = tmpdir
                # Force reload to pick up new env var
                from importlib import reload

                import videoannotator.storage.config

                reload(videoannotator.storage.config)
                from videoannotator.storage.config import get_job_storage_path as gjsp

                path = gjsp(job_id)

                # Path should not exist yet
                assert not path.exists()
            finally:
                if original_env:
                    os.environ["STORAGE_ROOT"] = original_env
                else:
                    os.environ.pop("STORAGE_ROOT", None)


class TestEnsureJobStoragePath:
    """Test ensure_job_storage_path() function."""

    def test_ensure_creates_directory(self):
        """Test that ensure_job_storage_path creates the directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set up clean environment
            original_env = os.environ.get("STORAGE_ROOT")
            try:
                os.environ["STORAGE_ROOT"] = tmpdir

                # Force reload
                from importlib import reload

                import videoannotator.storage.config

                reload(videoannotator.storage.config)
                from videoannotator.storage.config import (
                    ensure_job_storage_path as ejsp,
                )

                job_id = "new-job-dir"
                path = ejsp(job_id)

                # Directory should now exist
                assert path.exists()
                assert path.is_dir()
                assert path.parent == Path(tmpdir)
                assert path.name == job_id
            finally:
                if original_env:
                    os.environ["STORAGE_ROOT"] = original_env
                else:
                    os.environ.pop("STORAGE_ROOT", None)

    def test_ensure_idempotent(self):
        """Test that ensure_job_storage_path is idempotent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_env = os.environ.get("STORAGE_ROOT")
            try:
                os.environ["STORAGE_ROOT"] = tmpdir

                from importlib import reload

                import videoannotator.storage.config

                reload(videoannotator.storage.config)
                from videoannotator.storage.config import (
                    ensure_job_storage_path as ejsp,
                )

                job_id = "idempotent-job"

                # Call multiple times
                path1 = ejsp(job_id)
                path2 = ejsp(job_id)
                path3 = ejsp(job_id)

                # All should succeed and return same path
                assert path1 == path2 == path3
                assert path1.exists()
            finally:
                if original_env:
                    os.environ["STORAGE_ROOT"] = original_env
                else:
                    os.environ.pop("STORAGE_ROOT", None)

    def test_ensure_creates_parent_directories(self):
        """Test that ensure_job_storage_path creates parent dirs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_env = os.environ.get("STORAGE_ROOT")
            try:
                # Use a nested path that doesn't exist yet
                nested_storage = os.path.join(tmpdir, "nested", "storage", "jobs")
                os.environ["STORAGE_ROOT"] = nested_storage

                from importlib import reload

                import videoannotator.storage.config

                reload(videoannotator.storage.config)
                from videoannotator.storage.config import (
                    ensure_job_storage_path as ejsp,
                )

                job_id = "nested-job"
                path = ejsp(job_id)

                # All parent directories should be created
                assert path.exists()
                assert path.is_dir()
                assert Path(nested_storage).exists()
            finally:
                if original_env:
                    os.environ["STORAGE_ROOT"] = original_env
                else:
                    os.environ.pop("STORAGE_ROOT", None)

    def test_ensure_with_existing_directory(self):
        """Test ensure_job_storage_path with pre-existing directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_env = os.environ.get("STORAGE_ROOT")
            try:
                os.environ["STORAGE_ROOT"] = tmpdir

                # Pre-create the directory
                job_id = "existing-dir"
                existing_path = Path(tmpdir) / job_id
                existing_path.mkdir(parents=True, exist_ok=True)

                # Create a file in it to verify it's not deleted
                test_file = existing_path / "test.txt"
                test_file.write_text("test content")

                from importlib import reload

                import videoannotator.storage.config

                reload(videoannotator.storage.config)
                from videoannotator.storage.config import (
                    ensure_job_storage_path as ejsp,
                )

                # Now call ensure
                path = ejsp(job_id)

                # Should succeed and not delete existing content
                assert path.exists()
                assert test_file.exists()
                assert test_file.read_text() == "test content"
            finally:
                if original_env:
                    os.environ["STORAGE_ROOT"] = original_env
                else:
                    os.environ.pop("STORAGE_ROOT", None)


class TestStoragePathIntegration:
    """Integration tests for storage path utilities."""

    def test_full_workflow(self):
        """Test complete workflow: get root, get path, ensure path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_env = os.environ.get("STORAGE_ROOT")
            try:
                os.environ["STORAGE_ROOT"] = tmpdir

                from importlib import reload

                import videoannotator.storage.config

                reload(videoannotator.storage.config)
                from videoannotator.storage.config import (
                    ensure_job_storage_path as ejsp,
                )
                from videoannotator.storage.config import (
                    get_job_storage_path as gjsp,
                )
                from videoannotator.storage.config import (
                    get_storage_root as gsr,
                )

                # Get root
                root = gsr()
                assert root == Path(tmpdir)

                # Get job path (doesn't create)
                job_id = "workflow-job"
                job_path = gjsp(job_id)
                assert not job_path.exists()

                # Ensure job path (creates)
                ensured_path = ejsp(job_id)
                assert ensured_path.exists()
                assert ensured_path == job_path

                # Verify structure
                assert ensured_path.parent == root
                assert ensured_path.name == job_id
            finally:
                if original_env:
                    os.environ["STORAGE_ROOT"] = original_env
                else:
                    os.environ.pop("STORAGE_ROOT", None)

    def test_multiple_jobs_isolation(self):
        """Test that multiple jobs get isolated directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_env = os.environ.get("STORAGE_ROOT")
            try:
                os.environ["STORAGE_ROOT"] = tmpdir

                from importlib import reload

                import videoannotator.storage.config

                reload(videoannotator.storage.config)
                from videoannotator.storage.config import (
                    ensure_job_storage_path as ejsp,
                )

                # Create multiple job directories
                job_ids = ["job-1", "job-2", "job-3"]
                paths = [ejsp(job_id) for job_id in job_ids]

                # All should exist
                assert all(p.exists() for p in paths)

                # All should be different
                assert len(set(paths)) == len(paths)

                # All should be under same root
                roots = {p.parent for p in paths}
                assert len(roots) == 1
            finally:
                if original_env:
                    os.environ["STORAGE_ROOT"] = original_env
                else:
                    os.environ.pop("STORAGE_ROOT", None)
