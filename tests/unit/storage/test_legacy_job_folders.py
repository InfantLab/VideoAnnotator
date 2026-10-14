"""Jobs made before v1.6.0 live under ./storage/jobs; they must still open
after the default storage root moved to the per-user data directory."""

from pathlib import Path

from videoannotator.storage import manager
from videoannotator.storage.config import legacy_storage_root
from videoannotator.storage.providers.local import LocalStorageProvider


def test_finds_old_job_folders_under_the_start_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "new"))
    assert legacy_storage_root() is None
    (tmp_path / "storage" / "jobs" / "job-1").mkdir(parents=True)
    assert legacy_storage_root() == (tmp_path / "storage" / "jobs").resolve()
    # Not "old" when it is the storage root itself.
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage" / "jobs"))
    assert legacy_storage_root() is None


def test_an_old_job_is_read_from_the_folder_it_recorded(tmp_path, monkeypatch):
    current = LocalStorageProvider(root_path=tmp_path / "new")
    monkeypatch.setattr(manager, "get_storage_provider", lambda: current)
    old = tmp_path / "storage" / "jobs" / "job-1"
    old.mkdir(parents=True)
    (old / "clip.webm").write_bytes(b"v")

    provider = manager.provider_for_job("job-1", str(old))
    assert [a.name for a in provider.list_files("job-1")] == ["clip.webm"]
    # A job in today's root, or with nothing recorded, uses the configured provider.
    assert manager.provider_for_job("job-2", None) is current
    assert manager.provider_for_job("job-1", str(tmp_path / "elsewhere")) is current


def test_deleting_an_old_job_removes_the_folder_it_recorded(tmp_path, monkeypatch):
    from videoannotator.batch.types import BatchJob
    from videoannotator.storage.sqlite_backend import SQLiteStorageBackend

    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "new"))
    backend = SQLiteStorageBackend(tmp_path / "va.db")
    try:
        old = tmp_path / "storage" / "jobs" / "job-1"
        old.mkdir(parents=True)
        job = BatchJob(job_id="job-1", video_path=old / "clip.webm")
        job.storage_path = old
        backend.save_job_metadata(job)

        backend.delete_job("job-1")
        assert not old.exists()
        assert Path(tmp_path / "storage" / "jobs").exists()
    finally:
        backend.close()
