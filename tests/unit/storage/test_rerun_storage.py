"""Jobs remember which job they run again (spec 019)."""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from videoannotator.batch.types import BatchJob
from videoannotator.storage.file_backend import FileStorageBackend
from videoannotator.storage.sqlite_backend import SQLiteStorageBackend


def _jobs():
    original = BatchJob(video_path=Path("a.mp4"))
    first = BatchJob(video_path=Path("a.mp4"), rerun_of=original.job_id)
    second = BatchJob(video_path=Path("a.mp4"), rerun_of=original.job_id)
    second.created_at = first.created_at + timedelta(seconds=1)
    return original, first, second


@pytest.fixture(params=["sqlite", "file"])
def backend(request, tmp_path):
    if request.param == "sqlite":
        b = SQLiteStorageBackend(tmp_path / "va.db")
        yield b
        b.close()
    else:
        yield FileStorageBackend(tmp_path)


def test_reruns_are_listed_oldest_first(backend):
    original, first, second = _jobs()
    for job in (second, original, first):
        backend.save_job_metadata(job)
    assert backend.list_reruns(original.job_id) == [first.job_id, second.job_id]
    assert backend.load_job_metadata(first.job_id).rerun_of == original.job_id
    assert backend.list_reruns(first.job_id) == []


def test_an_older_database_gains_the_column(tmp_path):
    path = tmp_path / "old.db"
    SQLiteStorageBackend(path).close()
    conn = sqlite3.connect(path)
    conn.execute("DROP INDEX IF EXISTS ix_jobs_rerun_of")
    conn.execute("ALTER TABLE jobs DROP COLUMN rerun_of")
    conn.commit()
    conn.close()

    backend = SQLiteStorageBackend(path)
    try:
        original, first, _ = _jobs()
        backend.save_job_metadata(original)
        backend.save_job_metadata(first)
        assert backend.list_reruns(original.job_id) == [first.job_id]
    finally:
        backend.close()


def test_job_metadata_from_before_spec_019_has_no_link():
    data = BatchJob(video_path=Path("a.mp4"), created_at=datetime.now()).to_dict()
    del data["rerun_of"]
    assert BatchJob.from_dict(data).rerun_of is None
