"""Jobs can be listed by the saved dataset they were run from (spec 025)."""

from datetime import timedelta
from pathlib import Path

import pytest

from videoannotator.batch.types import BatchJob
from videoannotator.storage.file_backend import FileStorageBackend
from videoannotator.storage.sqlite_backend import SQLiteStorageBackend


@pytest.fixture(params=["sqlite", "file"])
def backend(request, tmp_path):
    if request.param == "sqlite":
        b = SQLiteStorageBackend(tmp_path / "va.db")
        yield b
        b.close()
    else:
        yield FileStorageBackend(tmp_path)


def test_jobs_of_a_dataset_are_listed_oldest_first(backend):
    first = BatchJob(video_path=Path("a.mp4"), dataset_id="ds-1")
    second = BatchJob(video_path=Path("b.mp4"), dataset_id="ds-1")
    second.created_at = first.created_at + timedelta(seconds=1)
    other = BatchJob(video_path=Path("c.mp4"), dataset_id="ds-2")
    loose = BatchJob(video_path=Path("d.mp4"))
    for job in (second, other, loose, first):
        backend.save_job_metadata(job)

    assert backend.list_jobs_by_dataset("ds-1") == [first.job_id, second.job_id]
    assert backend.list_jobs_by_dataset("ds-2") == [other.job_id]
    assert backend.list_jobs_by_dataset("missing") == []
