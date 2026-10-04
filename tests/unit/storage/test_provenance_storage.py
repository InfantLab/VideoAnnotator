"""Jobs keep each pipeline's provenance record (spec 017 US4)."""

import sqlite3
from pathlib import Path

import pytest

from videoannotator.batch.types import BatchJob, JobStatus, PipelineResult
from videoannotator.storage.file_backend import FileStorageBackend
from videoannotator.storage.sqlite_backend import SQLiteStorageBackend

RECORD = {"schema_version": 1, "pipeline": {"name": "scene_detection"}, "models": []}


def _job(provenance):
    job = BatchJob(video_path=Path("clip.mp4"), status=JobStatus.COMPLETED)
    job.pipeline_results = {
        "scene_detection": PipelineResult(
            "scene_detection", JobStatus.COMPLETED, provenance=provenance
        )
    }
    return job


@pytest.fixture
def sqlite_backend(tmp_path):
    backend = SQLiteStorageBackend(tmp_path / "va.db")
    yield backend
    backend.close()


@pytest.mark.parametrize("provenance", [RECORD, None])
def test_sqlite_round_trip(sqlite_backend, provenance):
    job = _job(provenance)
    sqlite_backend.save_job_metadata(job)
    sqlite_backend.save_job_metadata(job)  # the update path too
    loaded = sqlite_backend.load_job_metadata(job.job_id)
    assert loaded.pipeline_results["scene_detection"].provenance == provenance


def test_file_backend_round_trip_and_old_metadata(tmp_path):
    backend = FileStorageBackend(tmp_path)
    job = _job(RECORD)
    backend.save_job_metadata(job)
    assert (
        backend.load_job_metadata(job.job_id)
        .pipeline_results["scene_detection"]
        .provenance
        == RECORD
    )
    data = job.to_dict()
    del data["pipeline_results"]["scene_detection"]["provenance"]  # pre-017 file
    assert (
        BatchJob.from_dict(data).pipeline_results["scene_detection"].provenance is None
    )


def test_an_older_database_gains_the_column(tmp_path):
    path = tmp_path / "old.db"
    SQLiteStorageBackend(path).close()
    conn = sqlite3.connect(path)
    conn.execute("ALTER TABLE pipeline_results DROP COLUMN provenance")
    conn.commit()
    conn.close()

    backend = SQLiteStorageBackend(path)
    try:
        job = _job(RECORD)
        backend.save_job_metadata(job)
        loaded = backend.load_job_metadata(job.job_id)
        assert loaded.pipeline_results["scene_detection"].provenance == RECORD
    finally:
        backend.close()
