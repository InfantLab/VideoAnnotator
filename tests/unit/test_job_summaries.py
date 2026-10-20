"""job_summaries(): one read of every job for spec 029's video list, the same
from the SQLite backend's query as from the default per-job load."""

from __future__ import annotations

from tests.fixtures.video_index import JobsDb
from videoannotator.storage.base import PipelineSummary, StorageBackend
from videoannotator.storage.file_backend import FileStorageBackend


def _fill(jobs: JobsDb) -> None:
    first, _ = jobs.upload(
        "clip.mp4",
        batch_id="b1",
        batch_name="Peekaboo",
        dataset_id="d1",
        pipelines={
            "face_analysis": ("completed", None, "abc123"),
            "speech_recognition": ("failed", "No audio", None),
        },
    )
    jobs.rerun(first, pipelines={"face_analysis": ("running", None, None)})
    jobs.make_job(jobs.tmp_path / "in_place.mp4", pipelines={})


def _comparable(storage: StorageBackend) -> list[tuple]:
    return [
        (
            s.job_id,
            s.video_path,
            s.status,
            s.created_at,
            s.batch_id,
            s.batch_name,
            s.dataset_id,
            s.rerun_of,
            s.storage_path,
            s.selected_pipelines,
            dict(s.pipelines),
        )
        for s in storage.job_summaries()
    ]


def test_sqlite_reads_what_a_job_records(tmp_path):
    jobs = JobsDb(tmp_path)
    _fill(jobs)

    first, rerun, in_place = jobs.storage.job_summaries()

    assert (first.batch_id, first.batch_name, first.dataset_id) == (
        "b1",
        "Peekaboo",
        "d1",
    )
    assert first.pipelines == {
        "face_analysis": PipelineSummary("completed", None, "abc123"),
        "speech_recognition": PipelineSummary("failed", "No audio", None),
    }
    assert rerun.rerun_of == first.job_id
    assert rerun.status == "running"
    assert in_place.pipelines == {}
    assert in_place.status == "pending"


def test_both_backends_return_the_same(tmp_path):
    sqlite = JobsDb(tmp_path / "a")
    files = JobsDb(tmp_path / "b", storage=FileStorageBackend(tmp_path / "b" / "fs"))
    _fill(sqlite)
    _fill(files)

    def without_ids(rows):
        return [(r[2], r[4], r[5], r[6], r[9], r[10]) for r in rows]

    assert without_ids(_comparable(sqlite.storage)) == without_ids(
        _comparable(files.storage)
    )


def test_the_signature_changes_when_a_job_moves_on(tmp_path):
    jobs = JobsDb(tmp_path)
    job = jobs.make_job(tmp_path / "v.mp4")
    before = jobs.storage.jobs_signature()
    assert jobs.storage.jobs_signature() == before

    from videoannotator.batch.types import JobStatus

    job.status = JobStatus.RUNNING
    jobs.storage.save_job_metadata(job)

    assert jobs.storage.jobs_signature() != before
