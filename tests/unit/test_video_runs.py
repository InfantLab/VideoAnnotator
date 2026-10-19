"""Every run on one video, its files and projects (spec 028 R2)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from videoannotator.batch.types import BatchJob, JobStatus, PipelineResult
from videoannotator.video_runs import projects_for, runs_for, sha256_for

T0 = datetime(2026, 10, 12, 9, 0)


class FakeStorage:
    def __init__(self, *jobs: BatchJob):
        self.jobs = {j.job_id: j for j in jobs}

    def list_jobs(self):
        return list(self.jobs)

    def load_job_metadata(self, job_id):
        return self.jobs[job_id]


class FakeDb:
    def __init__(self, *datasets):
        self.datasets = datasets

    def query(self, _model):
        return SimpleNamespace(all=lambda: list(self.datasets))


def _job(tmp_path: Path, video: Path, minutes: int, **results) -> BatchJob:
    folder = tmp_path / f"out_{minutes}"
    folder.mkdir()
    return BatchJob(
        video_path=video,
        output_dir=folder,
        status=JobStatus.COMPLETED,
        created_at=T0 + timedelta(minutes=minutes),
        pipeline_results=results,
    )


def _video(tmp_path: Path, name: str = "p01_t1.mp4") -> Path:
    video = tmp_path / name
    video.write_bytes(b"video")
    return video.resolve()


def test_jobs_through_a_symlink_are_one_list_newest_first(tmp_path):
    video = _video(tmp_path)
    link = tmp_path / "link.mp4"
    link.symlink_to(video)
    older = _job(tmp_path, video, 0)
    newer = _job(tmp_path, link, 10)
    other = _job(tmp_path, _video(tmp_path, "other.mp4"), 20)

    runs = runs_for(FakeStorage(older, other, newer), video)

    assert [r.job_id for r in runs] == [newer.job_id, older.job_id]


def test_failed_pipeline_carries_its_error(tmp_path):
    video = _video(tmp_path)
    job = _job(
        tmp_path,
        video,
        0,
        speech_recognition=PipelineResult(
            "speech_recognition", JobStatus.FAILED, error_message="No audio"
        ),
    )

    [run] = runs_for(FakeStorage(job), video)

    [pipeline] = run.pipelines
    assert (pipeline.name, pipeline.status, pipeline.error) == (
        "speech_recognition",
        "failed",
        "No audio",
    )
    assert run.run_name is None


def test_files_are_listed_with_their_kind(tmp_path):
    video = _video(tmp_path)
    job = _job(
        tmp_path,
        video,
        0,
        speaker_diarization=PipelineResult("speaker_diarization", JobStatus.COMPLETED),
    )
    folder = Path(job.output_dir)
    (folder / "p01_t1_speaker_diarization.rttm").write_text("SPEAKER x\n")
    (folder / "p01_t1_speaker_diarization.rttm.provenance.json").write_text("{}")

    [run] = runs_for(FakeStorage(job), video)

    files = {f.name: f.to_dict() for f in run.pipelines[0].files}
    assert files["p01_t1_speaker_diarization.rttm"]["kind"] == "result"
    assert files["p01_t1_speaker_diarization.rttm.provenance.json"]["kind"] == (
        "provenance"
    )
    assert files["p01_t1_speaker_diarization.rttm"]["download_url"] == (
        f"/api/v1/jobs/{job.job_id}/results/files/speaker_diarization"
        "?name=p01_t1_speaker_diarization.rttm"
    )


def test_run_name_is_the_run_folder(tmp_path):
    video = _video(tmp_path)
    run_folder = tmp_path / "Peekaboo study (2026-10-12)"
    (run_folder / "p01_t1").mkdir(parents=True)
    (run_folder / "run.json").write_text("{}")
    job = BatchJob(video_path=video, output_dir=run_folder / "p01_t1")

    [run] = runs_for(FakeStorage(job), video)

    assert run.run_name == "Peekaboo study (2026-10-12)"


def _dataset(name, *, id="d1", folder=None, recursive=False, selection=False, **kw):
    return SimpleNamespace(
        id=id,
        name=name,
        server_folder=str(folder) if folder else None,
        server_folder_recursive=recursive,
        server_selection=selection,
        video_manifest=kw.get("manifest", []),
    )


def test_projects_by_dataset_id_and_by_folder(tmp_path):
    video = _video(tmp_path)
    job = _job(tmp_path, video, 0)
    job.dataset_id = "by-run"
    runs = runs_for(FakeStorage(job), video)
    db = FakeDb(
        _dataset("Peekaboo study", id="by-run"),
        _dataset("Folder", id="by-folder", folder=tmp_path),
        _dataset("Parent folder, not recursive", id="no", folder=tmp_path.parent),
        _dataset(
            "Parent folder, recursive", id="rec", folder=tmp_path.parent, recursive=True
        ),
        _dataset("Unrelated", id="x"),
    )

    found = projects_for(db, video, runs)

    assert {p["id"] for p in found} == {"by-run", "by-folder", "rec"}


def test_project_choosing_from_a_folder_needs_the_video_in_its_manifest(tmp_path):
    video = _video(tmp_path)
    db = FakeDb(
        _dataset(
            "Chose it",
            id="yes",
            folder=tmp_path,
            selection=True,
            manifest=[{"filename": "p01_t1.mp4", "size_bytes": 5}],
        ),
        _dataset(
            "Chose another",
            id="no",
            folder=tmp_path,
            selection=True,
            manifest=[{"filename": "p02.mp4", "size_bytes": 5}],
        ),
    )

    assert [p["id"] for p in projects_for(db, video, [])] == ["yes"]


def test_sha256_from_a_provenance_file(tmp_path):
    video = _video(tmp_path)
    job = _job(
        tmp_path,
        video,
        0,
        speaker_diarization=PipelineResult("speaker_diarization", JobStatus.COMPLETED),
    )
    folder = Path(job.output_dir)
    fixtures = Path(__file__).parent.parent / "fixtures" / "viewer_contract"
    companion = fixtures / "demo_clip_speaker_diarization.rttm.provenance.json"
    (folder / "p01_t1_speaker_diarization.rttm").write_text("SPEAKER x\n")
    (folder / "p01_t1_speaker_diarization.rttm.provenance.json").write_text(
        companion.read_text()
    )

    runs = runs_for(FakeStorage(job), video)

    assert sha256_for(runs) == json.loads(companion.read_text())["input"]["sha256"]


def test_sha256_from_the_stored_provenance_first(tmp_path):
    video = _video(tmp_path)
    job = _job(
        tmp_path,
        video,
        0,
        scene_detection=PipelineResult(
            "scene_detection",
            JobStatus.COMPLETED,
            provenance={"input": {"sha256": "abc"}},
        ),
    )

    assert sha256_for(runs_for(FakeStorage(job), video)) == "abc"
    assert sha256_for([]) is None
