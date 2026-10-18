"""Which files an export reads, by run and video (spec 025)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from videoannotator.exporters.results_set import from_folder, from_jobs

from .conftest import FIXTURES, LEGACY, RUN_NAME, copy_outputs, make_run

KINDS = ["person", "face_deepface", "face_openface3", "scene", "vlm", "rttm", "webvtt"]


def test_a_run_folder_gives_one_video_per_folder(run_folder):
    results = from_folder(run_folder)
    assert [(v.run, v.video) for v in results.videos] == [(RUN_NAME, "demo_clip")]
    video = results.videos[0]
    assert [f.kind.kind for f in video.files] == KINDS
    assert video.media_path == "/data/videos/demo_clip.mp4"
    assert results.missing == [] and results.skipped == []


def test_provenance_is_read_from_every_format(run_folder):
    files = {f.kind.kind: f for f in from_folder(run_folder).videos[0].files}
    assert files["rttm"].provenance["pipeline"]["name"] == "speaker_diarization"
    assert files["webvtt"].provenance["models"][0]["name"] == (
        "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    )
    assert files["person"].pipeline == "person_tracking"
    assert files["face_deepface"].provenance["pipeline"]["name"] == "face_analysis"
    assert files["face_deepface"].pipeline == "face_analysis"


def test_one_video_folder_inside_a_run(run_folder):
    results = from_folder(run_folder / "demo_clip")
    assert [(v.run, v.video) for v in results.videos] == [(RUN_NAME, "demo_clip")]
    assert results.videos[0].media_path == "/data/videos/demo_clip.mp4"


def test_a_folder_of_runs_is_walked(tmp_path):
    make_run(tmp_path / "all", {"a": FIXTURES}, name="One (2026-10-01)")
    make_run(tmp_path / "all", {"b": LEGACY}, name="Two (2026-10-02)")
    results = from_folder(tmp_path / "all")
    assert [(v.run, v.video) for v in results.videos] == [
        ("One (2026-10-01)", "a"),
        ("Two (2026-10-02)", "b"),
    ]


def test_a_flat_job_folder_groups_by_video_stem(flat_folder):
    results = from_folder(flat_folder)
    assert [(v.run, v.video) for v in results.videos] == [("", "demo_clip")]
    # The video, job_metadata.json and the backend's bare person_tracking.json
    # aren't results, so they're neither read nor reported as skipped.
    assert results.skipped == []
    assert len(results.videos[0].files) == 7


def test_unknown_files_are_reported_as_skipped(run_folder):
    (run_folder / "demo_clip" / "notes.txt").write_text("mine", encoding="utf-8")
    (run_folder / "demo_clip" / "demo_clip_future_pipeline.json").write_text("{}")
    skipped = from_folder(run_folder).skipped
    assert sorted(s.file for s in skipped) == [
        "demo_clip_future_pipeline.json",
        "notes.txt",
    ]
    assert all(s.reason == "no reader for this file" for s in skipped)


def test_exports_and_companions_are_not_results(run_folder):
    video = run_folder / "demo_clip"
    for name in ("demo_clip.eaf", "Demo.csv", "Demo.parquet", "methods.md"):
        (video / name).write_text("x", encoding="utf-8")
    results = from_folder(run_folder)
    assert results.skipped == []
    assert all(
        not f.path.name.endswith(".provenance.json") for f in results.videos[0].files
    )


def test_failed_video_in_run_json_is_missing_with_its_reason(run_folder):
    (run_folder / "clip7").mkdir()
    record = json.loads((run_folder / "run.json").read_text())
    record["videos"].append(
        {
            "job_id": "j7",
            "folder": "clip7",
            "source": {},
            "status": "failed",
            "error": "No audio could be extracted",
        }
    )
    record["videos"].append({"job_id": "j8", "folder": "clip8", "status": "running"})
    (run_folder / "run.json").write_text(json.dumps(record))
    results = from_folder(run_folder)
    assert {(m.video, m.reason) for m in results.missing} == {
        ("clip7", "failed: No audio could be extracted"),
        ("clip8", "still running"),
    }


def test_not_a_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        from_folder(tmp_path / "nope")


def _job(**kw):
    defaults = dict(
        job_id="j1",
        video_path=None,
        output_dir=None,
        storage_path=None,
        batch_name=None,
        status=SimpleNamespace(value="completed"),
        error_message=None,
    )
    return SimpleNamespace(**{**defaults, **kw})


def test_from_jobs_reads_a_run_layout_job(run_folder):
    job = _job(
        video_path=Path("/data/videos/demo_clip.mp4"),
        output_dir=run_folder / "demo_clip",
    )
    results = from_jobs([job])
    assert [(v.run, v.video) for v in results.videos] == [(RUN_NAME, "demo_clip")]
    assert len(results.videos[0].files) == 7


def test_from_jobs_reads_a_legacy_job_folder(flat_folder):
    job = _job(
        video_path=flat_folder / "demo_clip.mp4",
        storage_path=str(flat_folder),
        batch_name="Old run",
    )
    results = from_jobs([job])
    assert [(v.run, v.video) for v in results.videos] == [("Old run", "demo_clip")]
    assert results.videos[0].media_path == str(flat_folder / "demo_clip.mp4")


def test_from_jobs_reports_failed_jobs_and_same_named_videos(tmp_path, flat_folder):
    empty = tmp_path / "empty-job"
    empty.mkdir()
    failed = _job(
        job_id="j2",
        video_path=Path("x/demo_clip.mp4"),
        storage_path=str(empty),
        status=SimpleNamespace(value="failed"),
        error_message="boom",
    )
    ok = _job(video_path=Path("y/demo_clip.mp4"), storage_path=str(flat_folder))
    results = from_jobs([ok, failed])
    assert [v.video for v in results.videos] == ["demo_clip"]
    assert [(m.video, m.reason) for m in results.missing] == [
        ("demo_clip 2", "failed: boom")
    ]


def test_from_jobs_notes_a_running_job(flat_folder):
    job = _job(
        video_path=Path("demo_clip.mp4"),
        storage_path=str(flat_folder),
        status=SimpleNamespace(value="running"),
    )
    results = from_jobs([job])
    assert results.notes and "still running" in results.notes[0]


def test_loose_subfolders_are_told_apart_by_path(tmp_path):
    copy_outputs(FIXTURES, tmp_path / "top")
    copy_outputs(LEGACY, tmp_path / "top" / "old" / "legacy")
    results = from_folder(tmp_path / "top")
    assert [(v.run, v.video) for v in results.videos] == [
        ("", "demo_clip"),
        ("old/legacy", "demo_clip"),
    ]
