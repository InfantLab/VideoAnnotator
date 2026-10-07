"""Results belong to the researcher under rootful Docker (spec 024, R6)."""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from videoannotator import results_folder as rf
from videoannotator.batch.types import BatchJob, JobStatus


@pytest.fixture
def chowned(monkeypatch):
    calls: list[Path] = []
    monkeypatch.setattr(
        rf.os, "chown", lambda path, uid, gid: calls.append(Path(path)), raising=False
    )
    return calls


def _finished_run(tmp_path):
    video = tmp_path / "child01.mp4"
    video.write_bytes(b"v")
    run = rf.RunFolder.create("Study", batch_id=None, pipelines=[], config={})
    job = BatchJob(video_path=video, status=JobStatus.COMPLETED)
    folder = run.add_video(job, video)
    (folder / "child01_scenes.json").write_text("[]")
    (folder / "frames").mkdir()
    (folder / "frames" / "0001.png").write_bytes(b"png")
    return run, job, folder


def test_everything_written_goes_to_the_owner(
    tmp_path, results_root, chowned, monkeypatch
):
    monkeypatch.setenv("VIDEOANNOTATOR_RESULTS_OWNER", "1000:1000")
    run, job, folder = _finished_run(tmp_path)
    rf.record_job_finished(job)

    assert results_root in chowned
    assert run.path in chowned
    assert folder in chowned
    assert folder / "child01_scenes.json" in chowned
    assert folder / "frames" in chowned
    assert folder / "frames" / "0001.png" in chowned
    assert chowned.count(run.path / rf.RUN_RECORD) >= 2  # created and updated


def test_nothing_is_chowned_without_an_owner(
    tmp_path, results_root, chowned, monkeypatch
):
    monkeypatch.delenv("VIDEOANNOTATOR_RESULTS_OWNER", raising=False)
    _, job, _ = _finished_run(tmp_path)
    rf.record_job_finished(job)
    assert chowned == []


def test_a_chown_failure_is_logged_not_raised(
    tmp_path, results_root, monkeypatch, caplog
):
    monkeypatch.setenv("VIDEOANNOTATOR_RESULTS_OWNER", "1000:1000")
    monkeypatch.setattr(
        rf.os, "chown", MagicMock(side_effect=PermissionError("no")), raising=False
    )
    _, job, _ = _finished_run(tmp_path)
    rf.record_job_finished(job)
    assert "1000:1000" in caplog.text


@pytest.mark.skipif(not hasattr(os, "chown"), reason="POSIX only")
def test_owner_is_the_one_given(tmp_path, results_root, monkeypatch):
    seen = []
    monkeypatch.setattr(rf.os, "chown", lambda p, u, g: seen.append((u, g)))
    monkeypatch.setenv("VIDEOANNOTATOR_RESULTS_OWNER", "1234:5678")
    _finished_run(tmp_path)
    assert set(seen) == {(1234, 5678)}
