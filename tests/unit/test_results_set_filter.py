"""ResultsSet.only: each video's latest result per pipeline (spec 029 R10)."""

from __future__ import annotations

from pathlib import Path

from tests.unit.exporters.conftest import FIXTURES, copy_outputs
from videoannotator.batch.types import BatchJob, JobStatus
from videoannotator.exporters.results_set import from_jobs


def _job(tmp_path: Path, name: str) -> BatchJob:
    return BatchJob(
        video_path=Path("/data/demo_clip.mp4"),
        output_dir=copy_outputs(FIXTURES, tmp_path / name),
        status=JobStatus.COMPLETED,
        batch_name=name,
    )


def test_only_keeps_the_named_pipelines_of_each_job(tmp_path):
    older, newer = _job(tmp_path, "older"), _job(tmp_path, "newer")
    results = from_jobs([older, newer])

    kept = results.only(
        {(newer.job_id, "scene_detection"), (older.job_id, "speaker_diarization")}
    )

    by_run = {v.run: {f.pipeline for f in v.files} for v in kept.videos}
    assert by_run == {"newer": {"scene_detection"}, "older": {"speaker_diarization"}}


def test_videos_left_with_nothing_are_dropped(tmp_path):
    older, newer = _job(tmp_path, "older"), _job(tmp_path, "newer")

    kept = from_jobs([older, newer]).only({(newer.job_id, "scene_detection")})

    assert [v.run for v in kept.videos] == ["newer"]
    assert from_jobs([older]).only(set()).is_empty()
