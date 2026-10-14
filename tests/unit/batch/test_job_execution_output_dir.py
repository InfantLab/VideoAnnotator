"""Where the runner writes (spec 022): the job's results folder, else its own folder."""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

from videoannotator import results_folder as rf
from videoannotator.batch.job_execution import run_job_pipelines
from videoannotator.batch.types import BatchJob, JobStatus
from videoannotator.storage.base import StorageBackend


class WritingPipeline:
    def __init__(self, config):
        pass

    def initialize(self):
        pass

    def cleanup(self):
        pass

    def process(self, video_path, output_dir, **kwargs):
        stem = Path(video_path).stem
        (Path(output_dir) / f"{stem}_scene_detection.json").write_text("[]")
        return [{"scene": 1}]


def _storage(job: BatchJob) -> MagicMock:
    storage = MagicMock(spec=StorageBackend)
    storage.annotation_exists.return_value = False
    storage.save_annotations.return_value = "fake://annotations"
    storage.load_job_metadata.return_value = job
    return storage


def _video(tmp_path, name="child01.mp4") -> Path:
    video = tmp_path / "videos" / name
    video.parent.mkdir(exist_ok=True)
    video.write_bytes(b"video")
    return video


def test_outputs_go_to_output_dir_and_nothing_to_storage_path(tmp_path, results_root):
    video = _video(tmp_path)
    run = rf.RunFolder.create(
        "R", batch_id=None, pipelines=["scene_detection"], config={}, today=date.today()
    )
    job = BatchJob(
        video_path=video,
        selected_pipelines=["scene_detection"],
        storage_path=tmp_path / "internal",
    )
    job.storage_path.mkdir()
    run.add_video(job, video)

    run_job_pipelines(job, _storage(job), {"scene_detection": WritingPipeline})

    assert job.status == JobStatus.COMPLETED
    assert (job.output_dir / "child01_scene_detection.json").is_file()
    assert list(job.storage_path.iterdir()) == []
    entry = rf.read_record(run.path)["videos"][0]
    assert entry["status"] == "completed"
    assert "child01_scene_detection.json" in entry["files"]


def test_jobs_from_before_results_folders_still_use_their_own_folder(tmp_path):
    video = _video(tmp_path)
    job = BatchJob(
        video_path=video,
        selected_pipelines=["scene_detection"],
        storage_path=tmp_path / "internal",
    )
    job.storage_path.mkdir()

    run_job_pipelines(job, _storage(job), {"scene_detection": WritingPipeline})

    assert job.output_dir == job.storage_path
    assert (job.storage_path / "child01_scene_detection.json").is_file()


def test_a_missing_video_fails_with_its_location(tmp_path, results_root, ingest_root):
    video = _video(ingest_root.parent)
    run = rf.RunFolder.create("R", batch_id=None, pipelines=[], config={})
    job = BatchJob(video_path=video, selected_pipelines=["scene_detection"])
    run.add_video(job, video)
    video.unlink()
    pipeline = MagicMock()

    run_job_pipelines(job, _storage(job), {"scene_detection": pipeline})

    assert job.status == JobStatus.FAILED
    assert job.error_message == (
        f"Video not found: {video} (moved or deleted since the job was created)"
    )
    pipeline.assert_not_called()
    entry = rf.read_record(run.path)["videos"][0]
    assert entry["status"] == "failed"
    assert entry["error"] == job.error_message


def test_a_video_in_a_folder_no_longer_shared_says_so(tmp_path, ingest_root):
    video = tmp_path / "Old study" / "child01.mp4"
    job = BatchJob(video_path=video, selected_pipelines=["s"])

    run_job_pipelines(job, _storage(job), {"s": WritingPipeline})

    assert job.status == JobStatus.FAILED
    assert job.error_message == (
        f"Video not found: {video} "
        f"({video.parent} isn't shared with VideoAnnotator any more)"
    )


def test_one_missing_video_does_not_stop_the_next_job(tmp_path):
    gone = BatchJob(video_path=tmp_path / "gone.mp4", selected_pipelines=["s"])
    present = BatchJob(
        video_path=_video(tmp_path), selected_pipelines=["s"], output_dir=tmp_path / "o"
    )
    run_job_pipelines(gone, _storage(gone), {"s": WritingPipeline})
    run_job_pipelines(present, _storage(present), {"s": WritingPipeline})
    assert gone.status == JobStatus.FAILED
    assert present.status == JobStatus.COMPLETED
