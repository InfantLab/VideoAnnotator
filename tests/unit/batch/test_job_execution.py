"""Unit tests for the spec-006 shared job-execution path."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from videoannotator.batch.job_execution import run_job_pipelines
from videoannotator.batch.types import BatchJob, JobStatus
from videoannotator.storage.base import StorageBackend


class _FakeStatefulStorage:
    """Models the real SQLiteStorageBackend's persistence semantics that a
    plain MagicMock does not: `save_job_metadata` blindly overwrites the
    whole stored row (including `status`) from whatever is in `job` at call
    time, and `load_job_metadata` returns whatever was last saved. A
    MagicMock's `load_job_metadata` is controlled independently of what was
    "saved" to it, so it can't reproduce a lost-update race between a
    concurrent external write (e.g. a /cancel request) and this module's
    own saves - exactly the gap that let the real bug (see
    test_progress_save_does_not_clobber_concurrent_cancel below) pass every
    mock-based test while failing live."""

    def __init__(self):
        self._jobs: dict[str, BatchJob] = {}
        self.annotation_exists = MagicMock(return_value=False)
        self.save_annotations = MagicMock(return_value="fake://annotations")

    def save_job_metadata(self, job: BatchJob) -> None:
        self._jobs[job.job_id] = copy.deepcopy(job)

    def load_job_metadata(self, job_id: str) -> BatchJob:
        return copy.deepcopy(self._jobs[job_id])

    def externally_cancel(self, job_id: str) -> None:
        """Simulates a concurrent /cancel API request landing mid-run: a
        totally separate load+mutate+save, independent of whatever the
        execution path's own in-memory `job` object currently holds."""
        current = self.load_job_metadata(job_id)
        current.status = JobStatus.CANCELLED
        current.error_message = "Job cancelled by user request"
        self.save_job_metadata(current)


class TestRunJobPipelines(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        video_file = tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False, dir=self._temp_dir.name
        )
        video_file.write(b"test")
        video_file.close()
        self.video_path = Path(video_file.name)
        self.output_dir = Path(self._temp_dir.name) / "output"

        self.storage = MagicMock(spec=StorageBackend)
        self.storage.annotation_exists.return_value = False
        self.storage.save_annotations.return_value = "fake://annotations"

    def tearDown(self):
        self._temp_dir.cleanup()

    def _job(self, pipelines: list[str]) -> BatchJob:
        return BatchJob(
            job_id="job-under-test",
            video_path=self.video_path,
            output_dir=self.output_dir,
            selected_pipelines=pipelines,
        )

    def _succeeding_pipeline_class(self) -> MagicMock:
        cls = MagicMock()
        cls.return_value.process.return_value = [{"annotation": "ok"}]
        return cls

    def test_cancellation_after_only_pipeline_settles_cancelled_not_completed(self):
        """The gap a live test caught: a single-pipeline job cancelled while
        that one pipeline was running must not be silently overwritten back
        to COMPLETED just because there's no "next" pipeline to check
        before (FR-003)."""
        job = self._job(["only_pipeline"])
        pipeline_classes = {"only_pipeline": self._succeeding_pipeline_class()}

        cancelled_job = BatchJob(job_id=job.job_id, status=JobStatus.CANCELLED)
        self.storage.load_job_metadata.return_value = cancelled_job

        result = run_job_pipelines(job, self.storage, pipeline_classes)

        self.assertEqual(result.status, JobStatus.CANCELLED)
        self.assertNotEqual(result.status, JobStatus.COMPLETED)

    def test_cancellation_between_pipelines_prevents_next_one_running(self):
        job = self._job(["first", "second"])
        second_cls = self._succeeding_pipeline_class()
        pipeline_classes = {
            "first": self._succeeding_pipeline_class(),
            "second": second_cls,
        }

        # Report cancelled only once "first" has already been recorded as done.
        def load_side_effect(job_id):
            if "first" in job.pipeline_results:
                return BatchJob(job_id=job_id, status=JobStatus.CANCELLED)
            return BatchJob(job_id=job_id, status=JobStatus.RUNNING)

        self.storage.load_job_metadata.side_effect = load_side_effect

        result = run_job_pipelines(job, self.storage, pipeline_classes)

        self.assertEqual(result.status, JobStatus.CANCELLED)
        second_cls.return_value.process.assert_not_called()
        self.assertIn("first", result.pipeline_results)
        self.assertNotIn("second", result.pipeline_results)

    def test_normal_completion_reaches_100_percent(self):
        job = self._job(["a", "b"])
        pipeline_classes = {
            "a": self._succeeding_pipeline_class(),
            "b": self._succeeding_pipeline_class(),
        }
        self.storage.load_job_metadata.return_value = BatchJob(
            job_id=job.job_id, status=JobStatus.RUNNING
        )

        result = run_job_pipelines(job, self.storage, pipeline_classes)

        self.assertEqual(result.status, JobStatus.COMPLETED)
        self.assertEqual(result.progress_percentage, 100.0)

    def test_progress_updates_incrementally(self):
        job = self._job(["a", "b", "c", "d"])
        pipeline_classes = {
            name: self._succeeding_pipeline_class() for name in ("a", "b", "c", "d")
        }
        self.storage.load_job_metadata.return_value = BatchJob(
            job_id=job.job_id, status=JobStatus.RUNNING
        )

        seen_progress = []

        def save_side_effect(saved_job):
            seen_progress.append(saved_job.progress_percentage)

        self.storage.save_job_metadata.side_effect = save_side_effect

        run_job_pipelines(job, self.storage, pipeline_classes)

        # 25/50/75/100 after each of the 4 pipelines (plus the initial
        # RUNNING save at 0%).
        self.assertIn(25.0, seen_progress)
        self.assertIn(50.0, seen_progress)
        self.assertIn(75.0, seen_progress)
        self.assertIn(100.0, seen_progress)


class TestProgressSaveVsConcurrentCancel(unittest.TestCase):
    """Reproduces the live bug found while verifying spec 006: the
    per-pipeline progress save writes `job.status` from its own stale
    in-memory copy (RUNNING), which - if saved before the cancellation
    checkpoint re-reads - clobbers a CANCELLED status written concurrently
    by a real /cancel request back to RUNNING, an instant before the
    checkpoint's own read sees it (and thus, too late, correctly reports
    "not cancelled"). Requires the stateful fake storage above; a MagicMock
    can't reproduce this since its save/load are independent of each
    other."""

    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        video_file = tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False, dir=self._temp_dir.name
        )
        video_file.write(b"test")
        video_file.close()
        self.video_path = Path(video_file.name)
        self.output_dir = Path(self._temp_dir.name) / "output"
        self.storage = _FakeStatefulStorage()

    def tearDown(self):
        self._temp_dir.cleanup()

    def test_progress_save_does_not_clobber_concurrent_cancel(self):
        job = BatchJob(
            job_id="race-job",
            video_path=self.video_path,
            output_dir=self.output_dir,
            selected_pipelines=["only_pipeline"],
        )

        # The pipeline's .process() call is where a real cancel would land
        # while the (only) pipeline is genuinely mid-run - simulate that by
        # firing the external cancel as a side effect of process() being
        # called, before run_job_pipelines gets anywhere near its own
        # post-pipeline checkpoint.
        cls = MagicMock()

        def process_and_cancel(*args, **kwargs):
            self.storage.externally_cancel("race-job")
            return [{"annotation": "ok"}]

        cls.return_value.process.side_effect = process_and_cancel

        result = run_job_pipelines(job, self.storage, {"only_pipeline": cls})

        self.assertEqual(result.status, JobStatus.CANCELLED)
        persisted = self.storage.load_job_metadata("race-job")
        self.assertEqual(persisted.status, JobStatus.CANCELLED)


if __name__ == "__main__":
    unittest.main()
