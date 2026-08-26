import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from videoannotator.api.job_processor import JobProcessor
from videoannotator.batch.types import BatchJob, JobStatus
from videoannotator.storage.base import StorageBackend


class TestJobProcessorPartialFailure(unittest.TestCase):
    """Exercises partial/total pipeline-failure aggregation end-to-end
    through the real shared execution path (batch.job_execution), rather
    than mocking an internal method — that internal seam (JobProcessor.
    _process_pipeline) no longer exists post-006-consolidation; both
    api/job_processor.py and batch/batch_orchestrator.py now delegate to
    the same batch.job_execution.run_job_pipelines()."""

    def setUp(self):
        self.processor = JobProcessor()
        self.processor.pipeline_classes = {
            "pipeline1": MagicMock(),
            "pipeline2": MagicMock(),
        }

        self._temp_dir = tempfile.TemporaryDirectory()
        self._video_file = tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False, dir=self._temp_dir.name
        )
        self._video_file.write(b"test")
        self._video_file.flush()
        self._video_file.close()

        self.video_path = Path(self._video_file.name)
        self.output_dir = Path(self._temp_dir.name) / "output"

        # Minimal fake storage: no annotations pre-exist, saves are no-ops,
        # and re-reading the job for the cancellation checkpoint just
        # returns whatever the in-memory job currently looks like (never
        # cancelled in these tests).
        self.storage = MagicMock(spec=StorageBackend)
        self.storage.annotation_exists.return_value = False
        self.storage.save_annotations.return_value = "fake://annotations"

    def tearDown(self):
        try:
            self.video_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            self._temp_dir.cleanup()
        except Exception:
            pass

    def _make_job(self) -> BatchJob:
        job = BatchJob(
            job_id="test_job",
            video_path=self.video_path,
            output_dir=self.output_dir,
            selected_pipelines=["pipeline1", "pipeline2"],
        )
        # The execution path re-reads job status via storage.load_job_metadata
        # to check for cancellation between pipelines; return the same job.
        self.storage.load_job_metadata.side_effect = lambda job_id: job
        return job

    def test_partial_failure(self):
        job = self._make_job()

        self.processor.pipeline_classes[
            "pipeline1"
        ].return_value.process.return_value = [{"annotation": "ok"}]
        self.processor.pipeline_classes[
            "pipeline2"
        ].return_value.process.side_effect = RuntimeError("Simulated failure")

        result = self.processor.process_job(job, self.storage)

        self.assertEqual(result.status, JobStatus.COMPLETED)
        self.assertIn("Completed with errors", result.error_message)
        self.assertIn("pipeline2", result.error_message)
        self.assertEqual(
            result.pipeline_results["pipeline1"].status, JobStatus.COMPLETED
        )
        self.assertEqual(result.pipeline_results["pipeline2"].status, JobStatus.FAILED)

    def test_all_failure(self):
        job = self._make_job()

        for name in ("pipeline1", "pipeline2"):
            self.processor.pipeline_classes[
                name
            ].return_value.process.side_effect = RuntimeError("Simulated failure")

        result = self.processor.process_job(job, self.storage)

        self.assertEqual(result.status, JobStatus.FAILED)
        self.assertIn("All pipelines failed", result.error_message)


if __name__ == "__main__":
    unittest.main()
