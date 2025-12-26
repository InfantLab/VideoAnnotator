import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from videoannotator.api.job_processor import JobProcessor
from videoannotator.batch.types import BatchJob, JobStatus, PipelineResult


class TestJobProcessorPartialFailure(unittest.TestCase):
    def setUp(self):
        self.processor = JobProcessor()
        # Mock pipeline classes
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

    def tearDown(self):
        try:
            self.video_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            self._temp_dir.cleanup()
        except Exception:
            pass

    def test_partial_failure(self):
        job = BatchJob(
            job_id="test_job",
            video_path=self.video_path,
            output_dir=self.output_dir,
            selected_pipelines=["pipeline1", "pipeline2"],
        )

        # Mock _process_pipeline to succeed for pipeline1 and fail for pipeline2
        def side_effect(job, pipeline_name):
            if pipeline_name == "pipeline1":
                return True
            else:
                # Simulate failure behavior of _process_pipeline
                # It creates a PipelineResult with FAILED status
                if not hasattr(job, "pipeline_results"):
                    job.pipeline_results = {}
                job.pipeline_results[pipeline_name] = PipelineResult(
                    pipeline_name=pipeline_name,
                    status=JobStatus.FAILED,
                    error_message="Simulated failure",
                )
                return False

        with patch.object(self.processor, "_process_pipeline", side_effect=side_effect):
            success = self.processor.process_job(job)

        self.assertTrue(
            success, "Job should be considered successful (partial success)"
        )
        self.assertIn("Completed with errors", job.error_message)
        self.assertIn("pipeline2", job.error_message)

    def test_all_failure(self):
        job = BatchJob(
            job_id="test_job",
            video_path=self.video_path,
            output_dir=self.output_dir,
            selected_pipelines=["pipeline1", "pipeline2"],
        )

        # Mock _process_pipeline to fail for all
        def side_effect(job, pipeline_name):
            if not hasattr(job, "pipeline_results"):
                job.pipeline_results = {}
            job.pipeline_results[pipeline_name] = PipelineResult(
                pipeline_name=pipeline_name,
                status=JobStatus.FAILED,
                error_message="Simulated failure",
            )
            return False

        with patch.object(self.processor, "_process_pipeline", side_effect=side_effect):
            success = self.processor.process_job(job)

        self.assertFalse(
            success, "Job should be considered failed if all pipelines fail"
        )
        self.assertIn("All pipelines failed", job.error_message)


if __name__ == "__main__":
    unittest.main()
