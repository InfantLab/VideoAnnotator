"""Single-job processing for the API server's background job manager.

Thin wrapper around `batch.job_execution.run_job_pipelines` — the one real
execution path (see specs/006-job-execution-consolidation/spec.md). Kept as
its own class/module because `api/background_tasks.py` imports and
instantiates it lazily to avoid loading pipeline modules at server startup.
"""

from datetime import datetime

from videoannotator.batch.job_execution import run_job_pipelines
from videoannotator.registry import get_pipeline_loader
from videoannotator.utils.logging_config import get_logger

from ..batch.types import BatchJob, JobStatus
from ..storage.base import StorageBackend

logger = get_logger("api")


class JobProcessor:
    """Loads pipeline classes once, then runs jobs through the shared
    execution path."""

    def __init__(self):
        loader = get_pipeline_loader()
        self.pipeline_classes = loader.load_all_pipelines()

        if not self.pipeline_classes:
            logger.warning("No pipeline classes loaded - check registry metadata")
        else:
            logger.info(
                f"Pipeline classes loaded: {sorted(self.pipeline_classes.keys())}"
            )

    def process_job(self, job: BatchJob, storage: StorageBackend) -> BatchJob:
        """Run `job`'s selected pipelines and return it with its final
        status (COMPLETED/FAILED/CANCELLED) already set and persisted."""
        if not self.pipeline_classes:
            logger.error("No pipeline classes loaded! Import may have failed.")
            job.error_message = "No pipeline classes available"
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now()
            storage.save_job_metadata(job)
            return job

        if job.video_path is None or not job.video_path.exists():
            error_msg = f"Video file not found: {job.video_path}"
            logger.error(error_msg)
            job.error_message = error_msg
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now()
            storage.save_job_metadata(job)
            return job

        logger.info(f"Processing job {job.job_id}: {job.video_path}")
        return run_job_pipelines(job, storage, self.pipeline_classes)
