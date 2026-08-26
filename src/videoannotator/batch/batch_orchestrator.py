"""BatchOrchestrator: Core batch processing engine for VideoAnnotator.

Manages job queues, worker pools, progress tracking, and failure recovery
for robust large-scale video processing.
"""

import logging
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from ..storage.base import StorageBackend
from .job_execution import run_job_pipelines
from .progress_tracker import ProgressTracker
from .recovery import FailureRecovery, RetryStrategy
from .types import (
    BatchJob,
    BatchReport,
    BatchStatus,
    ConfigDict,
    JobStatus,
    VideoPath,
)


class BatchOrchestrator:
    """Orchestrate batch processing jobs across configured pipelines."""

    async def start(self, max_workers: int = 4, save_checkpoints: bool = True):
        """Async entry point for batch processing (for test compatibility).

        Runs run_batch in a thread executor.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.run_batch, max_workers, save_checkpoints
        )

    async def stop(self) -> None:
        """Request batch processing stop.

        This is a lightweight async helper primarily for test compatibility.
        """

        self.should_stop = True

    # Alias for testing compatibility (so patch.object works in tests)
    def _process_job(self, job):
        return self._process_job_with_retry(job)

    def __init__(
        self,
        storage_backend: StorageBackend | None = None,
        max_retries: int = 3,
        retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
        checkpoint_interval: int = 10,
    ):
        """Initialize batch orchestrator.

        Args:
            storage_backend: Storage backend for annotations and metadata
            max_retries: Maximum retry attempts per failed job
            retry_strategy: Strategy for retry delays
            checkpoint_interval: Save checkpoint every N completed jobs
        """
        if storage_backend is None:
            # Lazy import to avoid circular import
            from videoannotator.storage.file_backend import FileStorageBackend

            storage_backend = FileStorageBackend(Path("batch_results"))

        self.storage_backend = storage_backend
        self.progress_tracker = ProgressTracker()
        self.failure_recovery = FailureRecovery(
            max_retries=max_retries, strategy=retry_strategy
        )
        self.checkpoint_interval = checkpoint_interval
        self.logger = logging.getLogger(__name__)

        # Current batch state
        self.batch_id: str | None = None
        self.jobs: list[BatchJob] = []
        self.is_running = False
        self.should_stop = False

        # Load pipeline classes dynamically from registry
        self._import_pipelines()

    def _import_pipelines(self):
        """Import pipeline classes using registry loader."""
        from videoannotator.registry import get_pipeline_loader

        loader = get_pipeline_loader()
        self.pipeline_classes = loader.load_all_pipelines()

        if not self.pipeline_classes:
            self.logger.warning("No pipeline classes loaded - check registry metadata")
        else:
            self.logger.info(
                f"Pipeline classes loaded: {sorted(self.pipeline_classes.keys())}"
            )

    def add_job(
        self,
        video_path: VideoPath,
        output_dir: VideoPath | None = None,
        config: ConfigDict | None = None,
        selected_pipelines: list[str] | None = None,
    ) -> str:
        """Add a video processing job to the batch queue.

        Args:
            video_path: Path to video file
            output_dir: Directory for output files (auto-generated if None)
            config: Pipeline configuration overrides
            selected_pipelines: List of pipeline names to run (all if None)

        Returns:
            Job ID for tracking
        """
        if isinstance(video_path, str) and not video_path.strip():
            raise FileNotFoundError("Video file not found: (empty path)")

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Generate output directory if not provided
        if output_dir is None:
            output_dir = Path("batch_results") / "jobs" / video_path.stem
        else:
            output_dir = Path(output_dir)

        # Create job
        job = BatchJob(
            video_path=video_path,
            output_dir=output_dir,
            config=config or {},
            selected_pipelines=selected_pipelines,
        )

        self.jobs.append(job)
        self.logger.info(f"Added job {job.job_id} for video: {video_path}")

        return job.job_id

    def add_jobs_from_directory(
        self,
        input_dir: VideoPath,
        output_dir: VideoPath | None = None,
        config: ConfigDict | None = None,
        selected_pipelines: list[str] | None = None,
        extensions: list[str] | None = None,
    ) -> list[str]:
        """Add multiple jobs from a directory of videos.

        Args:
            input_dir: Directory containing video files
            output_dir: Base output directory
            config: Pipeline configuration
            selected_pipelines: List of pipeline names to run
            extensions: Video file extensions to include

        Returns:
            List of job IDs
        """
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        extensions = extensions or [
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".wmv",
            ".flv",
            ".webm",
        ]

        job_ids = []
        for ext in extensions:
            for video_path in input_dir.rglob(f"*{ext}"):
                if video_path.is_file():
                    # Create individual output directory for each video
                    if output_dir:
                        video_output_dir = Path(output_dir) / video_path.stem
                    else:
                        video_output_dir = None

                    job_id = self.add_job(
                        video_path=video_path,
                        output_dir=video_output_dir,
                        config=config,
                        selected_pipelines=selected_pipelines,
                    )
                    job_ids.append(job_id)

        self.logger.info(f"Added {len(job_ids)} jobs from directory: {input_dir}")
        return job_ids

    def run_batch(
        self, max_workers: int = 4, save_checkpoints: bool = True
    ) -> BatchReport:
        """Execute all jobs in the batch queue.

        Args:
            max_workers: Maximum number of parallel workers
            save_checkpoints: Whether to save periodic checkpoints

        Returns:
            BatchReport with results and statistics
        """
        if not self.jobs:
            raise ValueError("No jobs in batch queue")

        self.batch_id = str(uuid.uuid4())
        self.is_running = True
        self.should_stop = False

        self.logger.info(
            f"Starting batch {self.batch_id} with {len(self.jobs)} jobs and {max_workers} workers"
        )

        # Initialize progress tracking
        self.progress_tracker.start_batch()

        # Create batch report
        report = BatchReport(
            batch_id=self.batch_id,
            start_time=datetime.now(),
            total_jobs=len(self.jobs),
        )

        # Process jobs with thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            from concurrent.futures import FIRST_COMPLETED, wait

            future_to_job: dict[Future, BatchJob] = {}
            submitted_job_ids: set[str] = set()
            completed_count = 0

            def submit_pending_jobs() -> None:
                nonlocal report

                for job in self.jobs:
                    if (
                        job.status == JobStatus.PENDING
                        and job.job_id not in submitted_job_ids
                    ):
                        future = executor.submit(self._process_job_with_retry, job)
                        future_to_job[future] = job
                        submitted_job_ids.add(job.job_id)
                        job.status = JobStatus.RUNNING
                        job.started_at = datetime.now()
                        self.progress_tracker.start_job(job.job_id)

                # Keep report total_jobs in sync if jobs are added during run.
                report.total_jobs = len(self.jobs)

            submit_pending_jobs()

            while future_to_job:
                if self.should_stop:
                    self.logger.info("Batch processing stopped by user request")
                    break

                done, _pending = wait(
                    future_to_job.keys(), timeout=0.1, return_when=FIRST_COMPLETED
                )

                # Pick up any jobs added while workers are running.
                submit_pending_jobs()

                if not done:
                    continue

                for future in done:
                    job = future_to_job.pop(future)

                    try:
                        result_job = future.result()
                        job.status = result_job.status
                        job.pipeline_results = result_job.pipeline_results
                        job.completed_at = datetime.now()
                        job.error_message = result_job.error_message

                        self.progress_tracker.complete_job(job)

                        if job.status == JobStatus.COMPLETED:
                            report.completed_jobs += 1
                        elif job.status == JobStatus.FAILED:
                            report.failed_jobs += 1
                            if job.error_message:
                                report.errors.append(
                                    f"Job {job.job_id}: {job.error_message}"
                                )

                        completed_count += 1

                        if (
                            save_checkpoints
                            and completed_count % self.checkpoint_interval == 0
                        ):
                            self._save_checkpoint()

                        self.progress_tracker.log_progress(self.jobs)

                    except Exception as e:
                        self.logger.error(
                            f"Unexpected error processing job {job.job_id}: {e}"
                        )
                        job.status = JobStatus.FAILED
                        job.error_message = str(e)
                        report.failed_jobs += 1
                        report.errors.append(f"Job {job.job_id}: {e!s}")

        # Finalize report
        report.end_time = datetime.now()
        report.total_jobs = len(self.jobs)
        report.jobs = self.jobs.copy()
        report.total_processing_time = self.progress_tracker.total_processing_time

        self.is_running = False

        # Save final results
        self._save_final_results(report)

        self.logger.info(
            f"Batch {self.batch_id} completed: "
            f"{report.completed_jobs}/{report.total_jobs} successful "
            f"({report.success_rate:.1f}% success rate)"
        )

        return report

    def resume_batch(self, checkpoint_file: str) -> BatchReport:
        """Resume batch processing from a checkpoint.

        Args:
            checkpoint_file: Path to checkpoint file

        Returns:
            BatchReport with results
        """
        jobs = self.failure_recovery.load_checkpoint(checkpoint_file)
        if not jobs:
            raise ValueError(f"Could not load checkpoint: {checkpoint_file}")

        # Filter to incomplete jobs
        incomplete_jobs = [job for job in jobs if not job.is_complete]

        self.jobs = jobs
        self.logger.info(f"Resuming batch with {len(incomplete_jobs)} incomplete jobs")

        # Reset incomplete jobs to pending
        for job in incomplete_jobs:
            job.status = JobStatus.PENDING
            job.started_at = None

        return self.run_batch()

    def get_status(self) -> BatchStatus:
        """Get current batch processing status."""
        return self.progress_tracker.get_status(self.jobs)

    def stop_batch(self) -> None:
        """Request graceful stop of batch processing."""
        self.should_stop = True
        self.logger.info("Batch stop requested")

    def clear_jobs(self) -> None:
        """Clear all jobs from the queue."""
        self.jobs.clear()
        self.logger.info("Cleared job queue")

    def get_job(self, job_id: str) -> BatchJob | None:
        """Get a job by its ID."""
        for job in self.jobs:
            if job.job_id == job_id:
                return job
        return None

    def update_job_status(self, job_id: str, status: JobStatus) -> bool:
        """Update the status of a job."""
        job = self.get_job(job_id)
        if job:
            job.status = status
            if status == JobStatus.RUNNING and job.started_at is None:
                job.started_at = datetime.now()
            elif status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                job.completed_at = datetime.now()
            return True
        return False

    def set_job_error(self, job_id: str, error_message: str) -> bool:
        """Set error message for a job and mark it as failed."""
        job = self.get_job(job_id)
        if job:
            job.status = JobStatus.FAILED
            job.error_message = error_message
            job.completed_at = datetime.now()
            return True
        return False

    def increment_retry_count(self, job_id: str) -> bool:
        """Increment the retry count for a job."""
        job = self.get_job(job_id)
        if job:
            job.retry_count += 1
            return True
        return False

    def add_pipeline_result(self, job_id: str, result) -> bool:
        """Add a pipeline result to a job."""
        job = self.get_job(job_id)
        if job:
            if job.pipeline_results is None:
                job.pipeline_results = {}
            job.pipeline_results[result.pipeline_name] = result
            return True
        return False

    def _should_retry_job(self, job_id: str) -> bool:
        """Check if a job should be retried based on failure recovery.

        settings.
        """
        job = self.get_job(job_id)
        if not job:
            return False

        # Simple retry logic - could be expanded to use failure_recovery
        max_retries = 3  # Default max retries
        return job.retry_count < max_retries

    def _process_job_with_retry(self, job: BatchJob) -> BatchJob:
        """Process a single job, automatically retrying a transient failure
        with backoff (distinct from spec 006's user-triggered retry of an
        already-terminal job — this is in-flight, same orchestrator run).

        `_process_single_job` (via the shared `run_job_pipelines` path)
        never raises — it always returns a job with a settled status — so
        retry-worthiness is decided from `job.status`/`job.error_message`,
        not a caught exception. `should_retry`/`prepare_retry` only inspect
        the error's string content, not its type, so synthesizing an
        exception from the stored message here is behaviourally identical
        to the original exception-based flow.

        Args:
            job: Job to process

        Returns:
            Updated job with results
        """
        max_attempts = self.failure_recovery.max_retries + 1

        for attempt in range(max_attempts):
            job = self._process_single_job(job)
            if job.status != JobStatus.FAILED:
                return job

            error = RuntimeError(job.error_message or "Job processing failed")
            self.logger.error(
                f"Job {job.job_id} failed (attempt {attempt + 1}/{max_attempts}): {error}"
            )
            if attempt < max_attempts - 1 and self.failure_recovery.should_retry(
                job, error
            ):
                job = self.failure_recovery.prepare_retry(job, error)
                delay = self.failure_recovery.calculate_retry_delay(job)
                if delay > 0:
                    self.logger.info(
                        f"Waiting {delay:.1f}s before retrying job {job.job_id}"
                    )
                    time.sleep(delay)
            else:
                break
        return job

    def _process_single_job(self, job: BatchJob) -> BatchJob:
        """Process a single job through selected pipelines.

        Delegates to the shared `run_job_pipelines` execution path (spec
        006) — the same one `api/job_processor.py` uses. `should_stop` is a
        batch-wide graceful-shutdown flag scoped to this orchestrator
        instance (distinct from per-job cancellation, which the shared path
        checks by re-reading the job's own persisted status); checked once
        here since a stopped batch shouldn't start a new job's pipelines at
        all.

        Args:
            job: Job to process

        Returns:
            Updated job with results
        """
        if self.should_stop:
            job.status = JobStatus.CANCELLED
            return job

        self.logger.info(f"Processing job {job.job_id}: {job.video_path}")
        return run_job_pipelines(job, self.storage_backend, self.pipeline_classes)

    def _save_checkpoint(self) -> None:
        """Save current batch state as checkpoint."""
        if self.batch_id:
            # Get storage base directory
            if hasattr(self.storage_backend, "base_dir"):
                base_dir = Path(self.storage_backend.base_dir)
            else:
                base_dir = Path("batch_results")

            # Create checkpoints directory
            checkpoints_dir = base_dir / "checkpoints"
            checkpoints_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_file = checkpoints_dir / f"checkpoint_{self.batch_id}.json"
            self.failure_recovery.create_checkpoint(self.jobs, str(checkpoint_file))

    def _save_final_results(self, report: BatchReport) -> None:
        """Save final batch results."""
        # Get storage base directory
        if hasattr(self.storage_backend, "base_dir"):
            base_dir = Path(self.storage_backend.base_dir)
        else:
            base_dir = Path("batch_results")

        # Create reports directory
        reports_dir = base_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Save batch report
        report_file = reports_dir / f"batch_report_{self.batch_id}.json"
        with open(report_file, "w") as f:
            import json

            json.dump(report.to_dict(), f, indent=2, default=str)

        # Save recovery statistics
        recovery_file = reports_dir / f"recovery_report_{self.batch_id}.json"
        with open(recovery_file, "w") as f:
            import json

            json.dump(self.failure_recovery.get_recovery_report(), f, indent=2)

        self.logger.info(f"Saved final results: {report_file}")
