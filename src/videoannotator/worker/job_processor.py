"""Background job processor for VideoAnnotator API server.

Continuously polls the database for pending jobs and processes them
using the BatchOrchestrator system.

v1.3.0: Enhanced with retry logic, cancellation support, and storage paths integration.
"""

import asyncio
import signal
from datetime import datetime

from ..api.database import get_storage_backend
from ..batch.batch_orchestrator import BatchOrchestrator
from ..batch.types import JobStatus
from ..config_env import (
    MAX_CONCURRENT_JOBS,
    MAX_JOB_RETRIES,
    RETRY_DELAY_BASE,
    WORKER_POLL_INTERVAL,
)
from ..storage.base import StorageBackend
from ..storage.config import ensure_job_storage_path
from ..utils.logging_config import get_logger

logger = get_logger("api")


class JobProcessor:
    """Background worker that processes pending jobs from the database.

    v1.3.0: Enhanced with retry logic, cancellation support, and storage paths.
    """

    def __init__(
        self,
        storage_backend: StorageBackend | None = None,
        poll_interval: int | None = None,
        max_concurrent_jobs: int | None = None,
        max_retries: int | None = None,
        retry_delay_base: float | None = None,
    ):
        """Initialize job processor.

        Args:
            storage_backend: Storage backend for job management
            poll_interval: Seconds between database polls (default: env WORKER_POLL_INTERVAL or 5)
            max_concurrent_jobs: Maximum jobs to process simultaneously (default: env MAX_CONCURRENT_JOBS or 2)
            max_retries: Maximum retry attempts for failed jobs (default: env MAX_JOB_RETRIES or 3)
            retry_delay_base: Base delay for exponential backoff (default: env RETRY_DELAY_BASE or 2.0)
        """
        self.storage = storage_backend or get_storage_backend()
        self.poll_interval = poll_interval if poll_interval is not None else WORKER_POLL_INTERVAL
        self.max_concurrent_jobs = max_concurrent_jobs if max_concurrent_jobs is not None else MAX_CONCURRENT_JOBS
        self.max_retries = max_retries if max_retries is not None else MAX_JOB_RETRIES
        self.retry_delay_base = retry_delay_base if retry_delay_base is not None else RETRY_DELAY_BASE
        self.orchestrator = BatchOrchestrator(storage_backend=self.storage)

        self.running = False
        self.processing_jobs: set[str] = set()
        self.cancellation_requests: set[str] = set()  # v1.3.0: Track cancel requests

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False

    def request_cancellation(self, job_id: str) -> None:
        """Request cancellation of a running job (v1.3.0).

        Args:
            job_id: ID of job to cancel
        """
        self.cancellation_requests.add(job_id)
        logger.info(f"[CANCEL] Cancellation requested for job {job_id}")

    def _is_cancellation_requested(self, job_id: str) -> bool:
        """Check if cancellation has been requested for a job (v1.3.0).

        Args:
            job_id: Job ID to check

        Returns:
            True if cancellation requested
        """
        return job_id in self.cancellation_requests

    def _should_retry_job(self, job) -> bool:
        """Determine if a failed job should be retried (v1.3.0).

        Args:
            job: BatchJob instance

        Returns:
            True if job should be retried
        """
        return job.retry_count < self.max_retries

    def _calculate_retry_delay(self, retry_count: int) -> float:
        """Calculate exponential backoff delay (v1.3.0).

        Args:
            retry_count: Current retry attempt number

        Returns:
            Delay in seconds
        """
        return self.retry_delay_base**retry_count

    async def start(self) -> None:
        """Start the background job processor."""
        self.running = True
        logger.info("[START] VideoAnnotator job processor started")
        logger.info(
            f"[CONFIG] Poll interval: {self.poll_interval}s, Max concurrent: {self.max_concurrent_jobs}"
        )

        try:
            while self.running:
                await self._process_cycle()
                await asyncio.sleep(self.poll_interval)

        except Exception as e:
            logger.error(f"Job processor crashed: {e}", exc_info=True)
        finally:
            logger.info("[STOP] Job processor shutting down...")
            # Wait for any running jobs to complete
            if self.processing_jobs:
                logger.info(
                    f"[WAIT] Waiting for {len(self.processing_jobs)} jobs to complete..."
                )
                # In a real implementation, we'd wait for them gracefully
                await asyncio.sleep(2)

    async def _process_cycle(self) -> None:
        """Single processing cycle - check for jobs and process them."""
        try:
            # v1.3.0: Get pending jobs AND retrying jobs from database
            pending_job_ids = self.storage.list_jobs(status_filter="pending")
            retrying_job_ids = self.storage.list_jobs(status_filter="retrying")
            all_processable_ids = pending_job_ids + retrying_job_ids

            if not all_processable_ids:
                logger.debug("No pending or retrying jobs found")
                return

            # v1.3.0 (T065): Count RUNNING jobs to ensure we respect concurrent limit
            running_job_ids = self.storage.list_jobs(status_filter="running")
            
            # Remove any completed jobs from our tracking
            self.processing_jobs = {
                job_id
                for job_id in self.processing_jobs
                if job_id in all_processable_ids or job_id in running_job_ids
            }

            # v1.3.0 (T065): Determine how many new jobs we can start
            # Count both tracked jobs and database RUNNING jobs to avoid race conditions
            current_running_count = len(running_job_ids)
            available_slots = self.max_concurrent_jobs - current_running_count

            # v1.3.0 (T065): Log queue status and skip if at limit
            if available_slots <= 0:
                pending_count = len(pending_job_ids)
                logger.info(
                    f"[QUEUE] At concurrent job limit ({current_running_count}/{self.max_concurrent_jobs}), "
                    f"{pending_count} jobs waiting in queue"
                )
                return

            # Select jobs to process
            jobs_to_process = []
            for job_id in all_processable_ids[:available_slots]:
                if job_id not in self.processing_jobs:
                    try:
                        job = self.storage.load_job_metadata(job_id)
                        jobs_to_process.append(job)
                        self.processing_jobs.add(job_id)
                    except Exception as e:
                        # Include full traceback to aid debugging of import errors
                        logger.error(f"Failed to load job {job_id}: {e}", exc_info=True)

            # Start processing selected jobs
            tasks = []
            for job in jobs_to_process:
                task = asyncio.create_task(self._process_job_async(job))
                tasks.append(task)  # Store reference as required by RUF006
                retry_info = (
                    f" (retry {job.retry_count}/{self.max_retries})"
                    if job.retry_count > 0
                    else ""
                )
                logger.info(
                    f"[START] Processing job {job.job_id}{retry_info} ({job.video_path})"
                )

        except Exception as e:
            logger.error(f"Error in processing cycle: {e}", exc_info=True)

    async def _process_job_async(self, job):
        """Process a single job asynchronously (v1.3.0: enhanced)."""
        job_id = job.job_id

        try:
            # v1.3.0: Check for cancellation before starting
            if self._is_cancellation_requested(job_id):
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.now()
                job.error_message = "Job cancelled by user"
                self.storage.save_job_metadata(job)
                self.cancellation_requests.discard(job_id)
                logger.info(f"[CANCELLED] Job {job_id} cancelled before processing")
                return

            # v1.3.0: Ensure storage directory exists
            if job.storage_path:
                try:
                    ensure_job_storage_path(job_id)
                    logger.debug(f"[STORAGE] Created directory for job {job_id}")
                except Exception as e:
                    logger.warning(f"Failed to create storage directory: {e}")

            # Update job status to running
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            self.storage.save_job_metadata(job)

            logger.info(f"[PROCESS] Starting job {job_id}")

            # Process the job using BatchOrchestrator
            # Run in thread pool since orchestrator is sync
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                None, self._run_single_job_processing, job
            )

            # v1.3.0: Check for cancellation after processing
            if self._is_cancellation_requested(job_id):
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.now()
                job.error_message = "Job cancelled during processing"
                self.storage.save_job_metadata(job)
                self.cancellation_requests.discard(job_id)
                logger.info(f"[CANCELLED] Job {job_id} cancelled after processing")
                return

            if success:
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.now()
                logger.info(f"[SUCCESS] Completed job {job_id}")
            else:
                # v1.3.0: Retry logic
                if self._should_retry_job(job):
                    job.retry_count += 1
                    job.status = JobStatus.RETRYING
                    retry_delay = self._calculate_retry_delay(job.retry_count)
                    logger.warning(
                        f"[RETRY] Job {job_id} will retry ({job.retry_count}/{self.max_retries}) "
                        f"after {retry_delay:.1f}s delay"
                    )
                    # Schedule retry (just set status, next cycle will pick it up)
                    await asyncio.sleep(retry_delay)
                    job.status = JobStatus.PENDING  # Reset to pending for retry
                else:
                    job.status = JobStatus.FAILED
                    job.completed_at = datetime.now()
                    if not job.error_message:
                        job.error_message = "Processing failed - max retries exceeded"
                    logger.error(
                        f"[FAILED] Job {job_id} failed after {job.retry_count} retries"
                    )

            # Save final status
            self.storage.save_job_metadata(job)

        except Exception as e:
            # Handle job failure
            try:
                # v1.3.0: Retry logic on exception
                if self._should_retry_job(job):
                    job.retry_count += 1
                    job.status = JobStatus.RETRYING
                    job.error_message = f"Attempt {job.retry_count} failed: {e}"
                    logger.warning(
                        f"[RETRY] Job {job_id} exception, will retry ({job.retry_count}/{self.max_retries})"
                    )
                    await asyncio.sleep(self._calculate_retry_delay(job.retry_count))
                    job.status = JobStatus.PENDING  # Reset for retry
                else:
                    job.status = JobStatus.FAILED
                    job.error_message = str(e)
                    job.completed_at = datetime.now()
                    logger.error(
                        f"[ERROR] Job {job_id} failed: {e} (max retries exceeded)",
                        exc_info=True,
                    )

                self.storage.save_job_metadata(job)
            except Exception as save_error:
                logger.error(
                    f"Failed to save error state for job {job_id}: {save_error}"
                )

        finally:
            # Remove from processing set
            self.processing_jobs.discard(job_id)
            # v1.3.0: Clean up cancellation request if present
            self.cancellation_requests.discard(job_id)

    def _run_single_job_processing(self, job) -> bool:
        """Run the actual job processing using existing pipeline infrastructure.

        This is a synchronous method that runs in a thread executor.
        """
        try:
            # Create a minimal batch with just this job
            self.orchestrator.add_job(job)

            # Use BatchOrchestrator to process
            results = self.orchestrator.run_batch(
                max_workers=1,  # Single job processing
                save_checkpoints=False,
            )

            # Check if processing was successful
            if results and len(results.failed_jobs) == 0:
                logger.info(f"Job {job.job_id} processed successfully")
                return True
            else:
                if results and results.failed_jobs:
                    failed_job = results.failed_jobs[0]
                    job.error_message = failed_job.error_message
                logger.error(f"Job {job.job_id} processing failed")
                return False

        except Exception as e:
            logger.error(f"Exception during job processing: {e}", exc_info=True)
            job.error_message = str(e)
            return False

    def stop(self):
        """Stop the job processor."""
        self.running = False


async def run_job_processor(
    poll_interval: int = 5,
    max_concurrent_jobs: int = 2,
    storage_backend: StorageBackend | None = None,
):
    """Entry point for running the job processor.

    Args:
        poll_interval: Seconds between database polls
        max_concurrent_jobs: Maximum concurrent jobs
        storage_backend: Optional storage backend
    """
    processor = JobProcessor(
        storage_backend=storage_backend,
        poll_interval=poll_interval,
        max_concurrent_jobs=max_concurrent_jobs,
    )

    await processor.start()


if __name__ == "__main__":
    # Allow direct execution for testing
    asyncio.run(run_job_processor())
