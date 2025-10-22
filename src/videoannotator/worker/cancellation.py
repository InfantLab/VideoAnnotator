"""Cancellation management for job processing.

Handles job cancellation requests and coordinates with JobProcessor
for graceful shutdown and resource cleanup.

v1.3.0: Initial implementation for US2 (Stop Runaway Jobs).
"""

import asyncio

from .utils.logging_config import get_logger

logger = get_logger("api")


class CancellationManager:
    """Manages job cancellation requests and cleanup.

    Coordinates with JobProcessor to handle cancellation requests,
    ensuring graceful shutdown and resource cleanup (including GPU memory).

    Note: Current implementation uses asyncio tasks. Future versions may
    support multiprocessing with SIGTERM/SIGKILL escalation.
    """

    def __init__(self):
        """Initialize cancellation manager."""
        self.running_tasks: dict[str, asyncio.Task] = {}
        self.cancellation_requests: set[str] = set()
        logger.debug("[CANCELLATION] CancellationManager initialized")

    def register_task(self, job_id: str, task: asyncio.Task) -> None:
        """Register a running task for potential cancellation.

        Args:
            job_id: Job ID
            task: Asyncio task processing the job
        """
        self.running_tasks[job_id] = task
        logger.debug(f"[CANCELLATION] Registered task for job {job_id}")

    def unregister_task(self, job_id: str) -> None:
        """Unregister a completed task.

        Args:
            job_id: Job ID to unregister
        """
        if job_id in self.running_tasks:
            del self.running_tasks[job_id]
            logger.debug(f"[CANCELLATION] Unregistered task for job {job_id}")

    async def cancel_job(self, job_id: str, timeout: float = 5.0) -> bool:
        """Request cancellation of a running job.

        Args:
            job_id: ID of job to cancel
            timeout: Maximum seconds to wait for cancellation (default: 5.0)

        Returns:
            True if job was cancelled successfully, False otherwise
        """
        self.cancellation_requests.add(job_id)
        logger.info(f"[CANCELLATION] Requested cancellation for job {job_id}")

        # If job is currently running as a task, cancel it
        task = self.running_tasks.get(job_id)
        if task and not task.done():
            logger.info(f"[CANCELLATION] Cancelling active task for job {job_id}")
            task.cancel()

            try:
                # Wait for task to acknowledge cancellation
                await asyncio.wait_for(task, timeout=timeout)
                return True
            except asyncio.CancelledError:
                logger.info(
                    f"[CANCELLATION] Task for job {job_id} cancelled successfully"
                )
                return True
            except TimeoutError:
                logger.warning(
                    f"[CANCELLATION] Task for job {job_id} did not cancel within {timeout}s"
                )
                # Force cancellation
                task.cancel()
                return False
            except Exception as e:
                logger.error(f"[CANCELLATION] Error cancelling job {job_id}: {e}")
                return False
        else:
            # Job not currently running (may be queued)
            logger.info(
                f"[CANCELLATION] Job {job_id} not actively running (queued or completed)"
            )
            return True

    def is_cancellation_requested(self, job_id: str) -> bool:
        """Check if cancellation has been requested for a job.

        Args:
            job_id: Job ID to check

        Returns:
            True if cancellation has been requested
        """
        return job_id in self.cancellation_requests

    def clear_cancellation_request(self, job_id: str) -> None:
        """Clear cancellation request after handling.

        Args:
            job_id: Job ID to clear
        """
        self.cancellation_requests.discard(job_id)
        logger.debug(f"[CANCELLATION] Cleared cancellation request for job {job_id}")

    def cleanup(self) -> None:
        """Clean up all resources and cancel remaining tasks."""
        logger.info("[CANCELLATION] Cleaning up cancellation manager")

        # Cancel all running tasks
        for job_id, task in list(self.running_tasks.items()):
            if not task.done():
                logger.info(f"[CANCELLATION] Cancelling task for job {job_id}")
                task.cancel()

        self.running_tasks.clear()
        self.cancellation_requests.clear()
        logger.info("[CANCELLATION] Cleanup complete")

    def get_running_job_ids(self) -> set[str]:
        """Get set of currently running job IDs.

        Returns:
            Set of job IDs currently being processed
        """
        return set(self.running_tasks.keys())
