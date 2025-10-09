"""
Background job processor for VideoAnnotator API server.

Continuously polls the database for pending jobs and processes them
using the BatchOrchestrator system.
"""

import asyncio
import logging
import time
from typing import Optional
from pathlib import Path
import signal
import sys

from ..batch.batch_orchestrator import BatchOrchestrator
from ..batch.types import JobStatus
from ..storage.base import StorageBackend
from ..api.database import get_storage_backend
from ..utils.logging_config import get_logger

logger = get_logger("api")

class JobProcessor:
    """Background worker that processes pending jobs from the database."""
    
    def __init__(self, 
                 storage_backend: Optional[StorageBackend] = None,
                 poll_interval: int = 5,
                 max_concurrent_jobs: int = 2):
        """
        Initialize job processor.
        
        Args:
            storage_backend: Storage backend for job management
            poll_interval: Seconds between database polls
            max_concurrent_jobs: Maximum jobs to process simultaneously
        """
        self.storage = storage_backend or get_storage_backend()
        self.poll_interval = poll_interval
        self.max_concurrent_jobs = max_concurrent_jobs
        self.orchestrator = BatchOrchestrator(storage_backend=self.storage)
        
        self.running = False
        self.processing_jobs = set()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
    
    async def start(self):
        """Start the background job processor."""
        self.running = True
        logger.info(f"[START] VideoAnnotator job processor started")
        logger.info(f"[CONFIG] Poll interval: {self.poll_interval}s, Max concurrent: {self.max_concurrent_jobs}")
        
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
                logger.info(f"[WAIT] Waiting for {len(self.processing_jobs)} jobs to complete...")
                # In a real implementation, we'd wait for them gracefully
                await asyncio.sleep(2)
    
    async def _process_cycle(self):
        """Single processing cycle - check for jobs and process them."""
        try:
            # Get pending jobs from database
            pending_job_ids = self.storage.list_jobs(status_filter="pending")
            
            if not pending_job_ids:
                logger.debug("No pending jobs found")
                return
            
            # Remove any completed jobs from our tracking
            self.processing_jobs = {job_id for job_id in self.processing_jobs 
                                  if job_id in pending_job_ids}
            
            # Determine how many new jobs we can start
            available_slots = self.max_concurrent_jobs - len(self.processing_jobs)
            
            if available_slots <= 0:
                logger.debug(f"All {self.max_concurrent_jobs} slots busy, waiting...")
                return
            
            # Select jobs to process
            jobs_to_process = []
            for job_id in pending_job_ids[:available_slots]:
                if job_id not in self.processing_jobs:
                    try:
                        job = self.storage.load_job_metadata(job_id)
                        jobs_to_process.append(job)
                        self.processing_jobs.add(job_id)
                    except Exception as e:
                        # Include full traceback to aid debugging of import errors
                        logger.error(f"Failed to load job {job_id}: {e}", exc_info=True)
            
            # Start processing selected jobs
            for job in jobs_to_process:
                asyncio.create_task(self._process_job_async(job))
                logger.info(f"[START] Processing job {job.job_id} ({job.video_path})")
        
        except Exception as e:
            logger.error(f"Error in processing cycle: {e}", exc_info=True)
    
    async def _process_job_async(self, job):
        """Process a single job asynchronously."""
        job_id = job.job_id
        
        try:
            # Update job status to running
            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            self.storage.save_job_metadata(job)
            
            logger.info(f"[PROCESS] Starting job {job_id}")
            
            # Process the job using BatchOrchestrator
            # Run in thread pool since orchestrator is sync
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                None, 
                self._run_single_job_processing,
                job
            )
            
            if success:
                job.status = JobStatus.COMPLETED
                job.completed_at = time.time()
                logger.info(f"[SUCCESS] Completed job {job_id}")
            else:
                job.status = JobStatus.FAILED
                job.error_message = "Processing failed - check logs"
                logger.error(f"[FAILED] Job {job_id} processing failed")
            
            # Save final status
            self.storage.save_job_metadata(job)
            
        except Exception as e:
            # Handle job failure
            try:
                job.status = JobStatus.FAILED
                job.error_message = str(e)
                job.completed_at = time.time()
                self.storage.save_job_metadata(job)
                logger.error(f"[ERROR] Job {job_id} failed: {e}", exc_info=True)
            except Exception as save_error:
                logger.error(f"Failed to save error state for job {job_id}: {save_error}")
        
        finally:
            # Remove from processing set
            self.processing_jobs.discard(job_id)
    
    def _run_single_job_processing(self, job) -> bool:
        """
        Run the actual job processing using existing pipeline infrastructure.
        
        This is a synchronous method that runs in a thread executor.
        """
        try:
            # Create a minimal batch with just this job
            jobs = [job]
            
            # Use BatchOrchestrator to process
            results = self.orchestrator.run_batch(
                jobs=jobs,
                max_workers=1,  # Single job processing
                save_checkpoints=False
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
    storage_backend: Optional[StorageBackend] = None
):
    """
    Entry point for running the job processor.
    
    Args:
        poll_interval: Seconds between database polls
        max_concurrent_jobs: Maximum concurrent jobs
        storage_backend: Optional storage backend
    """
    processor = JobProcessor(
        storage_backend=storage_backend,
        poll_interval=poll_interval,
        max_concurrent_jobs=max_concurrent_jobs
    )
    
    await processor.start()

if __name__ == "__main__":
    # Allow direct execution for testing
    asyncio.run(run_job_processor())