"""
Background task management for VideoAnnotator API server.

Integrates job processing directly into the API server using FastAPI's
background tasks and asyncio for seamless operation.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Optional, Set
from contextlib import asynccontextmanager

from api.job_processor import JobProcessor
from batch.types import JobStatus
from storage.base import StorageBackend
from api.database import get_storage_backend
from utils.logging_config import get_logger

logger = get_logger("api")

class BackgroundJobManager:
    """
    Manages background job processing within the API server.
    
    Runs as a background task that continuously polls for pending jobs
    and processes them using the existing pipeline infrastructure.
    """
    
    def __init__(self, 
                 storage_backend: Optional[StorageBackend] = None,
                 poll_interval: int = 5,
                 max_concurrent_jobs: int = 2):
        """
        Initialize the background job manager.
        
        Args:
            storage_backend: Storage backend for job management
            poll_interval: Seconds between database polls
            max_concurrent_jobs: Maximum jobs to process simultaneously
        """
        self.storage = storage_backend or get_storage_backend()
        self.poll_interval = poll_interval
        self.max_concurrent_jobs = max_concurrent_jobs
        self.job_processor = JobProcessor()
        
        self.running = False
        self.processing_jobs: Set[str] = set()
        self.background_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the background job processing."""
        if self.running:
            logger.warning("Background job manager is already running")
            return
        
        self.running = True
        self.background_task = asyncio.create_task(self._job_processing_loop())
        logger.info(f"[START] Background job processing started (poll: {self.poll_interval}s, max concurrent: {self.max_concurrent_jobs})")
    
    async def stop(self):
        """Stop the background job processing gracefully."""
        if not self.running:
            return
        
        logger.info("[STOP] Stopping background job processing...")
        self.running = False
        
        if self.background_task:
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                pass
        
        # Wait for any ongoing jobs to complete (with timeout)
        if self.processing_jobs:
            logger.info(f"[WAIT] Waiting for {len(self.processing_jobs)} jobs to complete...")
            wait_time = 0
            while self.processing_jobs and wait_time < 30:
                await asyncio.sleep(1)
                wait_time += 1
            
            if self.processing_jobs:
                logger.warning(f"[TIMEOUT] {len(self.processing_jobs)} jobs still running after shutdown timeout")
        
        logger.info("[STOP] Background job processing stopped")
    
    async def _job_processing_loop(self):
        """Main background processing loop."""
        try:
            while self.running:
                await self._process_cycle()
                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            logger.info("Background job processing loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Background job processing loop crashed: {e}", exc_info=True)
            # Restart the loop after a brief delay
            if self.running:
                await asyncio.sleep(10)
                self.background_task = asyncio.create_task(self._job_processing_loop())
    
    async def _process_cycle(self):
        """Single processing cycle - check for jobs and process them."""
        try:
            # Get pending jobs from database
            pending_job_ids = self.storage.list_jobs(status_filter="pending")
            
            if not pending_job_ids:
                logger.debug("No pending jobs found")
                return
            
            # Clean up completed jobs from our tracking
            completed_jobs = self.processing_jobs - set(pending_job_ids)
            for job_id in completed_jobs:
                self.processing_jobs.discard(job_id)
                logger.debug(f"Job {job_id} completed, removed from tracking")
            
            # Determine how many new jobs we can start
            available_slots = self.max_concurrent_jobs - len(self.processing_jobs)
            
            if available_slots <= 0:
                logger.debug(f"All {self.max_concurrent_jobs} slots busy with jobs: {list(self.processing_jobs)}")
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
                        logger.error(f"Failed to load job {job_id}: {e}")
            
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
            job.started_at = datetime.fromtimestamp(time.time())
            self.storage.save_job_metadata(job)
            
            logger.info(f"[PROCESS] Starting job {job_id}")
            
            # Process the job using BatchOrchestrator in thread pool
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                None, 
                self._run_single_job_processing,
                job
            )
            
            if success:
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.fromtimestamp(time.time())
                logger.info(f"[SUCCESS] Completed job {job_id}")
            else:
                job.status = JobStatus.FAILED
                job.error_message = job.error_message or "Processing failed - check logs"
                job.completed_at = datetime.fromtimestamp(time.time())
                logger.error(f"[FAILED] Job {job_id} processing failed")
            
            # Save final status
            self.storage.save_job_metadata(job)
            
        except Exception as e:
            # Handle job failure
            try:
                job.status = JobStatus.FAILED
                job.error_message = str(e)
                job.completed_at = datetime.fromtimestamp(time.time())
                self.storage.save_job_metadata(job)
                logger.error(f"[ERROR] Job {job_id} failed: {e}", exc_info=True)
            except Exception as save_error:
                logger.error(f"Failed to save error state for job {job_id}: {save_error}")
        
        finally:
            # Remove from processing set
            self.processing_jobs.discard(job_id)
    
    def _run_single_job_processing(self, job) -> bool:
        """
        Run the actual job processing using JobProcessor.
        
        This is a synchronous method that runs in a thread executor.
        """
        try:
            logger.info(f"Starting pipeline processing for job {job.job_id}")
            
            # Use JobProcessor to process the single job
            success = self.job_processor.process_job(job)
            
            if success:
                logger.info(f"Job {job.job_id} processed successfully")
                return True
            else:
                logger.error(f"Job {job.job_id} failed: {job.error_message}")
                return False
                
        except Exception as e:
            logger.error(f"Exception during job processing: {e}", exc_info=True)
            job.error_message = str(e)
            return False
    
    def get_status(self) -> dict:
        """Get current status of the background job manager."""
        return {
            "running": self.running,
            "processing_jobs": list(self.processing_jobs),
            "concurrent_jobs": len(self.processing_jobs),
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "poll_interval": self.poll_interval
        }

# Global background job manager instance
_background_manager: Optional[BackgroundJobManager] = None

def get_background_manager() -> BackgroundJobManager:
    """Get the global background job manager instance."""
    global _background_manager
    if _background_manager is None:
        _background_manager = BackgroundJobManager()
    return _background_manager

async def start_background_processing():
    """Start background job processing."""
    manager = get_background_manager()
    await manager.start()

async def stop_background_processing():
    """Stop background job processing."""
    manager = get_background_manager()
    await manager.stop()

@asynccontextmanager
async def background_job_lifespan():
    """Context manager for background job processing lifecycle."""
    try:
        await start_background_processing()
        yield
    finally:
        await stop_background_processing()