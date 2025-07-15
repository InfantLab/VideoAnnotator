"""
BatchOrchestrator: Core batch processing engine for VideoAnnotator.

Manages job queues, worker pools, progress tracking, and failure recovery
for robust large-scale video processing.
"""

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Union

from .types import BatchJob, JobStatus, BatchReport, BatchStatus, VideoPath, ConfigDict, PipelineResult
from .progress_tracker import ProgressTracker
from .recovery import FailureRecovery, RetryStrategy
from ..storage.base import StorageBackend


class BatchOrchestrator:
    """
    Orchestrates batch processing of videos through VideoAnnotator pipelines.
    
    Features:
    - Parallel processing with configurable worker pools
    - Intelligent job queue management
    - Resume capability for interrupted batches
    - Robust failure recovery with retry mechanisms
    - Real-time progress tracking and ETA calculation
    - Flexible storage backends (files, SQLite, PostgreSQL)
    """
    
    def __init__(self, 
                 storage_backend: Optional[StorageBackend] = None,
                 max_retries: int = 3,
                 retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
                 checkpoint_interval: int = 10):
        """
        Initialize batch orchestrator.
        
        Args:
            storage_backend: Storage backend for annotations and metadata
            max_retries: Maximum retry attempts per failed job
            retry_strategy: Strategy for retry delays
            checkpoint_interval: Save checkpoint every N completed jobs
        """
        if storage_backend is None:
            # Lazy import to avoid circular import
            from ..storage.file_backend import FileStorageBackend
            storage_backend = FileStorageBackend(Path("batch_results"))
        
        self.storage_backend = storage_backend
        self.progress_tracker = ProgressTracker()
        self.failure_recovery = FailureRecovery(
            max_retries=max_retries,
            strategy=retry_strategy
        )
        self.checkpoint_interval = checkpoint_interval
        self.logger = logging.getLogger(__name__)
        
        # Current batch state
        self.batch_id: Optional[str] = None
        self.jobs: List[BatchJob] = []
        self.is_running = False
        self.should_stop = False
        
        # Import pipeline classes
        self._import_pipelines()
    
    def _import_pipelines(self):
        """Import pipeline classes for processing."""
        try:
            from ..pipelines.scene_detection.scene_pipeline import SceneDetectionPipeline
            from ..pipelines.person_tracking.person_pipeline import PersonTrackingPipeline  
            from ..pipelines.face_analysis.face_pipeline import FaceAnalysisPipeline
            from ..pipelines.audio_processing import AudioPipeline
            
            # Import LAION pipelines
            from ..pipelines.face_analysis.laion_face_pipeline import LAIONFacePipeline
            from ..pipelines.audio_processing.laion_voice_pipeline import LAIONVoicePipeline
            
            self.pipeline_classes = {
                'scene_detection': SceneDetectionPipeline,
                'scene': SceneDetectionPipeline,  # Alias for convenience
                'person_tracking': PersonTrackingPipeline,
                'person': PersonTrackingPipeline,  # Alias for convenience  
                'face_analysis': FaceAnalysisPipeline,
                'face': FaceAnalysisPipeline,  # Alias for convenience
                'audio_processing': AudioPipeline,
                'audio': AudioPipeline,  # Alias for convenience
                # LAION emotion analysis pipelines
                'laion_face_analysis': LAIONFacePipeline,
                'laion_face': LAIONFacePipeline,  # Alias for convenience
                'laion_voice_analysis': LAIONVoicePipeline,
                'laion_voice': LAIONVoicePipeline,  # Alias for convenience
            }
            self.logger.debug("Pipeline classes imported successfully")
            
        except ImportError as e:
            self.logger.error(f"Failed to import pipeline classes: {e}")
            self.pipeline_classes = {}
    
    def add_job(self, 
                video_path: VideoPath, 
                output_dir: Optional[VideoPath] = None,
                config: Optional[ConfigDict] = None,
                selected_pipelines: Optional[List[str]] = None) -> str:
        """
        Add a video processing job to the batch queue.
        
        Args:
            video_path: Path to video file
            output_dir: Directory for output files (auto-generated if None)
            config: Pipeline configuration overrides
            selected_pipelines: List of pipeline names to run (all if None)
            
        Returns:
            Job ID for tracking
        """
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
    
    def add_jobs_from_directory(self,
                              input_dir: VideoPath,
                              output_dir: Optional[VideoPath] = None,
                              config: Optional[ConfigDict] = None,
                              selected_pipelines: Optional[List[str]] = None,
                              extensions: List[str] = None) -> List[str]:
        """
        Add multiple jobs from a directory of videos.
        
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
        
        extensions = extensions or ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        
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
                        selected_pipelines=selected_pipelines
                    )
                    job_ids.append(job_id)
        
        self.logger.info(f"Added {len(job_ids)} jobs from directory: {input_dir}")
        return job_ids
    
    def run_batch(self, max_workers: int = 4, save_checkpoints: bool = True) -> BatchReport:
        """
        Execute all jobs in the batch queue.
        
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
        
        self.logger.info(f"Starting batch {self.batch_id} with {len(self.jobs)} jobs and {max_workers} workers")
        
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
            # Submit all pending jobs
            future_to_job: Dict[Future, BatchJob] = {}
            for job in self.jobs:
                if job.status == JobStatus.PENDING:
                    future = executor.submit(self._process_job_with_retry, job)
                    future_to_job[future] = job
                    job.status = JobStatus.RUNNING
                    job.started_at = datetime.now()
                    self.progress_tracker.start_job(job.job_id)
            
            # Process completed jobs
            completed_count = 0
            for future in as_completed(future_to_job):
                if self.should_stop:
                    self.logger.info("Batch processing stopped by user request")
                    break
                
                job = future_to_job[future]
                
                try:
                    # Get result and update job
                    result_job = future.result()
                    job.status = result_job.status
                    job.pipeline_results = result_job.pipeline_results
                    job.completed_at = datetime.now()
                    job.error_message = result_job.error_message
                    
                    # Update progress tracking
                    self.progress_tracker.complete_job(job)
                    
                    # Update report counters
                    if job.status == JobStatus.COMPLETED:
                        report.completed_jobs += 1
                    elif job.status == JobStatus.FAILED:
                        report.failed_jobs += 1
                        if job.error_message:
                            report.errors.append(f"Job {job.job_id}: {job.error_message}")
                    
                    completed_count += 1
                    
                    # Save checkpoint periodically
                    if save_checkpoints and completed_count % self.checkpoint_interval == 0:
                        self._save_checkpoint()
                    
                    # Log progress
                    self.progress_tracker.log_progress(self.jobs)
                    
                except Exception as e:
                    self.logger.error(f"Unexpected error processing job {job.job_id}: {e}")
                    job.status = JobStatus.FAILED
                    job.error_message = str(e)
                    report.failed_jobs += 1
                    report.errors.append(f"Job {job.job_id}: {str(e)}")
        
        # Finalize report
        report.end_time = datetime.now()
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
        """
        Resume batch processing from a checkpoint.
        
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

    def get_job(self, job_id: str) -> Optional[BatchJob]:
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
        """Check if a job should be retried based on failure recovery settings."""
        job = self.get_job(job_id)
        if not job:
            return False
        
        # Simple retry logic - could be expanded to use failure_recovery
        max_retries = 3  # Default max retries
        return job.retry_count < max_retries

    def _process_job_with_retry(self, job: BatchJob) -> BatchJob:
        """
        Process a single job with retry logic.
        
        Args:
            job: Job to process
            
        Returns:
            Updated job with results
        """
        max_attempts = self.failure_recovery.max_retries + 1
        
        for attempt in range(max_attempts):
            try:
                return self._process_single_job(job)
                
            except Exception as e:
                self.logger.error(f"Job {job.job_id} failed (attempt {attempt + 1}/{max_attempts}): {e}")
                
                # Check if we should retry
                if attempt < max_attempts - 1 and self.failure_recovery.should_retry(job, e):
                    # Prepare for retry
                    job = self.failure_recovery.prepare_retry(job, e)
                    
                    # Wait before retry
                    delay = self.failure_recovery.calculate_retry_delay(job)
                    if delay > 0:
                        self.logger.info(f"Waiting {delay:.1f}s before retrying job {job.job_id}")
                        time.sleep(delay)
                else:
                    # Final failure
                    job.status = JobStatus.FAILED
                    job.error_message = str(e)
                    break
        
        return job
    
    def _process_single_job(self, job: BatchJob) -> BatchJob:
        """
        Process a single job through selected pipelines.
        
        Args:
            job: Job to process
            
        Returns:
            Updated job with results
        """
        self.logger.info(f"Processing job {job.job_id}: {job.video_path}")
        
        # Ensure output directory exists
        job.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save job metadata
        self.storage_backend.save_job_metadata(job)
        
        # Determine pipelines to run
        pipelines_to_run = job.selected_pipelines or list(self.pipeline_classes.keys())
        
        # Process each pipeline
        for pipeline_name in pipelines_to_run:
            if self.should_stop:
                job.status = JobStatus.CANCELLED
                return job
            
            try:
                # Skip if already processed (resume case)
                if self.storage_backend.annotation_exists(job.job_id, pipeline_name):
                    self.logger.info(f"Skipping {pipeline_name} for job {job.job_id} (already exists)")
                    continue
                
                # Initialize and run pipeline
                if pipeline_name not in self.pipeline_classes:
                    raise ValueError(f"Unknown pipeline: {pipeline_name}. Available: {list(self.pipeline_classes.keys())}")
                
                pipeline_class = self.pipeline_classes[pipeline_name]
                pipeline_config = job.config.get(pipeline_name, {})
                pipeline = pipeline_class(pipeline_config)
                
                self.logger.info(f"Running {pipeline_name} for job {job.job_id}")
                start_time = datetime.now()
                
                # Run pipeline
                annotations = pipeline.process(
                    video_path=str(job.video_path),
                    output_dir=str(job.output_dir)
                )
                
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                
                # Save annotations
                output_file = self.storage_backend.save_annotations(
                    job.job_id, pipeline_name, annotations
                )
                
                # Record success
                job.pipeline_results[pipeline_name] = PipelineResult(
                    pipeline_name=pipeline_name,
                    status=JobStatus.COMPLETED,
                    start_time=start_time,
                    end_time=end_time,
                    processing_time=processing_time,
                    annotation_count=len(annotations),
                    output_file=Path(output_file)
                )
                
                self.logger.info(
                    f"Completed {pipeline_name} for job {job.job_id} "
                    f"in {processing_time:.2f}s ({len(annotations)} annotations)"
                )
                
            except Exception as e:
                # Handle partial failure
                if self.failure_recovery.handle_partial_failure(job, pipeline_name, e):
                    # Continue with other pipelines
                    continue
                else:
                    # Fail entire job
                    raise e
        
        # Update job status
        failed_pipelines = [
            name for name, result in job.pipeline_results.items()
            if result.status == JobStatus.FAILED
        ]
        
        if failed_pipelines and len(failed_pipelines) == len(pipelines_to_run):
            job.status = JobStatus.FAILED
            job.error_message = f"All pipelines failed: {', '.join(failed_pipelines)}"
        else:
            job.status = JobStatus.COMPLETED
        
        # Save final job metadata
        self.storage_backend.save_job_metadata(job)
        
        return job
    
    def _save_checkpoint(self) -> None:
        """Save current batch state as checkpoint."""
        if self.batch_id:
            # Get storage base directory
            if hasattr(self.storage_backend, 'base_dir'):
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
        if hasattr(self.storage_backend, 'base_dir'):
            base_dir = Path(self.storage_backend.base_dir)
        else:
            base_dir = Path("batch_results")
        
        # Create reports directory
        reports_dir = base_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Save batch report
        report_file = reports_dir / f"batch_report_{self.batch_id}.json"
        with open(report_file, 'w') as f:
            import json
            json.dump(report.to_dict(), f, indent=2, default=str)
        
        # Save recovery statistics
        recovery_file = reports_dir / f"recovery_report_{self.batch_id}.json"
        with open(recovery_file, 'w') as f:
            import json
            json.dump(self.failure_recovery.get_recovery_report(), f, indent=2)
        
        self.logger.info(f"Saved final results: {report_file}")
