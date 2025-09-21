"""
Progress tracking for VideoAnnotator batch processing.

Provides real-time progress monitoring, ETA calculation, and performance metrics.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import deque

from .types import BatchJob, JobStatus, BatchStatus


class ProgressTracker:
    """Tracks progress and performance metrics for batch processing."""
    
    def __init__(self, max_history: int = 100):
        """
        Initialize progress tracker.
        
        Args:
            max_history: Maximum number of completed jobs to keep for averaging
        """
        self.max_history = max_history
        self.start_time: Optional[datetime] = None
        self.completion_history: deque = deque(maxlen=max_history)
        self.current_jobs: Dict[str, datetime] = {}  # job_id -> start_time
        self.logger = logging.getLogger(__name__)
        
        # Performance metrics
        self.total_processing_time = 0.0
        self.jobs_completed = 0
        self.jobs_failed = 0
    
    def start_batch(self) -> None:
        """Mark the start of batch processing."""
        self.start_time = datetime.now()
        self.completion_history.clear()
        self.current_jobs.clear()
        self.total_processing_time = 0.0
        self.jobs_completed = 0
        self.jobs_failed = 0
        self.logger.info("Batch processing started")
    
    def start_job(self, job_id: str) -> None:
        """Mark the start of a job."""
        self.current_jobs[job_id] = datetime.now()
        self.logger.debug(f"Started tracking job {job_id}")
    
    def complete_job(self, job: BatchJob) -> None:
        """Mark the completion of a job."""
        if job.job_id in self.current_jobs:
            start_time = self.current_jobs.pop(job.job_id)
            duration = (datetime.now() - start_time).total_seconds()
            
            # Add to history for averaging
            self.completion_history.append({
                'job_id': job.job_id,
                'duration': duration,
                'completed_at': datetime.now(),
                'status': job.status
            })
            
            # Update metrics
            if job.status == JobStatus.COMPLETED:
                self.jobs_completed += 1
                self.total_processing_time += duration
            elif job.status == JobStatus.FAILED:
                self.jobs_failed += 1
            
            self.logger.debug(f"Completed job {job.job_id} in {duration:.2f}s")
    
    def get_status(self, jobs: List[BatchJob]) -> BatchStatus:
        """Get current batch status."""
        status = BatchStatus()
        
        # Count jobs by status
        for job in jobs:
            status.total_jobs += 1
            if job.status == JobStatus.PENDING:
                status.pending_jobs += 1
            elif job.status == JobStatus.RUNNING:
                status.running_jobs += 1
            elif job.status == JobStatus.COMPLETED:
                status.completed_jobs += 1
            elif job.status == JobStatus.FAILED:
                status.failed_jobs += 1
            elif job.status == JobStatus.CANCELLED:
                status.cancelled_jobs += 1
        
        # Calculate timing metrics
        status.total_processing_time = self.total_processing_time
        status.current_jobs = list(self.current_jobs.keys())
        
        # Calculate average processing time
        if self.completion_history:
            completed_durations = [
                entry['duration'] for entry in self.completion_history
                if entry['status'] == JobStatus.COMPLETED
            ]
            if completed_durations:
                status.average_processing_time = sum(completed_durations) / len(completed_durations)
        
        # Estimate completion time
        if status.average_processing_time > 0 and status.pending_jobs > 0:
            remaining_time = status.average_processing_time * status.pending_jobs
            status.estimated_completion_time = datetime.now() + timedelta(seconds=remaining_time)
        
        return status
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        metrics = {
            'batch_start_time': self.start_time.isoformat() if self.start_time else None,
            'batch_duration': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'jobs_completed': self.jobs_completed,
            'jobs_failed': self.jobs_failed,
            'total_processing_time': self.total_processing_time,
            'average_processing_time': 0.0,
            'current_jobs_count': len(self.current_jobs),
            'throughput_jobs_per_hour': 0.0,
        }
        
        # Calculate averages and throughput
        if self.completion_history:
            completed_durations = [
                entry['duration'] for entry in self.completion_history
                if entry['status'] == JobStatus.COMPLETED
            ]
            if completed_durations:
                metrics['average_processing_time'] = sum(completed_durations) / len(completed_durations)
        
        # Calculate throughput
        if self.start_time and self.jobs_completed > 0:
            elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600
            if elapsed_hours > 0:
                metrics['throughput_jobs_per_hour'] = self.jobs_completed / elapsed_hours
        
        return metrics
    
    def get_eta_report(self, jobs: List[BatchJob]) -> Dict[str, Any]:
        """Get detailed ETA report."""
        status = self.get_status(jobs)
        
        report = {
            'progress_percentage': status.progress_percentage,
            'jobs_remaining': status.pending_jobs,
            'estimated_completion_time': None,
            'estimated_remaining_seconds': None,
            'current_processing_rate': None,
        }
        
        # Calculate ETA based on recent performance
        if status.average_processing_time > 0 and status.pending_jobs > 0:
            # Use recent history for more accurate estimates
            recent_entries = list(self.completion_history)[-10:]  # Last 10 jobs
            if recent_entries:
                recent_durations = [
                    entry['duration'] for entry in recent_entries
                    if entry['status'] == JobStatus.COMPLETED
                ]
                if recent_durations:
                    recent_avg = sum(recent_durations) / len(recent_durations)
                    remaining_seconds = recent_avg * status.pending_jobs
                    
                    report['estimated_remaining_seconds'] = remaining_seconds
                    report['estimated_completion_time'] = (
                        datetime.now() + timedelta(seconds=remaining_seconds)
                    ).isoformat()
                    report['current_processing_rate'] = f"{recent_avg:.1f}s per job"
        
        return report
    
    def log_progress(self, jobs: List[BatchJob]) -> None:
        """Log current progress to console."""
        status = self.get_status(jobs)
        eta_report = self.get_eta_report(jobs)
        
        self.logger.info(
            f"Progress: {status.progress_percentage:.1f}% "
            f"({status.completed_jobs + status.failed_jobs}/{status.total_jobs} jobs) | "
            f"Running: {status.running_jobs} | "
            f"Success Rate: {status.success_rate:.1f}%"
        )
        
        if eta_report['estimated_completion_time']:
            eta_time = datetime.fromisoformat(eta_report['estimated_completion_time'])
            eta_str = eta_time.strftime("%H:%M:%S")
            self.logger.info(f"ETA: {eta_str} ({eta_report['estimated_remaining_seconds']:.0f}s remaining)")
    
    def export_metrics(self, output_file: str) -> None:
        """Export detailed metrics to JSON file."""
        import json
        
        metrics = {
            'performance_metrics': self.get_performance_metrics(),
            'completion_history': list(self.completion_history),
            'export_time': datetime.now().isoformat(),
        }
        
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        self.logger.info(f"Exported metrics to {output_file}")
