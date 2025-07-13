"""
File-based storage backend for VideoAnnotator batch processing.

This backend organizes data in a hierarchical directory structure:
{base_dir}/
├── jobs/
│   ├── {job_id}/
│   │   ├── job_metadata.json
│   │   ├── scene_detection.json
│   │   ├── person_tracking.json  
│   │   ├── face_analysis.json
│   │   ├── audio_processing.json
│   │   └── batch_summary.json
│   └── ...
└── batch_queue.json  # Current batch state
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base import StorageBackend
from ..batch.types import BatchJob, JobStatus


class FileStorageBackend(StorageBackend):
    """File-based storage backend using JSON files."""
    
    def __init__(self, base_dir: Path):
        """
        Initialize file storage backend.
        
        Args:
            base_dir: Base directory for storing all batch data
        """
        self.base_dir = Path(base_dir)
        self.jobs_dir = self.base_dir / "jobs"
        self.batch_queue_file = self.base_dir / "batch_queue.json"
        
        # Create directory structure
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    def _get_job_dir(self, job_id: str) -> Path:
        """Get directory path for a specific job."""
        return self.jobs_dir / job_id
    
    def _get_annotation_file(self, job_id: str, pipeline: str) -> Path:
        """Get file path for pipeline annotations."""
        return self._get_job_dir(job_id) / f"{pipeline}.json"
    
    def _get_metadata_file(self, job_id: str) -> Path:
        """Get file path for job metadata."""
        return self._get_job_dir(job_id) / "job_metadata.json"
    
    def save_annotations(self, job_id: str, pipeline: str, annotations: List[Dict[str, Any]]) -> str:
        """Save pipeline annotations for a job."""
        job_dir = self._get_job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        
        annotation_file = self._get_annotation_file(job_id, pipeline)
        
        # Add metadata to annotations
        annotated_data = {
            "metadata": {
                "job_id": job_id,
                "pipeline": pipeline,
                "created_at": datetime.now().isoformat(),
                "annotation_count": len(annotations),
            },
            "annotations": annotations
        }
        
        with open(annotation_file, 'w', encoding='utf-8') as f:
            json.dump(annotated_data, f, indent=2, default=str)
        
        self.logger.debug(f"Saved {len(annotations)} annotations for {pipeline} in job {job_id}")
        return str(annotation_file)
    
    def load_annotations(self, job_id: str, pipeline: str) -> List[Dict[str, Any]]:
        """Load pipeline annotations for a job."""
        annotation_file = self._get_annotation_file(job_id, pipeline)
        
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotations not found: {annotation_file}")
        
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both old format (direct list) and new format (with metadata)
        if isinstance(data, list):
            return data
        else:
            return data.get("annotations", [])
    
    def annotation_exists(self, job_id: str, pipeline: str) -> bool:
        """Check if annotations exist for a job and pipeline."""
        annotation_file = self._get_annotation_file(job_id, pipeline)
        return annotation_file.exists()
    
    def save_job_metadata(self, job: BatchJob) -> None:
        """Save job metadata."""
        job_dir = self._get_job_dir(job.job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_file = self._get_metadata_file(job.job_id)
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(job.to_dict(), f, indent=2, default=str)
        
        self.logger.debug(f"Saved metadata for job {job.job_id}")
    
    def load_job_metadata(self, job_id: str) -> BatchJob:
        """Load job metadata."""
        metadata_file = self._get_metadata_file(job_id)
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"Job metadata not found: {metadata_file}")
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return BatchJob.from_dict(data)
    
    def list_jobs(self, status_filter: Optional[str] = None) -> List[str]:
        """List all job IDs, optionally filtered by status."""
        job_ids = []
        
        if not self.jobs_dir.exists():
            return job_ids
        
        for job_dir in self.jobs_dir.iterdir():
            if job_dir.is_dir():
                job_id = job_dir.name
                
                # Apply status filter if specified
                if status_filter:
                    try:
                        job = self.load_job_metadata(job_id)
                        if job.status.value != status_filter:
                            continue
                    except FileNotFoundError:
                        continue
                
                job_ids.append(job_id)
        
        return sorted(job_ids)
    
    def delete_job(self, job_id: str) -> None:
        """Delete all data for a job."""
        job_dir = self._get_job_dir(job_id)
        
        if job_dir.exists():
            import shutil
            shutil.rmtree(job_dir)
            self.logger.info(f"Deleted job {job_id}")
        else:
            self.logger.warning(f"Job directory not found: {job_dir}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            "backend_type": "file",
            "base_dir": str(self.base_dir),
            "total_jobs": 0,
            "jobs_by_status": {},
            "total_size_mb": 0.0,
            "pipelines": set(),
        }
        
        if not self.jobs_dir.exists():
            return stats
        
        total_size = 0
        for job_dir in self.jobs_dir.iterdir():
            if job_dir.is_dir():
                stats["total_jobs"] += 1
                
                # Get job status
                try:
                    job = self.load_job_metadata(job_dir.name)
                    status = job.status.value
                    stats["jobs_by_status"][status] = stats["jobs_by_status"].get(status, 0) + 1
                    
                    # Collect pipeline info
                    stats["pipelines"].update(job.pipeline_results.keys())
                except FileNotFoundError:
                    stats["jobs_by_status"]["unknown"] = stats["jobs_by_status"].get("unknown", 0) + 1
                
                # Calculate directory size
                for file_path in job_dir.rglob("*"):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
        
        stats["total_size_mb"] = total_size / (1024 * 1024)
        stats["pipelines"] = list(stats["pipelines"])
        
        return stats
    
    def save_batch_queue(self, jobs: List[BatchJob]) -> None:
        """Save current batch queue state."""
        queue_data = {
            "updated_at": datetime.now().isoformat(),
            "total_jobs": len(jobs),
            "jobs": [job.to_dict() for job in jobs]
        }
        
        with open(self.batch_queue_file, 'w', encoding='utf-8') as f:
            json.dump(queue_data, f, indent=2, default=str)
    
    def load_batch_queue(self) -> List[BatchJob]:
        """Load batch queue state."""
        if not self.batch_queue_file.exists():
            return []
        
        with open(self.batch_queue_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return [BatchJob.from_dict(job_data) for job_data in data.get("jobs", [])]
    
    def clear_batch_queue(self) -> None:
        """Clear the batch queue file."""
        if self.batch_queue_file.exists():
            self.batch_queue_file.unlink()
    
    def export_job_results(self, job_id: str, output_file: Path) -> None:
        """Export complete job results to a single file."""
        job = self.load_job_metadata(job_id)
        
        export_data = {
            "job": job.to_dict(),
            "annotations": {}
        }
        
        # Load all annotations
        for pipeline_name in job.pipeline_results.keys():
            try:
                annotations = self.load_annotations(job_id, pipeline_name)
                export_data["annotations"][pipeline_name] = annotations
            except FileNotFoundError:
                export_data["annotations"][pipeline_name] = []
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Exported job {job_id} to {output_file}")
