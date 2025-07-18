"""
Types and data classes for batch processing.
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import uuid


class JobStatus(Enum):
    """Status enumeration for batch jobs."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


@dataclass
class PipelineResult:
    """Result from a single pipeline execution."""
    pipeline_name: str
    status: JobStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    processing_time: Optional[float] = None
    annotation_count: Optional[int] = None
    output_file: Optional[Path] = None
    error_message: Optional[str] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Get processing duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return self.processing_time


@dataclass 
class BatchJob:
    """Represents a single video processing job in a batch."""
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    video_path: Path = None
    output_dir: Path = None
    config: Dict[str, Any] = field(default_factory=dict)
    status: JobStatus = JobStatus.PENDING
    pipeline_results: Dict[str, PipelineResult] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    error_message: Optional[str] = None
    selected_pipelines: Optional[List[str]] = None
    
    @property
    def video_id(self) -> str:
        """Get video identifier from filename."""
        return self.video_path.stem if self.video_path else "unknown"
    
    @property
    def duration(self) -> Optional[float]:
        """Get total job duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def is_complete(self) -> bool:
        """Check if job is in a terminal state."""
        return self.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "job_id": self.job_id,
            "video_path": str(self.video_path) if self.video_path else None,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "config": self.config,
            "status": self.status.value,
            "pipeline_results": {
                name: {
                    "pipeline_name": result.pipeline_name,
                    "status": result.status.value,
                    "start_time": result.start_time.isoformat() if result.start_time else None,
                    "end_time": result.end_time.isoformat() if result.end_time else None,
                    "processing_time": result.processing_time,
                    "annotation_count": result.annotation_count,
                    "output_file": str(result.output_file) if result.output_file else None,
                    "error_message": result.error_message,
                }
                for name, result in self.pipeline_results.items()
            },
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retry_count": self.retry_count,
            "error_message": self.error_message,
            "selected_pipelines": self.selected_pipelines,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchJob':
        """Create from dictionary representation."""
        # Parse pipeline results
        pipeline_results = {}
        for name, result_data in data.get("pipeline_results", {}).items():
            pipeline_results[name] = PipelineResult(
                pipeline_name=result_data["pipeline_name"],
                status=JobStatus(result_data["status"]),
                start_time=datetime.fromisoformat(result_data["start_time"]) if result_data["start_time"] else None,
                end_time=datetime.fromisoformat(result_data["end_time"]) if result_data["end_time"] else None,
                processing_time=result_data["processing_time"],
                annotation_count=result_data["annotation_count"],
                output_file=Path(result_data["output_file"]) if result_data["output_file"] else None,
                error_message=result_data["error_message"],
            )
        
        return cls(
            job_id=data["job_id"],
            video_path=Path(data["video_path"]) if data["video_path"] else None,
            output_dir=Path(data["output_dir"]) if data["output_dir"] else None,
            config=data.get("config", {}),
            status=JobStatus(data["status"]),
            pipeline_results=pipeline_results,
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data["started_at"] else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data["completed_at"] else None,
            retry_count=data.get("retry_count", 0),
            error_message=data.get("error_message"),
            selected_pipelines=data.get("selected_pipelines"),
        )


@dataclass
class BatchStatus:
    """Current status of a batch processing operation."""
    total_jobs: int = 0
    pending_jobs: int = 0
    running_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    cancelled_jobs: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    estimated_completion_time: Optional[datetime] = None
    current_jobs: List[str] = field(default_factory=list)  # Currently running job IDs
    
    @property
    def progress_percentage(self) -> float:
        """Get progress as percentage (0-100)."""
        if self.total_jobs == 0:
            return 0.0
        completed = self.completed_jobs + self.failed_jobs + self.cancelled_jobs
        return (completed / self.total_jobs) * 100.0
    
    @property
    def success_rate(self) -> float:
        """Get success rate as percentage (0-100)."""
        completed = self.completed_jobs + self.failed_jobs + self.cancelled_jobs
        if completed == 0:
            return 0.0
        return (self.completed_jobs / completed) * 100.0


@dataclass
class BatchReport:
    """Final report from a batch processing operation."""
    batch_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    cancelled_jobs: int = 0
    total_processing_time: float = 0.0
    jobs: List[BatchJob] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    @property
    def duration(self) -> Optional[float]:
        """Get total batch duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def success_rate(self) -> float:
        """Get success rate as percentage (0-100)."""
        if self.total_jobs == 0:
            return 0.0
        return (self.completed_jobs / self.total_jobs) * 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "batch_id": self.batch_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_jobs": self.total_jobs,
            "completed_jobs": self.completed_jobs,
            "failed_jobs": self.failed_jobs,
            "cancelled_jobs": self.cancelled_jobs,
            "total_processing_time": self.total_processing_time,
            "duration": self.duration,
            "success_rate": self.success_rate,
            "jobs": [job.to_dict() for job in self.jobs],
            "errors": self.errors,
        }


# Type aliases for convenience
VideoPath = Union[str, Path]
ConfigDict = Dict[str, Any]
