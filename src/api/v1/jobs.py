"""
Job management endpoints for VideoAnnotator API
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, status
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
import json
import tempfile
import os

from pydantic import BaseModel, Field
# TODO: Import batch system after fixing dependencies
# from ...batch.batch_orchestrator import BatchOrchestrator  
# from ...batch.types import BatchJob, JobStatus

# Temporary mock classes for API development
class MockJobStatus:
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"

class MockBatchJob:
    def __init__(self, video_path, config=None, selected_pipelines=None):
        import uuid
        from datetime import datetime
        self.id = uuid.uuid4()
        self.video_path = video_path
        self.config = config
        self.selected_pipelines = selected_pipelines
        self.status = MockJobStatus.PENDING
        self.created_at = datetime.utcnow()
        self.completed_at = None
        self.error_message = None

class MockBatchOrchestrator:
    def __init__(self):
        self.jobs = []
    
    def add_job(self, video_path, config=None, selected_pipelines=None):
        job = MockBatchJob(video_path, config, selected_pipelines)
        self.jobs.append(job)
        return str(job.id)


router = APIRouter()

# TODO: Replace with proper dependency injection from database
# For now, use a simple in-memory store
_batch_orchestrator = None


def get_batch_orchestrator() -> MockBatchOrchestrator:
    """Get batch orchestrator instance."""
    global _batch_orchestrator
    if _batch_orchestrator is None:
        _batch_orchestrator = MockBatchOrchestrator()
    return _batch_orchestrator


# Pydantic models for API
class JobSubmissionRequest(BaseModel):
    """Request model for job submission."""
    config: Optional[Dict[str, Any]] = Field(default=None, description="Processing configuration")
    selected_pipelines: Optional[List[str]] = Field(default=None, description="Pipelines to run")


class JobResponse(BaseModel):
    """Response model for job information."""
    id: str
    status: str
    video_path: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    selected_pipelines: Optional[List[str]] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class JobListResponse(BaseModel):
    """Response model for job listing."""
    jobs: List[JobResponse]
    total: int
    page: int
    per_page: int


@router.post("/", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
async def submit_job(
    video: UploadFile = File(..., description="Video file to process"),
    config: Optional[str] = Form(None, description="JSON configuration"),
    selected_pipelines: Optional[str] = Form(None, description="Comma-separated pipeline names"),
    orchestrator: MockBatchOrchestrator = Depends(get_batch_orchestrator)
):
    """
    Submit a video processing job.
    
    Args:
        video: Video file to process
        config: Optional JSON configuration string
        selected_pipelines: Optional comma-separated pipeline names
    
    Returns:
        Job information including ID and status
    """
    try:
        # Parse configuration if provided
        parsed_config = None
        if config:
            try:
                parsed_config = json.loads(config)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid JSON configuration"
                )
        
        # Parse selected pipelines if provided
        parsed_pipelines = None
        if selected_pipelines:
            parsed_pipelines = [p.strip() for p in selected_pipelines.split(",") if p.strip()]
        
        # Save uploaded video to temporary file
        # TODO: Implement proper file storage (local/S3)
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, video.filename)
        
        with open(video_path, "wb") as f:
            content = await video.read()
            f.write(content)
        
        # Submit job to batch orchestrator
        job_id = orchestrator.add_job(
            video_path=video_path,
            config=parsed_config,
            selected_pipelines=parsed_pipelines
        )
        
        # Find the job in orchestrator to return details
        job = None
        for batch_job in orchestrator.jobs:
            if str(batch_job.id) == job_id:
                job = batch_job
                break
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Job created but not found in orchestrator"
            )
        
        return JobResponse(
            id=str(job.id),
            status=job.status.value,
            video_path=job.video_path,
            config=job.config,
            selected_pipelines=job.selected_pipelines,
            created_at=job.created_at,
            completed_at=job.completed_at
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit job: {str(e)}"
        )


@router.get("/{job_id}", response_model=JobResponse)
async def get_job_status(
    job_id: str,
    orchestrator: MockBatchOrchestrator = Depends(get_batch_orchestrator)
):
    """
    Get job status and details.
    
    Args:
        job_id: Job ID to query
        
    Returns:
        Job information including current status
    """
    try:
        # Find job in orchestrator
        job = None
        for batch_job in orchestrator.jobs:
            if str(batch_job.id) == job_id:
                job = batch_job
                break
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        return JobResponse(
            id=str(job.id),
            status=job.status.value,
            video_path=job.video_path,
            config=job.config,
            selected_pipelines=job.selected_pipelines,
            created_at=job.created_at,
            completed_at=job.completed_at,
            error_message=getattr(job, 'error_message', None)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job status: {str(e)}"
        )


@router.get("/", response_model=JobListResponse)
async def list_jobs(
    page: int = 1,
    per_page: int = 10,
    status_filter: Optional[str] = None,
    orchestrator: MockBatchOrchestrator = Depends(get_batch_orchestrator)
):
    """
    List jobs with pagination and filtering.
    
    Args:
        page: Page number (1-based)
        per_page: Items per page
        status_filter: Optional status filter
        
    Returns:
        Paginated list of jobs
    """
    try:
        # Get all jobs
        all_jobs = orchestrator.jobs
        
        # Apply status filter if provided
        if status_filter:
            all_jobs = [job for job in all_jobs if job.status.value == status_filter]
        
        # Apply pagination
        total = len(all_jobs)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        page_jobs = all_jobs[start_idx:end_idx]
        
        # Convert to response format
        job_responses = [
            JobResponse(
                id=str(job.id),
                status=job.status.value,
                video_path=job.video_path,
                config=job.config,
                selected_pipelines=job.selected_pipelines,
                created_at=job.created_at,
                completed_at=job.completed_at,
                error_message=getattr(job, 'error_message', None)
            )
            for job in page_jobs
        ]
        
        return JobListResponse(
            jobs=job_responses,
            total=total,
            page=page,
            per_page=per_page
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list jobs: {str(e)}"
        )


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_job(
    job_id: str,
    orchestrator: MockBatchOrchestrator = Depends(get_batch_orchestrator)
):
    """
    Cancel/delete a job.
    
    Args:
        job_id: Job ID to cancel
    """
    try:
        # Find job in orchestrator
        job = None
        for batch_job in orchestrator.jobs:
            if str(batch_job.id) == job_id:
                job = batch_job
                break
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        # TODO: Implement proper job cancellation logic
        # For now, just remove from list if not completed
        if job.status not in [JobStatus.COMPLETED, JobStatus.FAILED]:
            orchestrator.jobs.remove(job)
        
        return
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel job: {str(e)}"
        )