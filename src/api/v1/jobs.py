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
from pathlib import Path

from ...batch.types import BatchJob, JobStatus
from ...storage.base import StorageBackend
from ..database import get_storage_backend


router = APIRouter()

def get_storage() -> StorageBackend:
    """Get storage backend for job management."""
    return get_storage_backend()


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
    storage: StorageBackend = Depends(get_storage)
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
        
        # Create BatchJob instance
        batch_job = BatchJob(
            video_path=Path(video_path),
            output_dir=None,  # Will be set by processing system
            config=parsed_config or {},
            status=JobStatus.PENDING,
            selected_pipelines=parsed_pipelines
        )
        
        # Save job to database
        storage.save_job_metadata(batch_job)
        
        return JobResponse(
            id=batch_job.job_id,
            status=batch_job.status.value,
            video_path=str(batch_job.video_path),
            config=batch_job.config,
            selected_pipelines=batch_job.selected_pipelines,
            created_at=batch_job.created_at,
            completed_at=batch_job.completed_at
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit job: {str(e)}"
        )


@router.get("/{job_id}", response_model=JobResponse)
async def get_job_status(
    job_id: str,
    storage: StorageBackend = Depends(get_storage)
):
    """
    Get job status and details.
    
    Args:
        job_id: Job ID to query
        
    Returns:
        Job information including current status
    """
    try:
        # Load job from database
        job = storage.load_job_metadata(job_id)
        
        return JobResponse(
            id=job.job_id,
            status=job.status.value,
            video_path=str(job.video_path) if job.video_path else None,
            config=job.config,
            selected_pipelines=job.selected_pipelines,
            created_at=job.created_at,
            completed_at=job.completed_at,
            error_message=job.error_message
        )
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
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
    storage: StorageBackend = Depends(get_storage)
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
        # Get job IDs from storage
        all_job_ids = storage.list_jobs(status_filter=status_filter)
        
        # Apply pagination
        total = len(all_job_ids)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        page_job_ids = all_job_ids[start_idx:end_idx]
        
        # Load job details for this page
        job_responses = []
        for job_id in page_job_ids:
            try:
                job = storage.load_job_metadata(job_id)
                job_responses.append(JobResponse(
                    id=job.job_id,
                    status=job.status.value,
                    video_path=str(job.video_path) if job.video_path else None,
                    config=job.config,
                    selected_pipelines=job.selected_pipelines,
                    created_at=job.created_at,
                    completed_at=job.completed_at,
                    error_message=job.error_message
                ))
            except FileNotFoundError:
                # Skip jobs that can't be loaded (shouldn't happen but be defensive)
                continue
        
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
    storage: StorageBackend = Depends(get_storage)
):
    """
    Cancel/delete a job.
    
    Args:
        job_id: Job ID to cancel
    """
    try:
        # Check if job exists
        try:
            job = storage.load_job_metadata(job_id)
        except FileNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        # TODO: Implement proper job cancellation logic
        # For now, allow deletion of any job
        storage.delete_job(job_id)
        
        return
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel job: {str(e)}"
        )