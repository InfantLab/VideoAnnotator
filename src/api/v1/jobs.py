"""Job management endpoints for VideoAnnotator API."""

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from api.database import get_storage_backend
from api.dependencies import validate_optional_api_key
from api.errors import APIError
from batch.types import BatchJob, JobStatus
from storage.base import StorageBackend
from storage.config import get_job_storage_path

router = APIRouter()

# Module-level logger for API job endpoints
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def get_storage() -> StorageBackend:
    """Get storage backend for job management."""
    return get_storage_backend()


# Pydantic models for API
class JobSubmissionRequest(BaseModel):
    """Request model for job submission."""

    config: dict[str, Any] | None = Field(
        default=None, description="Processing configuration"
    )
    selected_pipelines: list[str] | None = Field(
        default=None, description="Pipelines to run"
    )


class JobResponse(BaseModel):
    """Response model for job information (aligned with DB Job model)."""

    id: str
    status: str
    video_path: str | None = None
    config: dict[str, Any] | None = None
    selected_pipelines: list[str] | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    result_path: str | None = None
    storage_path: str | None = None  # v1.3.0: Persistent job storage directory


class JobListResponse(BaseModel):
    """Response model for job listing."""

    jobs: list[JobResponse]
    total: int
    page: int
    per_page: int


class PipelineResultResponse(BaseModel):
    """Response model for individual pipeline results."""

    pipeline_name: str
    status: str
    start_time: datetime | None = None
    end_time: datetime | None = None
    processing_time: float | None = None
    annotation_count: int | None = None
    output_file: str | None = None
    download_url: str | None = None
    error_message: str | None = None


class JobResultsResponse(BaseModel):
    """Response model for job results (aligned with DB schema)."""

    job_id: str
    status: str
    pipeline_results: dict[str, PipelineResultResponse]
    created_at: datetime | None = None
    completed_at: datetime | None = None
    result_path: str | None = None


@router.post("/", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
async def submit_job(
    video: UploadFile = File(..., description="Video file to process"),
    config: str | None = Form(None, description="JSON configuration"),
    selected_pipelines: str | None = Form(
        None, description="Comma-separated pipeline names"
    ),
    storage: StorageBackend = Depends(get_storage),
    user: dict[str, Any] | None = Depends(validate_optional_api_key),
) -> JobResponse:
    """Submit a video processing job.

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
                    detail="Invalid JSON configuration",
                )

        # Parse selected pipelines if provided
        parsed_pipelines = None
        if selected_pipelines:
            parsed_pipelines = [
                p.strip() for p in selected_pipelines.split(",") if p.strip()
            ]

        # Save uploaded video to temporary file
        # TODO: Implement proper file storage (local/S3)
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, video.filename)

        with open(video_path, "wb") as f:
            content = await video.read()
            f.write(content)

        # Create BatchJob instance with storage path
        batch_job = BatchJob(
            video_path=Path(video_path),
            output_dir=None,  # Will be set by processing system
            config=parsed_config or {},
            status=JobStatus.PENDING,
            selected_pipelines=parsed_pipelines,
        )

        # Set storage path for persistent job artifacts
        batch_job.storage_path = get_job_storage_path(batch_job.job_id)

        # Save job to database
        storage.save_job_metadata(batch_job)

        return JobResponse(
            id=batch_job.job_id,
            status=batch_job.status.value,
            video_path=str(batch_job.video_path),
            config=batch_job.config,
            selected_pipelines=batch_job.selected_pipelines,
            created_at=batch_job.created_at,
            completed_at=batch_job.completed_at,
            result_path=None,
            storage_path=str(batch_job.storage_path)
            if batch_job.storage_path
            else None,
        )

    except APIError:
        raise
    except Exception:
        raise APIError(
            status_code=500,
            code="JOB_SUBMIT_FAILED",
            message="Failed to submit job",
            hint="Check server logs",
        )


@router.get("/{job_id}", response_model=JobResponse)
async def get_job_status(
    job_id: str,
    storage: StorageBackend = Depends(get_storage),
    user: dict[str, Any] | None = Depends(validate_optional_api_key),
) -> JobResponse:
    """Get job status and details.

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
            error_message=job.error_message,
            result_path=getattr(job, "result_path", None),
            storage_path=str(job.storage_path) if job.storage_path else None,
        )

    except FileNotFoundError:
        # Message must include exact substring expected by tests: "Job not found"
        raise APIError(
            status_code=404,
            code="JOB_NOT_FOUND",
            message="Job not found",
            hint="List jobs with 'videoannotator job list'",
        )
    except APIError:
        raise
    except Exception:
        raise APIError(
            status_code=500,
            code="JOB_STATUS_FAILED",
            message="Failed to get job status",
            hint="Check server logs",
        )


@router.get("/", response_model=JobListResponse)
async def list_jobs(
    page: int = 1,
    per_page: int = 10,
    status_filter: str | None = None,
    storage: StorageBackend = Depends(get_storage),
    user: dict[str, Any] | None = Depends(validate_optional_api_key),
) -> JobListResponse:
    """List jobs with pagination and filtering.

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

        # Load job details for this page; be defensive so a single bad job doesn't break the whole list
        job_responses = []
        for job_id in page_job_ids:
            try:
                job = storage.load_job_metadata(job_id)
                job_responses.append(
                    JobResponse(
                        id=job.job_id,
                        status=job.status.value,
                        video_path=str(job.video_path) if job.video_path else None,
                        config=job.config,
                        selected_pipelines=job.selected_pipelines,
                        created_at=job.created_at,
                        completed_at=job.completed_at,
                        error_message=job.error_message,
                        result_path=getattr(job, "result_path", None),
                        storage_path=str(job.storage_path)
                        if job.storage_path
                        else None,
                    )
                )
            except FileNotFoundError:
                # Skip jobs that can't be loaded (shouldn't happen but be defensive)
                logger.warning(
                    f"[WARNING] Job {job_id} listed but not found when loading details; skipping"
                )
                continue
            except Exception as e:
                # Log the problematic job and continue with others. Avoid returning 500 for a single bad entry.
                logger.error(
                    f"[ERROR] Failed to load job {job_id} while listing jobs: {e}"
                )
                continue

        return JobListResponse(
            jobs=job_responses, total=total, page=page, per_page=per_page
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list jobs: {e!s}",
        )


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_job(
    job_id: str,
    storage: StorageBackend = Depends(get_storage),
    user: dict[str, Any] | None = Depends(validate_optional_api_key),
) -> None:
    """Cancel/delete a job.

    Args:
        job_id: Job ID to cancel
    """
    try:
        # Check if job exists
        try:
            # Ensure job exists (we don't need the object here)
            storage.load_job_metadata(job_id)
        except FileNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found"
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
            detail=f"Failed to cancel job: {e!s}",
        )


@router.get("/{job_id}/results", response_model=JobResultsResponse)
async def get_job_results(
    job_id: str,
    storage: StorageBackend = Depends(get_storage),
    user: dict[str, Any] | None = Depends(validate_optional_api_key),
) -> JobResultsResponse:
    """Get detailed results for a completed job.

    Args:
        job_id: Job ID to get results for

    Returns:
        Detailed job results including pipeline outputs and file paths
    """
    try:
        # Load job from database
        job = storage.load_job_metadata(job_id)

        # Check if job exists
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found"
            )

        # Convert pipeline results to response format
        pipeline_results = {}
        for name, result in job.pipeline_results.items():
            pipeline_results[name] = PipelineResultResponse(
                pipeline_name=result.pipeline_name,
                status=result.status.value,
                start_time=result.start_time,
                end_time=result.end_time,
                processing_time=result.processing_time,
                annotation_count=result.annotation_count,
                output_file=str(result.output_file) if result.output_file else None,
                error_message=result.error_message,
            )

        # Build full pipeline results with download URLs
        for name, result in pipeline_results.items():
            if result.output_file:
                try:
                    # Construct a download URL for convenience; client may use server base URL
                    # Note: This is a relative path; frontend should prepend server origin
                    result.download_url = (
                        f"/api/v1/jobs/{job.job_id}/results/files/{name}"
                    )
                except Exception:
                    pass

        return JobResultsResponse(
            job_id=job.job_id,
            status=job.status.value,
            pipeline_results=pipeline_results,
            created_at=job.created_at,
            completed_at=job.completed_at,
            result_path=getattr(job, "result_path", None),
        )

    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job results: {e!s}",
        )


@router.get("/{job_id}/results/files/{pipeline_name}")
async def download_result_file(
    job_id: str,
    pipeline_name: str,
    storage: StorageBackend = Depends(get_storage),
    user: dict[str, Any] | None = Depends(validate_optional_api_key),
) -> Any:
    """Download a specific result file from a job.

    Args:
        job_id: Job ID
        pipeline_name: Name of pipeline to download results for

    Returns:
        File download response
    """
    try:
        # Load job from database
        job = storage.load_job_metadata(job_id)

        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found"
            )

        # Check if pipeline result exists
        if pipeline_name not in job.pipeline_results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pipeline '{pipeline_name}' results not found for job {job_id}",
            )

        result = job.pipeline_results[pipeline_name]

        # Check if output file exists
        if not result.output_file:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No output file for pipeline '{pipeline_name}' in job {job_id}",
            )

        output_file_path = Path(result.output_file)

        # Verify file exists on disk
        if not output_file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Output file not found: {output_file_path}",
            )

        # Return file
        return FileResponse(
            path=str(output_file_path),
            filename=output_file_path.name,
            media_type="application/octet-stream",
        )

    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download result file: {e!s}",
        )
