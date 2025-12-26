"""Job management endpoints for VideoAnnotator API."""

import contextlib
import json
import logging
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ...batch.types import BatchJob, JobStatus
from ...storage.base import StorageBackend
from ...storage.manager import get_storage_provider
from ...validation.validator import ConfigValidator
from ..database import get_storage_backend
from ..errors import APIError
from ..middleware.auth import validate_api_key
from .exceptions import (
    InvalidConfigException,
    InvalidRequestException,
    JobAlreadyCompletedException,
    JobNotFoundException,
)

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


def extract_video_metadata(
    video_path: Path | str | None,
) -> tuple[str | None, int | None, int | None]:
    """Extract video metadata (filename, size, duration).

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (filename, size_bytes, duration_seconds)
    """
    video_filename = None
    video_size_bytes = None
    video_duration_seconds = None

    if not video_path:
        return video_filename, video_size_bytes, video_duration_seconds

    video_path = Path(video_path)
    if not video_path.exists():
        return video_filename, video_size_bytes, video_duration_seconds

    video_filename = video_path.name

    try:
        video_size_bytes = video_path.stat().st_size
    except Exception as e:
        logger.debug(f"Could not get video file size: {e}")

    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps > 0:
                video_duration_seconds = int(frame_count / fps)
            cap.release()
    except Exception as e:
        logger.debug(f"Could not extract video duration: {e}")

    return video_filename, video_size_bytes, video_duration_seconds


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
    video_filename: str | None = Field(
        default=None, description="Original video filename"
    )
    video_size_bytes: int | None = Field(
        default=None, description="Video file size in bytes"
    )
    video_duration_seconds: int | None = Field(
        default=None, description="Video duration in seconds"
    )
    config: dict[str, Any] | None = None
    selected_pipelines: list[str] | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    result_path: str | None = None
    storage_path: str | None = None  # v1.3.0: Persistent job storage directory
    queue_position: int | None = Field(
        default=None,
        description="1-based position in the pending queue (only set when status is 'pending')",
    )


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
    error_message: str | None = None  # Job-level error message for failed jobs


@router.post("", include_in_schema=False)
@router.post(
    "/",
    response_model=JobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit Video Processing Job",
    description="""
Submit a video file for processing with one or more annotation pipelines.

The video is uploaded as multipart/form-data and processed asynchronously.
Returns immediately with a job ID that can be used to check status and retrieve results.

**Configuration Validation**: If both config and selected_pipelines are provided,
the configuration is validated against each pipeline's requirements before job creation.

**Supported Video Formats**: MP4, AVI, MOV, MKV, WEBM (FFmpeg-compatible formats)

**curl Example - Basic Submission**:
```bash
curl -X POST "http://localhost:18011/api/v1/jobs/" \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -F "video=@/path/to/video.mp4"
```

**curl Example - With Pipeline Selection**:
```bash
curl -X POST "http://localhost:18011/api/v1/jobs/" \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -F "video=@/path/to/video.mp4" \\
  -F "selected_pipelines=person_tracking,face_recognition"
```

**curl Example - With Configuration**:
```bash
curl -X POST "http://localhost:18011/api/v1/jobs/" \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -F "video=@/path/to/video.mp4" \\
  -F "selected_pipelines=person_tracking" \\
  -F 'config={"person_tracking":{"confidence_threshold":0.7}}'
```

**Success Response** (201 Created):
```json
{
  "id": "abc123-def456-ghi789",
  "status": "pending",
  "video_path": "/tmp/uploads/video.mp4",
  "config": {"person_tracking": {"confidence_threshold": 0.7}},
  "selected_pipelines": ["person_tracking"],
  "created_at": "2025-10-22T10:30:00Z",
  "completed_at": null,
  "error_message": null,
  "result_path": null,
  "storage_path": "/storage/jobs/abc123-def456-ghi789"
}
```

**Error Response** (400 Bad Request - Invalid Configuration):
```json
{
  "error": {
    "code": "INVALID_CONFIG",
    "message": "Configuration validation failed: person_tracking: Invalid value for confidence_threshold (must be between 0 and 1)",
    "hint": "Fix the validation errors and resubmit",
    "timestamp": "2025-10-22T10:30:00Z"
  }
}
```

**Error Response** (401 Unauthorized - Missing/Invalid API Key):
```json
{
  "error": {
    "code": "UNAUTHORIZED",
    "message": "Invalid or missing API key",
    "hint": "Include valid API key in Authorization header: Bearer YOUR_KEY",
    "timestamp": "2025-10-22T10:30:00Z"
  }
}
```
""",
)
async def submit_job(
    video: UploadFile = File(
        ..., description="Video file to process (MP4, AVI, MOV, MKV, WEBM)"
    ),
    config: str | None = Form(
        None,
        description="JSON configuration object for pipeline parameters. "
        'Example: \'{"person_tracking":{"confidence_threshold":0.7}}\'',
    ),
    selected_pipelines: str | None = Form(
        None,
        description="Comma-separated list of pipeline names to run. "
        "Example: 'person_tracking,face_recognition'. "
        "Use GET /api/v1/pipelines to list available pipelines.",
    ),
    storage: StorageBackend = Depends(get_storage),
    user: dict[str, Any] | None = Depends(validate_api_key),
) -> JobResponse:
    """Submit a video processing job (see endpoint description for details)."""
    try:
        # Parse configuration if provided
        parsed_config = None
        if config:
            try:
                parsed_config = json.loads(config)
            except json.JSONDecodeError as e:
                raise InvalidRequestException(
                    message="Invalid JSON configuration",
                    hint="Check JSON syntax for missing brackets, quotes, or commas",
                ) from e

        # Parse selected pipelines if provided
        parsed_pipelines = None
        if selected_pipelines:
            parsed_pipelines = [
                p.strip() for p in selected_pipelines.split(",") if p.strip()
            ]

        # Validate configuration if pipelines are specified (v1.3.0)
        if parsed_pipelines and parsed_config:
            validator = ConfigValidator()
            validation_results = validator.validate_batch(
                {pipeline: parsed_config for pipeline in parsed_pipelines}
            )

            # Check if any pipeline validation failed
            failed = {
                pipeline: result
                for pipeline, result in validation_results.items()
                if not result.valid
            }

            if failed:
                # Build comprehensive error response
                error_messages = []
                for pipeline, result in failed.items():
                    for error in result.errors:
                        msg = f"{pipeline}: {error.message}"
                        if error.hint:
                            msg += f" ({error.hint})"
                        error_messages.append(msg)

                raise InvalidConfigException(
                    message=f"Configuration validation failed: {'; '.join(error_messages)}",
                    hint="Fix the validation errors and resubmit",
                )

        # Save uploaded video to temporary file first.
        # NOTE: In some ASGI stacks the underlying UploadFile file pointer may be
        # positioned at EOF after request parsing. Always seek to start.
        temp_dir = tempfile.mkdtemp()
        filename = video.filename or "video"
        temp_video_path = os.path.join(temp_dir, filename)

        try:
            await video.seek(0)
            with open(temp_video_path, "wb") as f:
                while True:
                    chunk = await video.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)

            # Extract video metadata from temp file
            video_filename, video_size_bytes, video_duration_seconds = (
                extract_video_metadata(temp_video_path)
            )

            # Create BatchJob instance to get job_id
            batch_job = BatchJob(
                video_path=Path(temp_video_path),  # Temporary, will be updated
                output_dir=None,  # Will be set by processing system
                config=parsed_config or {},
                status=JobStatus.PENDING,
                selected_pipelines=parsed_pipelines,
            )

            # Use StorageProvider to save the file
            provider = get_storage_provider()

            with open(temp_video_path, "rb") as f:
                provider.save_file(batch_job.job_id, filename, f)

            # Update job to use persistent path
            batch_job.storage_path = provider.get_absolute_path(batch_job.job_id, "")
            batch_job.video_path = provider.get_absolute_path(
                batch_job.job_id, filename
            )

        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as cleanup_error:
                logger.warning(
                    f"Failed to cleanup temp directory {temp_dir}: {cleanup_error}"
                )

        # Save job to database
        storage.save_job_metadata(batch_job)

        queue_position: int | None = None
        if batch_job.status == JobStatus.PENDING:
            try:
                pending_ids = storage.list_jobs(status_filter=JobStatus.PENDING.value)
                queue_position = next(
                    (
                        idx + 1
                        for idx, pending_job_id in enumerate(pending_ids)
                        if pending_job_id == batch_job.job_id
                    ),
                    None,
                )
            except Exception as e:
                logger.debug(
                    f"Could not compute queue_position for {batch_job.job_id}: {e}"
                )

        return JobResponse(
            id=batch_job.job_id,
            status=batch_job.status.value,
            video_path=str(batch_job.video_path),
            video_filename=video_filename,
            video_size_bytes=video_size_bytes,
            video_duration_seconds=video_duration_seconds,
            config=batch_job.config,
            selected_pipelines=batch_job.selected_pipelines,
            created_at=batch_job.created_at,
            completed_at=batch_job.completed_at,
            result_path=None,
            storage_path=str(batch_job.storage_path)
            if batch_job.storage_path
            else None,
            queue_position=queue_position,
        )

    except (InvalidRequestException, InvalidConfigException, APIError):
        # Let validation errors and other custom exceptions propagate
        raise
    except Exception as e:
        raise APIError(
            status_code=500,
            code="JOB_SUBMIT_FAILED",
            message="Failed to submit job",
            hint="Check server logs",
        ) from e


@router.get("/{job_id}/", include_in_schema=False)
@router.get(
    "/{job_id}",
    response_model=JobResponse,
    summary="Get Job Status",
    description="""
Retrieve the current status and details of a specific video processing job.

Use this endpoint to poll job status during processing. The status field indicates
the current state of the job: pending, running, completed, failed, or cancelled.

**Status Values**:
- `pending`: Job queued, not yet started
- `running`: Job currently processing
- `completed`: Job finished successfully, results available
- `failed`: Job encountered an error (see error_message)
- `cancelled`: Job was cancelled by user request

**curl Example**:
```bash
curl -X GET "http://localhost:18011/api/v1/jobs/abc123-def456" \\
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Success Response** (200 OK - Running Job):
```json
{
  "id": "abc123-def456",
  "status": "running",
  "video_path": "/tmp/uploads/video.mp4",
  "config": {"person_tracking": {"confidence_threshold": 0.7}},
  "selected_pipelines": ["person_tracking", "face_recognition"],
  "created_at": "2025-10-22T10:00:00Z",
  "completed_at": null,
  "error_message": null,
  "result_path": null,
  "storage_path": "/storage/jobs/abc123-def456"
}
```

**Success Response** (200 OK - Completed Job):
```json
{
  "id": "abc123-def456",
  "status": "completed",
  "video_path": "/tmp/uploads/video.mp4",
  "config": {"person_tracking": {"confidence_threshold": 0.7}},
  "selected_pipelines": ["person_tracking"],
  "created_at": "2025-10-22T10:00:00Z",
  "completed_at": "2025-10-22T10:05:23Z",
  "error_message": null,
  "result_path": "/storage/jobs/abc123-def456/results",
  "storage_path": "/storage/jobs/abc123-def456"
}
```

**Error Response** (404 Not Found):
```json
{
  "error": {
    "code": "JOB_NOT_FOUND",
    "message": "Job abc123-invalid not found",
    "detail": {"job_id": "abc123-invalid"},
    "hint": "Check job ID or use GET /api/v1/jobs to list all jobs",
    "timestamp": "2025-10-22T10:30:00Z"
  }
}
```

**Error Response** (401 Unauthorized):
```json
{
  "error": {
    "code": "UNAUTHORIZED",
    "message": "Invalid or missing API key",
    "hint": "Include valid API key in Authorization header",
    "timestamp": "2025-10-22T10:30:00Z"
  }
}
```
""",
)
async def get_job_status(
    job_id: str,
    storage: StorageBackend = Depends(get_storage),
    user: dict[str, Any] | None = Depends(validate_api_key),
) -> JobResponse:
    """Get job status and details (see endpoint description for details)."""
    try:
        # Load job from database
        job = storage.load_job_metadata(job_id)

        queue_position: int | None = None
        if job.status == JobStatus.PENDING:
            try:
                pending_ids = storage.list_jobs(status_filter=JobStatus.PENDING.value)
                queue_position = next(
                    (
                        idx + 1
                        for idx, pending_job_id in enumerate(pending_ids)
                        if pending_job_id == job.job_id
                    ),
                    None,
                )
            except Exception as e:
                logger.debug(f"Could not compute queue_position for {job.job_id}: {e}")

        # Extract video metadata if available
        video_filename, video_size_bytes, video_duration_seconds = (
            extract_video_metadata(job.video_path)
        )

        return JobResponse(
            id=job.job_id,
            status=job.status.value,
            video_path=str(job.video_path) if job.video_path else None,
            video_filename=video_filename,
            video_size_bytes=video_size_bytes,
            video_duration_seconds=video_duration_seconds,
            config=job.config,
            selected_pipelines=job.selected_pipelines,
            created_at=job.created_at,
            completed_at=job.completed_at,
            error_message=job.error_message,
            result_path=getattr(job, "result_path", None),
            storage_path=str(job.storage_path) if job.storage_path else None,
            queue_position=queue_position,
        )

    except FileNotFoundError as e:
        # Message must include exact substring expected by tests: "Job not found"
        raise JobNotFoundException(
            job_id=job_id,
            hint="Check job ID or use GET /api/v1/jobs to list all jobs",
        ) from e
    except (
        JobNotFoundException,
        InvalidRequestException,
        InvalidConfigException,
        JobAlreadyCompletedException,
        APIError,
    ):
        raise
    except Exception as e:
        raise APIError(
            status_code=500,
            code="JOB_STATUS_FAILED",
            message="Failed to get job status",
            hint="Check server logs",
        ) from e


@router.get("", include_in_schema=False)
@router.get(
    "/",
    response_model=JobListResponse,
    summary="List Processing Jobs",
    description="""
List all video processing jobs with pagination and optional status filtering.

Returns a paginated list of jobs with their current status, configuration, and metadata.
Use this endpoint to monitor multiple jobs or filter by status.

**Status Values**: pending, running, completed, failed, cancelled

**curl Example - List All Jobs**:
```bash
curl -X GET "http://localhost:18011/api/v1/jobs/" \\
  -H "Authorization: Bearer YOUR_API_KEY"
```

**curl Example - Filter by Status**:
```bash
curl -X GET "http://localhost:18011/api/v1/jobs/?status_filter=completed" \\
  -H "Authorization: Bearer YOUR_API_KEY"
```

**curl Example - Pagination**:
```bash
curl -X GET "http://localhost:18011/api/v1/jobs/?page=2&per_page=20" \\
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Success Response** (200 OK):
```json
{
  "jobs": [
    {
      "id": "abc123-def456",
      "status": "completed",
      "video_path": "/tmp/uploads/video1.mp4",
      "config": {"person_tracking": {"confidence_threshold": 0.7}},
      "selected_pipelines": ["person_tracking"],
      "created_at": "2025-10-22T10:00:00Z",
      "completed_at": "2025-10-22T10:05:23Z",
      "error_message": null,
      "result_path": "/storage/jobs/abc123-def456/results",
      "storage_path": "/storage/jobs/abc123-def456"
    },
    {
      "id": "xyz789-uvw012",
      "status": "running",
      "video_path": "/tmp/uploads/video2.mp4",
      "config": null,
      "selected_pipelines": ["face_recognition"],
      "created_at": "2025-10-22T10:10:00Z",
      "completed_at": null,
      "error_message": null,
      "result_path": null,
      "storage_path": "/storage/jobs/xyz789-uvw012"
    }
  ],
  "total": 2,
  "page": 1,
  "per_page": 10
}
```

**Error Response** (401 Unauthorized):
```json
{
  "error": {
    "code": "UNAUTHORIZED",
    "message": "Invalid or missing API key",
    "hint": "Include valid API key in Authorization header",
    "timestamp": "2025-10-22T10:30:00Z"
  }
}
```
""",
)
async def list_jobs(
    page: int = 1,
    per_page: int = 10,
    status_filter: str | None = None,
    storage: StorageBackend = Depends(get_storage),
    user: dict[str, Any] | None = Depends(validate_api_key),
) -> JobListResponse:
    """List jobs with pagination and filtering (see endpoint description for details)."""
    try:
        pending_positions: dict[str, int] = {}
        if status_filter is None or status_filter == JobStatus.PENDING.value:
            pending_ids = storage.list_jobs(status_filter=JobStatus.PENDING.value)
            pending_positions = {
                pending_job_id: idx + 1
                for idx, pending_job_id in enumerate(pending_ids)
            }

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

                # Extract video metadata
                (
                    video_filename,
                    video_size_bytes,
                    video_duration_seconds,
                ) = extract_video_metadata(job.video_path)

                job_responses.append(
                    JobResponse(
                        id=job.job_id,
                        status=job.status.value,
                        video_path=str(job.video_path) if job.video_path else None,
                        video_filename=video_filename,
                        video_size_bytes=video_size_bytes,
                        video_duration_seconds=video_duration_seconds,
                        config=job.config,
                        selected_pipelines=job.selected_pipelines,
                        created_at=job.created_at,
                        completed_at=job.completed_at,
                        error_message=job.error_message,
                        result_path=getattr(job, "result_path", None),
                        storage_path=str(job.storage_path)
                        if job.storage_path
                        else None,
                        queue_position=pending_positions.get(job.job_id)
                        if job.status == JobStatus.PENDING
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
        raise APIError(
            status_code=500,
            code="JOB_LIST_FAILED",
            message=f"Failed to list jobs: {e!s}",
            hint="Check server logs for details",
        ) from e


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_job(
    job_id: str,
    storage: StorageBackend = Depends(get_storage),
    user: dict[str, Any] | None = Depends(validate_api_key),
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
        except FileNotFoundError as e:
            raise JobNotFoundException(
                job_id=job_id,
                hint="Check job ID or use GET /api/v1/jobs to list all jobs",
            ) from e

        # TODO: Implement proper job cancellation logic
        # For now, allow deletion of any job
        storage.delete_job(job_id)

        return

    except (JobNotFoundException, APIError):
        raise
    except Exception as e:
        raise APIError(
            status_code=500,
            code="JOB_DELETE_FAILED",
            message=f"Failed to cancel job: {e!s}",
            hint="Check server logs for details",
        ) from e


@router.post(
    "/{job_id}/cancel",
    response_model=JobResponse,
    summary="Cancel Job",
    description="""
Cancel a running or pending video processing job.

Attempts to gracefully stop job execution. Jobs that are already completed, failed,
or previously cancelled return their current status (idempotent operation).

**Cancellation Behavior**:
- `pending` jobs: Removed from queue, never started
- `running` jobs: Interrupted, partial results may be available
- `completed/failed` jobs: Cannot be cancelled (returns 409 error)
- `cancelled` jobs: Returns current state (idempotent)

**curl Example**:
```bash
curl -X POST "http://localhost:18011/api/v1/jobs/abc123-def456/cancel" \\
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Success Response** (200 OK - Job Cancelled):
```json
{
  "id": "abc123-def456",
  "status": "cancelled",
  "video_path": "/tmp/uploads/video.mp4",
  "config": {"person_tracking": {"confidence_threshold": 0.7}},
  "selected_pipelines": ["person_tracking"],
  "created_at": "2025-10-22T10:00:00Z",
  "completed_at": "2025-10-22T10:02:15Z",
  "error_message": "Job cancelled by user request",
  "result_path": null,
  "storage_path": "/storage/jobs/abc123-def456"
}
```

**Error Response** (404 Not Found):
```json
{
  "error": {
    "code": "JOB_NOT_FOUND",
    "message": "Job abc123-invalid not found",
    "detail": {"job_id": "abc123-invalid"},
    "hint": "Check job ID or use GET /api/v1/jobs to list all jobs",
    "timestamp": "2025-10-22T10:30:00Z"
  }
}
```

**Error Response** (409 Conflict - Already Completed):
```json
{
  "error": {
    "code": "JOB_ALREADY_COMPLETED",
    "message": "Job abc123-def456 is already completed (status: completed)",
    "detail": {"job_id": "abc123-def456", "status": "completed"},
    "hint": "Cannot cancel a job that has already finished",
    "timestamp": "2025-10-22T10:30:00Z"
  }
}
```
""",
)
async def cancel_job_endpoint(
    job_id: str,
    storage: StorageBackend = Depends(get_storage),
    user: dict[str, Any] | None = Depends(validate_api_key),
) -> JobResponse:
    """Cancel a running or pending job (see endpoint description for details)."""
    try:
        # Load job from database
        try:
            job_data = storage.load_job_metadata(job_id)
        except FileNotFoundError as e:
            raise JobNotFoundException(
                job_id=job_id,
                hint="Check job ID or use GET /api/v1/jobs to list all jobs",
            ) from e

        # Check current status
        current_status = job_data.status

        # If already cancelled, return current state (idempotent)
        if current_status == JobStatus.CANCELLED:
            logger.info(f"[CANCEL] Job {job_id} already cancelled (idempotent)")

            # Extract video metadata
            video_filename, video_size_bytes, video_duration_seconds = (
                extract_video_metadata(job_data.video_path)
            )

            return JobResponse(
                id=str(job_id),
                status=JobStatus.CANCELLED.value,
                video_path=str(job_data.video_path) if job_data.video_path else None,
                video_filename=video_filename,
                video_size_bytes=video_size_bytes,
                video_duration_seconds=video_duration_seconds,
                config=job_data.config,
                selected_pipelines=job_data.selected_pipelines,
                created_at=job_data.created_at if job_data.created_at else None,
                completed_at=job_data.completed_at if job_data.completed_at else None,
                error_message=job_data.error_message,
                result_path=None,  # BatchJob doesn't have result_path
                storage_path=str(job_data.storage_path)
                if job_data.storage_path
                else None,
            )

        # If already in a final state (completed/failed), return error
        if current_status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            raise JobAlreadyCompletedException(
                job_id=job_id,
                status=current_status.value,
            )

        # Cancel the job
        logger.info(
            f"[CANCEL] Cancelling job {job_id} (current status: {current_status})"
        )

        # Update job status to CANCELLED
        job_data.status = JobStatus.CANCELLED
        job_data.error_message = "Job cancelled by user request"

        # Set completed_at if not already set
        if job_data.completed_at is None:
            job_data.completed_at = datetime.now()

        # Save updated job
        storage.save_job_metadata(job_data)

        logger.info(f"[OK] Job {job_id} cancelled successfully")

        # Extract video metadata
        video_filename, video_size_bytes, video_duration_seconds = (
            extract_video_metadata(job_data.video_path)
        )

        # Return updated job response
        return JobResponse(
            id=str(job_id),
            status=JobStatus.CANCELLED.value,
            video_path=str(job_data.video_path) if job_data.video_path else None,
            video_filename=video_filename,
            video_size_bytes=video_size_bytes,
            video_duration_seconds=video_duration_seconds,
            config=job_data.config,
            selected_pipelines=job_data.selected_pipelines,
            created_at=job_data.created_at if job_data.created_at else None,
            completed_at=job_data.completed_at if job_data.completed_at else None,
            error_message=job_data.error_message,
            result_path=None,  # BatchJob doesn't have result_path
            storage_path=str(job_data.storage_path) if job_data.storage_path else None,
        )

    except (JobNotFoundException, JobAlreadyCompletedException, APIError):
        raise
    except Exception as e:
        logger.error(f"[ERROR] Failed to cancel job {job_id}: {e}", exc_info=True)
        raise APIError(
            status_code=500,
            code="JOB_CANCEL_FAILED",
            message=f"Failed to cancel job: {e!s}",
            hint="Check server logs for details",
        ) from e


@router.get(
    "/{job_id}/results",
    response_model=JobResultsResponse,
    summary="Get Job Results",
    description="""
Retrieve detailed results for a completed video processing job.

Returns pipeline-specific outputs, annotation counts, processing times, and download URLs
for generated files. Only available after job completes successfully.

**Result Files**: Each pipeline generates output files (e.g., person_tracking.json,
face_recognition.json) that can be downloaded using the provided download_url.

**curl Example**:
```bash
curl -X GET "http://localhost:18011/api/v1/jobs/abc123-def456/results" \\
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Success Response** (200 OK):
```json
{
  "job_id": "abc123-def456",
  "status": "completed",
  "pipeline_results": {
    "person_tracking": {
      "pipeline_name": "person_tracking",
      "status": "completed",
      "start_time": "2025-10-22T10:00:05Z",
      "end_time": "2025-10-22T10:03:12Z",
      "processing_time": 187.3,
      "annotation_count": 1245,
      "output_file": "/storage/jobs/abc123-def456/person_tracking.json",
      "download_url": "/api/v1/jobs/abc123-def456/results/files/person_tracking",
      "error_message": null
    },
    "face_recognition": {
      "pipeline_name": "face_recognition",
      "status": "completed",
      "start_time": "2025-10-22T10:03:15Z",
      "end_time": "2025-10-22T10:05:23Z",
      "processing_time": 128.1,
      "annotation_count": 423,
      "output_file": "/storage/jobs/abc123-def456/face_recognition.json",
      "download_url": "/api/v1/jobs/abc123-def456/results/files/face_recognition",
      "error_message": null
    }
  },
  "created_at": "2025-10-22T10:00:00Z",
  "completed_at": "2025-10-22T10:05:23Z",
  "result_path": "/storage/jobs/abc123-def456/results"
}
```

**Error Response** (404 Not Found):
```json
{
  "error": {
    "code": "JOB_NOT_FOUND",
    "message": "Job abc123-invalid not found",
    "detail": {"job_id": "abc123-invalid"},
    "hint": "Check job ID or use GET /api/v1/jobs to list all jobs",
    "timestamp": "2025-10-22T10:30:00Z"
  }
}
```

**Note**: For jobs still in progress, use GET /jobs/{job_id} to check status first.
""",
)
async def get_job_results(
    job_id: str,
    storage: StorageBackend = Depends(get_storage),
    user: dict[str, Any] | None = Depends(validate_api_key),
) -> JobResultsResponse:
    """Get detailed results for a completed job (see endpoint description for details)."""
    try:
        # Load job from database
        job = storage.load_job_metadata(job_id)

        # Check if job exists
        if not job:
            raise JobNotFoundException(
                job_id=job_id,
                hint="Check job ID or use GET /api/v1/jobs to list all jobs",
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
                with contextlib.suppress(Exception):
                    # Construct a download URL for convenience; client may use server base URL
                    # Note: This is a relative path; frontend should prepend server origin
                    result.download_url = (
                        f"/api/v1/jobs/{job.job_id}/results/files/{name}"
                    )

        return JobResultsResponse(
            job_id=job.job_id,
            status=job.status.value,
            pipeline_results=pipeline_results,
            created_at=job.created_at,
            completed_at=job.completed_at,
            result_path=getattr(job, "result_path", None),
            error_message=job.error_message,  # Include job-level error for failed jobs
        )

    except (JobNotFoundException, APIError):
        raise
    except FileNotFoundError as e:
        raise JobNotFoundException(
            job_id=job_id,
            hint="Check job ID or use GET /api/v1/jobs to list all jobs",
        ) from e
    except Exception as e:
        raise APIError(
            status_code=500,
            code="JOB_RESULTS_FAILED",
            message=f"Failed to get job results: {e!s}",
            hint="Check server logs for details",
        ) from e


@router.get("/{job_id}/results/files/{pipeline_name}")
async def download_result_file(
    job_id: str,
    pipeline_name: str,
    storage: StorageBackend = Depends(get_storage),
    user: dict[str, Any] | None = Depends(validate_api_key),
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
            raise JobNotFoundException(
                job_id=job_id,
                hint="Check job ID or use GET /api/v1/jobs to list all jobs",
            )

        # Check if pipeline result exists
        if pipeline_name not in job.pipeline_results:
            raise InvalidRequestException(
                message=f"Pipeline '{pipeline_name}' results not found for job {job_id}",
                hint=f"Check pipeline name or use GET /api/v1/jobs/{job_id}/results to see available results",
            )

        result = job.pipeline_results[pipeline_name]

        # Check if output file exists
        if not result.output_file:
            raise InvalidRequestException(
                message=f"No output file for pipeline '{pipeline_name}' in job {job_id}",
                hint="This pipeline may not generate an output file",
            )

        output_file_path = Path(result.output_file)

        # Verify file exists on disk
        if not output_file_path.exists():
            raise APIError(
                status_code=500,
                code="OUTPUT_FILE_MISSING",
                message=f"Output file not found: {output_file_path}",
                hint="File may have been deleted or storage may be corrupt",
            )

        # Return file
        return FileResponse(
            path=str(output_file_path),
            filename=output_file_path.name,
            media_type="application/octet-stream",
        )

    except (JobNotFoundException, InvalidRequestException, APIError):
        raise
    except FileNotFoundError as e:
        raise JobNotFoundException(
            job_id=job_id,
            hint="Check job ID or use GET /api/v1/jobs to list all jobs",
        ) from e
    except Exception as e:
        raise APIError(
            status_code=500,
            code="DOWNLOAD_FAILED",
            message=f"Failed to download result file: {e!s}",
            hint="Check server logs for details",
        ) from e
