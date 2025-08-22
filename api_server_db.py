#!/usr/bin/env python3
"""
VideoAnnotator API Server v1.2.0 - Production Version with Database Integration

This is the production API server that integrates with the database layer
and existing batch processing system.
"""

import asyncio
import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

# Import database components
from src.database import get_db, create_tables
from src.database.models import Job, JobStatus
from src.database.crud import JobCRUD, APIKeyCRUD

# Import batch processing system
from src.batch.batch_orchestrator import BatchOrchestrator
from src.batch.types import BatchJob, JobStatus as BatchJobStatus
from src.storage.file_backend import FileStorageBackend

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)

# Lifespan event handler needs to be defined before FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await startup_tasks()
    yield
    # Shutdown (if needed)
    logger.info("🔄 API server shutting down...")

# FastAPI app configuration
app = FastAPI(
    title="VideoAnnotator API",
    description="Production REST API for video annotation processing with database persistence",
    version="1.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API responses
class JobResponse(BaseModel):
    id: str
    status: str
    video_path: Optional[str] = None
    video_filename: Optional[str] = None
    video_size_bytes: Optional[int] = None
    video_duration_seconds: Optional[int] = None
    selected_pipelines: Optional[List[str]] = None
    config: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    updated_at: Optional[str] = None
    result_path: Optional[str] = None
    error_message: Optional[str] = None
    progress_percentage: Optional[int] = None
    duration_seconds: Optional[float] = None
    is_complete: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None

class JobListResponse(BaseModel):
    jobs: List[JobResponse]
    total: int
    page: int
    per_page: int

class PipelineInfo(BaseModel):
    name: str
    description: str
    enabled: bool
    config_schema: Dict[str, Any]

class SystemHealth(BaseModel):
    status: str
    api_version: str
    videoannotator_version: str
    timestamp: str
    database: Dict[str, Any]
    system: Dict[str, Any]

# Authentication dependency
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Get current user from API key authentication."""
    if not credentials:
        return None  # Allow anonymous access for some endpoints
    
    try:
        # Extract API key from Bearer token
        api_key = credentials.credentials
        
        # Authenticate user
        user = APIKeyCRUD.authenticate(db, api_key)
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid API key"
            )
        
        return user
        
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=401,
            detail="Authentication failed"
        )

async def startup_tasks():
    """Initialize database, batch orchestrator, and display startup information."""
    logger.info("=" * 60)
    logger.info("🚀 VideoAnnotator API Server v1.2.0 - Production Mode")
    logger.info("=" * 60)
    
    # Initialize database
    logger.info("📊 Initializing database...")
    try:
        create_tables()
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise
    
    # Batch orchestrator will be initialized when first needed
    logger.info("🔧 Batch processing system will initialize on first use")
    
    # Create results directory
    try:
        Path("api_results").mkdir(exist_ok=True)
        logger.info("📁 API results directory ready")
    except Exception as e:
        logger.warning(f"⚠️ Could not create results directory: {e}")
    
    logger.info("📖 API Documentation: http://localhost:8000/docs")
    logger.info("🔒 Authentication: API Key required for protected endpoints")
    logger.info("🚀 Production Mode: Using database persistence + batch processing")
    logger.info("=" * 60)

# Health endpoints
@app.get("/health")
async def health_check():
    """Basic health check."""
    return {
        "status": "healthy",
        "api_version": "1.2.0",
        "videoannotator_version": "1.2.0",
        "message": "VideoAnnotator API is running"
    }

@app.get("/api/v1/system/health", response_model=SystemHealth)
async def detailed_health(db: Session = Depends(get_db)):
    """Detailed system health check."""
    import psutil
    import platform
    from src.database.migrations import get_database_info
    
    # Get database info
    db_info = get_database_info()
    
    return SystemHealth(
        status="healthy",
        api_version="1.2.0",
        videoannotator_version="1.2.0",
        timestamp=datetime.utcnow().isoformat(),
        database=db_info,
        system={
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent
        }
    )

# Global batch orchestrator instance
batch_orchestrator = None

def get_batch_orchestrator() -> BatchOrchestrator:
    """Get singleton batch orchestrator instance with lazy loading."""
    global batch_orchestrator
    if batch_orchestrator is None:
        try:
            # Create storage backend for batch processing
            storage_backend = FileStorageBackend(Path("api_results"))
            batch_orchestrator = BatchOrchestrator(
                storage_backend=storage_backend,
                max_retries=2,  # API jobs should have fewer retries
                checkpoint_interval=5  # More frequent checkpoints for API jobs
            )
            logger.info("🔧 Initialized batch orchestrator for API processing")
        except Exception as e:
            logger.error(f"Failed to initialize batch orchestrator: {e}")
            logger.warning("Creating minimal batch orchestrator without pipeline imports")
            # Create a minimal orchestrator that won't hang on imports
            from src.batch.progress_tracker import ProgressTracker
            from src.batch.recovery import FailureRecovery, RetryStrategy
            
            batch_orchestrator = BatchOrchestrator.__new__(BatchOrchestrator)
            batch_orchestrator.storage_backend = FileStorageBackend(Path("api_results"))
            batch_orchestrator.progress_tracker = ProgressTracker()
            batch_orchestrator.failure_recovery = FailureRecovery(max_retries=2, strategy=RetryStrategy.EXPONENTIAL_BACKOFF)
            batch_orchestrator.checkpoint_interval = 5
            batch_orchestrator.logger = logging.getLogger(__name__)
            batch_orchestrator.batch_id = None
            batch_orchestrator.jobs = []
            batch_orchestrator.is_running = False
            batch_orchestrator.should_stop = False
            batch_orchestrator.pipeline_classes = {}  # Empty to avoid heavy imports
            logger.info("🔧 Initialized minimal batch orchestrator (no pipeline imports)")
    return batch_orchestrator

def convert_db_job_to_batch_job(db_job: Job) -> BatchJob:
    """Convert database job to batch job format."""
    return BatchJob(
        job_id=str(db_job.id),
        video_path=Path(db_job.video_path),
        output_dir=Path("api_results") / "jobs" / str(db_job.id),
        config=db_job.config or {},
        selected_pipelines=db_job.selected_pipelines,
        status=BatchJobStatus(db_job.status)
    )

def convert_batch_status_to_db_status(batch_status: BatchJobStatus) -> str:
    """Convert batch job status to database job status."""
    status_mapping = {
        BatchJobStatus.PENDING: "pending",
        BatchJobStatus.RUNNING: "running", 
        BatchJobStatus.COMPLETED: "completed",
        BatchJobStatus.FAILED: "failed",
        BatchJobStatus.RETRYING: "running",  # Map retrying to running in DB
        BatchJobStatus.CANCELLED: "cancelled"
    }
    return status_mapping.get(batch_status, "pending")

# Job processing function (background task)
async def process_job_background(job_id: str):
    """
    Background task to process a video job using the VideoAnnotator batch system.
    """
    from src.database.database import SessionLocal
    
    db = SessionLocal()
    try:
        # Get job from database
        job = JobCRUD.get_by_id(db, job_id)
        if not job:
            logger.error(f"Job {job_id} not found for processing")
            return
        
        logger.info(f"🔄 Processing job {job_id}: {job.video_filename}")
        
        # Update job status to running
        JobCRUD.update_status(
            db=db,
            job_id=job_id,
            status="running",
            progress_percentage=0
        )
        
        # Get batch orchestrator
        orchestrator = get_batch_orchestrator()
        
        # Convert database job to batch job
        batch_job = convert_db_job_to_batch_job(job)
        
        # Add job to orchestrator
        orchestrator.clear_jobs()  # Clear any previous jobs
        orchestrator.jobs = [batch_job]  # Add current job
        
        # Create progress tracking callback
        async def update_progress(current_job: BatchJob):
            """Update database with current job progress."""
            try:
                db_status = convert_batch_status_to_db_status(current_job.status)
                
                # Calculate progress percentage based on pipeline completion
                progress = 0
                if current_job.selected_pipelines:
                    completed_pipelines = sum(
                        1 for result in current_job.pipeline_results.values()
                        if result.status == BatchJobStatus.COMPLETED
                    )
                    progress = int((completed_pipelines / len(current_job.selected_pipelines)) * 100)
                elif current_job.status == BatchJobStatus.RUNNING:
                    progress = 50  # Default progress for running jobs
                elif current_job.status == BatchJobStatus.COMPLETED:
                    progress = 100
                
                JobCRUD.update_status(
                    db=db,
                    job_id=job_id,
                    status=db_status,
                    progress_percentage=progress,
                    error_message=current_job.error_message
                )
                
            except Exception as e:
                logger.error(f"Error updating progress for job {job_id}: {e}")
        
        # Process the job using the batch orchestrator
        try:
            # Run batch processing (synchronous)
            report = await orchestrator.start(max_workers=1, save_checkpoints=True)
            
            # Get the processed job
            processed_job = orchestrator.jobs[0] if orchestrator.jobs else batch_job
            
            # Update database with final results
            final_status = convert_batch_status_to_db_status(processed_job.status)
            
            if processed_job.status == BatchJobStatus.COMPLETED:
                # Job completed successfully
                result_path = str(processed_job.output_dir / "annotations.json")
                JobCRUD.update_results(
                    db=db,
                    job_id=job_id,
                    result_path=result_path,
                    job_metadata={
                        "batch_id": report.batch_id,
                        "processing_duration": processed_job.duration,
                        "pipelines_completed": list(processed_job.pipeline_results.keys()),
                        "annotation_counts": {
                            name: result.annotation_count 
                            for name, result in processed_job.pipeline_results.items()
                            if result.annotation_count is not None
                        }
                    }
                )
                
                JobCRUD.update_status(
                    db=db,
                    job_id=job_id,
                    status="completed",
                    progress_percentage=100
                )
                
                logger.info(f"✅ Completed job {job_id} successfully")
                
            else:
                # Job failed
                JobCRUD.update_status(
                    db=db,
                    job_id=job_id,
                    status="failed",
                    error_message=processed_job.error_message or "Job failed during batch processing"
                )
                
                logger.error(f"❌ Job {job_id} failed: {processed_job.error_message}")
                
        except Exception as processing_error:
            logger.error(f"❌ Batch processing error for job {job_id}: {processing_error}")
            
            # Mark job as failed in database
            JobCRUD.update_status(
                db=db,
                job_id=job_id,
                status="failed",
                error_message=str(processing_error)
            )
            
    except Exception as e:
        logger.error(f"❌ Error setting up processing for job {job_id}: {e}")
        # Mark job as failed
        try:
            JobCRUD.update_status(
                db=db,
                job_id=job_id,
                status="failed",
                error_message=str(e)
            )
        except Exception as db_error:
            logger.error(f"Failed to update job status in database: {db_error}")
    finally:
        db.close()

# Job endpoints
@app.post("/api/v1/jobs", response_model=JobResponse)
async def submit_job(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    config: Optional[str] = Form(None),
    selected_pipelines: Optional[str] = Form(None),
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Submit a video processing job."""
    try:
        # Parse config and pipelines
        import json
        parsed_config = json.loads(config) if config else {}
        parsed_pipelines = [p.strip() for p in selected_pipelines.split(",")] if selected_pipelines else []
        
        # Save video to temp location
        temp_dir = Path(tempfile.mkdtemp())
        video_path = temp_dir / video.filename
        
        # Write video content
        content = await video.read()
        with open(video_path, "wb") as f:
            f.write(content)
        
        # Create job in database
        job = JobCRUD.create(
            db=db,
            user_id=str(user.id) if user else None,
            video_path=str(video_path),
            video_filename=video.filename,
            selected_pipelines=parsed_pipelines,
            config=parsed_config,
            job_metadata={
                "video_size_bytes": len(content),
                "uploaded_at": datetime.utcnow().isoformat(),
                "temp_directory": str(temp_dir)
            }
        )
        
        # Start background processing
        background_tasks.add_task(process_job_background, str(job.id))
        
        logger.info(f"📤 Submitted job {job.id} for video: {video.filename}")
        
        return JobResponse(**job.to_dict())
        
    except Exception as e:
        logger.error(f"Error submitting job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(
    job_id: str,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get job status by ID."""
    job = JobCRUD.get_by_id(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check if user has access to this job (if authenticated)
    if user and job.user_id and str(job.user_id) != str(user.id) and not user.is_admin:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return JobResponse(**job.to_dict())

@app.get("/api/v1/jobs", response_model=JobListResponse)
async def list_jobs(
    limit: int = 100,
    offset: int = 0,
    status: Optional[str] = None,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List jobs with pagination."""
    if user and not user.is_admin:
        # Regular users can only see their own jobs
        jobs = JobCRUD.get_by_user(db, str(user.id), limit, offset)
    elif user and user.is_admin:
        # Admin users can see all jobs
        if status:
            jobs = JobCRUD.get_by_status(db, status, limit)
        else:
            jobs = JobCRUD.get_all(db, limit, offset)
    else:
        # Anonymous users get limited access
        jobs = []
    
    job_responses = [JobResponse(**job.to_dict()) for job in jobs]
    
    return JobListResponse(
        jobs=job_responses,
        total=len(job_responses),
        page=(offset // limit) + 1,
        per_page=limit
    )

@app.delete("/api/v1/jobs/{job_id}")
async def delete_job(
    job_id: str,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a job."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    job = JobCRUD.get_by_id(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check permissions
    if str(job.user_id) != str(user.id) and not user.is_admin:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Delete job
    success = JobCRUD.delete(db, job_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete job")
    
    return {"message": "Job deleted successfully"}

# Pipeline endpoints
@app.get("/api/v1/pipelines")
async def list_pipelines():
    """List available pipelines."""
    pipelines = [
        PipelineInfo(
            name="scene_detection",
            description="Detect scene boundaries using PySceneDetect + CLIP",
            enabled=True,
            config_schema={
                "threshold": {"type": "float", "default": 30.0},
                "min_scene_length": {"type": "float", "default": 1.0}
            }
        ),
        PipelineInfo(
            name="person_tracking", 
            description="Track people with YOLO11 + ByteTrack",
            enabled=True,
            config_schema={
                "model": {"type": "string", "default": "yolo11n-pose.pt"},
                "conf_threshold": {"type": "float", "default": 0.4}
            }
        ),
        PipelineInfo(
            name="face_analysis",
            description="Face analysis with OpenFace 3.0",
            enabled=True,
            config_schema={
                "backend": {"type": "string", "default": "openface"},
                "confidence_threshold": {"type": "float", "default": 0.5}
            }
        ),
        PipelineInfo(
            name="audio_processing",
            description="Speech recognition with Whisper",
            enabled=True,
            config_schema={
                "whisper_model": {"type": "string", "default": "base"},
                "enable_diarization": {"type": "boolean", "default": True}
            }
        )
    ]
    
    return {"pipelines": pipelines, "total": len(pipelines)}


if __name__ == "__main__":
    print("[VideoAnnotator API Server v1.2.0 - Production Mode]")
    print("[Database: SQLite (development) | PostgreSQL (production)]")
    print("[API Documentation: http://localhost:8000/docs]")
    print("[Authentication: Use API key from database migration]")
    print("=" * 60)
    
    # Set up environment
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    uvicorn.run(
        "api_server_db:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )