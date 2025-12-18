# VideoAnnotator Database Implementation Plan v1.2.0

## Overview

This document outlines the implementation plan for adding database support to VideoAnnotator v1.2.0, prioritizing the primary use case of individual researchers who need zero-configuration local installations.

## Design Philosophy

### Primary Use Case (90% of users): Individual Researchers

- **Zero Configuration**: Database auto-creates on first run
- **Single File**: Entire project contained in one portable database file
- **No Services**: No separate database servers to manage
- **Researcher-Friendly**: Backup = copy file, share = send file

### Secondary Use Case (10% of users): Enterprise Labs

- **Optional PostgreSQL**: For multi-user labs requiring advanced features
- **Same Interface**: Identical API regardless of backend
- **Configuration-Based**: Environment variables control backend selection

## Current State Analysis

### ✅ Existing Infrastructure

- **Rich Data Models**: `BatchJob`, `PipelineResult`, `JobStatus` in `src/batch/types.py`
- **Storage Interface**: `StorageBackend` abstract base class in `src/storage/base.py`
- **File Backend**: Complete file-based implementation in `src/storage/file_backend.py`
- **API Structure**: FastAPI endpoints expect persistent storage
- **Serialization**: Complete to_dict/from_dict methods for all data types

### ❌ Missing Components

- **Database Backend**: No SQL-based storage backend
- **Schema Definitions**: No database tables/models
- **Migrations**: No schema versioning system
- **API Integration**: API currently uses mock in-memory storage

## Implementation Plan

### Phase 1: SQLite Backend (Week 1)

**Goal**: Get API working with persistent SQLite storage

#### 1.1 Database Models (Day 1-2)

Create SQLAlchemy models matching existing data structures:

**File**: `src/storage/models.py`

```python
from sqlalchemy import Column, String, DateTime, Text, Integer, JSON, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()

class Job(Base):
    __tablename__ = "jobs"

    # Core job fields
    id = Column(String, primary_key=True)  # job_id from BatchJob
    video_path = Column(String, nullable=False)
    output_dir = Column(String)
    config = Column(JSON)  # SQLite 3.38+ supports JSON
    status = Column(String, nullable=False, default="pending")
    selected_pipelines = Column(JSON)  # List[str] as JSON

    # Timestamps
    created_at = Column(DateTime, nullable=False)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

    # Retry and error handling
    retry_count = Column(Integer, default=0)
    error_message = Column(Text)

    # Relationships
    pipeline_results = relationship("PipelineResult", back_populates="job", cascade="all, delete-orphan")
    annotations = relationship("Annotation", back_populates="job", cascade="all, delete-orphan")

class PipelineResult(Base):
    __tablename__ = "pipeline_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String, ForeignKey("jobs.id"), nullable=False)
    pipeline_name = Column(String, nullable=False)
    status = Column(String, nullable=False)

    # Timing information
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    processing_time = Column(Integer)  # milliseconds

    # Result metadata
    annotation_count = Column(Integer)
    output_file = Column(String)  # Path to result file
    error_message = Column(Text)

    # Relationships
    job = relationship("Job", back_populates="pipeline_results")

class Annotation(Base):
    __tablename__ = "annotations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String, ForeignKey("jobs.id"), nullable=False)
    pipeline = Column(String, nullable=False)

    # Annotation data stored as JSON
    data = Column(JSON, nullable=False)  # The actual annotation content
    created_at = Column(DateTime, nullable=False)

    # Relationships
    job = relationship("Job", back_populates="annotations")

# Optional future tables for multi-user support
class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True)
    created_at = Column(DateTime, nullable=False)
    is_active = Column(Boolean, default=True)

class ApiKey(Base):
    __tablename__ = "api_keys"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"))
    name = Column(String)  # User-friendly name
    key_hash = Column(String, nullable=False)  # bcrypt hash of actual key
    created_at = Column(DateTime, nullable=False)
    last_used_at = Column(DateTime)
    is_active = Column(Boolean, default=True)
```

#### 1.2 SQLite Storage Backend (Day 2-3)

Create SQLite implementation of `StorageBackend`:

**File**: `src/storage/sqlite_backend.py`

```python
from pathlib import Path
from typing import Dict, Any, List, Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import json

from .base import StorageBackend
from .models import Base, Job, PipelineResult, Annotation
from ..batch.types import BatchJob, JobStatus, BatchReport

class SQLiteStorageBackend(StorageBackend):
    """SQLite-based storage backend for local research installations."""

    def __init__(self, database_path: Path = None):
        """
        Initialize SQLite storage backend.

        Args:
            database_path: Path to SQLite database file.
                          Defaults to ./videoannotator.db in current directory.
        """
        if database_path is None:
            database_path = Path.cwd() / "videoannotator.db"

        self.database_path = Path(database_path)
        self.database_url = f"sqlite:///{self.database_path}"

        # Create database and tables
        self.engine = create_engine(
            self.database_url,
            json_serializer=json.dumps,  # Handle JSON fields properly
            json_deserializer=json.loads,
            echo=False  # Set to True for SQL debugging
        )

        # Create all tables
        Base.metadata.create_all(self.engine)

        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine)

    def _batch_job_to_db_job(self, batch_job: BatchJob) -> Job:
        """Convert BatchJob to database Job model."""
        return Job(
            id=batch_job.job_id,
            video_path=str(batch_job.video_path) if batch_job.video_path else None,
            output_dir=str(batch_job.output_dir) if batch_job.output_dir else None,
            config=batch_job.config,
            status=batch_job.status.value,
            selected_pipelines=batch_job.selected_pipelines,
            created_at=batch_job.created_at,
            started_at=batch_job.started_at,
            completed_at=batch_job.completed_at,
            retry_count=batch_job.retry_count,
            error_message=batch_job.error_message
        )

    def _db_job_to_batch_job(self, db_job: Job) -> BatchJob:
        """Convert database Job model to BatchJob."""
        batch_job = BatchJob(
            job_id=db_job.id,
            video_path=Path(db_job.video_path) if db_job.video_path else None,
            output_dir=Path(db_job.output_dir) if db_job.output_dir else None,
            config=db_job.config or {},
            status=JobStatus(db_job.status),
            created_at=db_job.created_at,
            started_at=db_job.started_at,
            completed_at=db_job.completed_at,
            retry_count=db_job.retry_count,
            error_message=db_job.error_message,
            selected_pipelines=db_job.selected_pipelines
        )

        # Load pipeline results
        for result in db_job.pipeline_results:
            pipeline_result = PipelineResult(
                pipeline_name=result.pipeline_name,
                status=JobStatus(result.status),
                start_time=result.start_time,
                end_time=result.end_time,
                processing_time=result.processing_time,
                annotation_count=result.annotation_count,
                output_file=Path(result.output_file) if result.output_file else None,
                error_message=result.error_message
            )
            batch_job.pipeline_results[result.pipeline_name] = pipeline_result

        return batch_job

    def save_job_metadata(self, job: BatchJob) -> None:
        """Save job metadata to database."""
        with self.SessionLocal() as session:
            # Check if job exists
            existing = session.query(Job).filter_by(id=job.job_id).first()

            if existing:
                # Update existing job
                existing.status = job.status.value
                existing.started_at = job.started_at
                existing.completed_at = job.completed_at
                existing.retry_count = job.retry_count
                existing.error_message = job.error_message

                # Update pipeline results
                # Delete old results and create new ones (simpler than complex updates)
                session.query(PipelineResult).filter_by(job_id=job.job_id).delete()

                for name, result in job.pipeline_results.items():
                    db_result = PipelineResult(
                        job_id=job.job_id,
                        pipeline_name=result.pipeline_name,
                        status=result.status.value,
                        start_time=result.start_time,
                        end_time=result.end_time,
                        processing_time=result.processing_time,
                        annotation_count=result.annotation_count,
                        output_file=str(result.output_file) if result.output_file else None,
                        error_message=result.error_message
                    )
                    session.add(db_result)
            else:
                # Create new job
                db_job = self._batch_job_to_db_job(job)
                session.add(db_job)

                # Add pipeline results
                for name, result in job.pipeline_results.items():
                    db_result = PipelineResult(
                        job_id=job.job_id,
                        pipeline_name=result.pipeline_name,
                        status=result.status.value,
                        start_time=result.start_time,
                        end_time=result.end_time,
                        processing_time=result.processing_time,
                        annotation_count=result.annotation_count,
                        output_file=str(result.output_file) if result.output_file else None,
                        error_message=result.error_message
                    )
                    session.add(db_result)

            session.commit()

    def load_job_metadata(self, job_id: str) -> BatchJob:
        """Load job metadata from database."""
        with self.SessionLocal() as session:
            db_job = session.query(Job).filter_by(id=job_id).first()
            if not db_job:
                raise FileNotFoundError(f"Job {job_id} not found")

            return self._db_job_to_batch_job(db_job)

    def save_annotations(self, job_id: str, pipeline: str, annotations: List[Dict[str, Any]]) -> str:
        """Save pipeline annotations to database."""
        with self.SessionLocal() as session:
            # Delete existing annotations for this job+pipeline
            session.query(Annotation).filter_by(job_id=job_id, pipeline=pipeline).delete()

            # Save new annotations
            for annotation in annotations:
                db_annotation = Annotation(
                    job_id=job_id,
                    pipeline=pipeline,
                    data=annotation,
                    created_at=datetime.now()
                )
                session.add(db_annotation)

            session.commit()
            return f"database://annotations/{job_id}/{pipeline}"

    def load_annotations(self, job_id: str, pipeline: str) -> List[Dict[str, Any]]:
        """Load pipeline annotations from database."""
        with self.SessionLocal() as session:
            db_annotations = session.query(Annotation).filter_by(
                job_id=job_id, pipeline=pipeline
            ).all()

            if not db_annotations:
                raise FileNotFoundError(f"Annotations not found for {job_id}/{pipeline}")

            return [ann.data for ann in db_annotations]

    def annotation_exists(self, job_id: str, pipeline: str) -> bool:
        """Check if annotations exist for job and pipeline."""
        with self.SessionLocal() as session:
            count = session.query(Annotation).filter_by(
                job_id=job_id, pipeline=pipeline
            ).count()
            return count > 0

    def list_jobs(self, status_filter: Optional[str] = None) -> List[str]:
        """List all job IDs, optionally filtered by status."""
        with self.SessionLocal() as session:
            query = session.query(Job.id)

            if status_filter:
                query = query.filter(Job.status == status_filter)

            return [row[0] for row in query.all()]

    def delete_job(self, job_id: str) -> None:
        """Delete all data for a job."""
        with self.SessionLocal() as session:
            # Foreign key constraints will cascade delete annotations and pipeline_results
            session.query(Job).filter_by(id=job_id).delete()
            session.commit()

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        with self.SessionLocal() as session:
            total_jobs = session.query(Job).count()
            pending_jobs = session.query(Job).filter_by(status="pending").count()
            running_jobs = session.query(Job).filter_by(status="running").count()
            completed_jobs = session.query(Job).filter_by(status="completed").count()
            failed_jobs = session.query(Job).filter_by(status="failed").count()

            total_annotations = session.query(Annotation).count()

            return {
                "backend": "sqlite",
                "database_path": str(self.database_path),
                "database_size_mb": round(self.database_path.stat().st_size / (1024*1024), 2) if self.database_path.exists() else 0,
                "total_jobs": total_jobs,
                "pending_jobs": pending_jobs,
                "running_jobs": running_jobs,
                "completed_jobs": completed_jobs,
                "failed_jobs": failed_jobs,
                "total_annotations": total_annotations,
            }

    # Implement remaining abstract methods...
    def save_report(self, report: BatchReport) -> None:
        # Implementation for batch reports
        pass

    def load_report(self, batch_id: str) -> BatchReport:
        # Implementation for batch reports
        pass

    def list_reports(self) -> List[str]:
        # Implementation for batch reports
        pass
```

#### 1.3 API Integration (Day 3-4)

Update API to use real storage backend instead of mock:

**File**: `src/api/database.py`

```python
from functools import lru_cache
from pathlib import Path
import os
from ..storage.sqlite_backend import SQLiteStorageBackend
from ..storage.base import StorageBackend

@lru_cache()
def get_storage_backend() -> StorageBackend:
    """Get storage backend based on configuration."""

    # Check for database URL in environment
    database_url = os.environ.get("DATABASE_URL")

    if database_url and database_url.startswith("postgresql://"):
        # Future: PostgreSQL backend
        from ..storage.postgresql_backend import PostgreSQLStorageBackend
        return PostgreSQLStorageBackend(database_url)

    else:
        # Default: SQLite backend
        db_path = os.environ.get("VIDEOANNOTATOR_DB_PATH")
        if db_path:
            return SQLiteStorageBackend(Path(db_path))
        else:
            # Default: videoannotator.db in current directory
            return SQLiteStorageBackend()
```

**Update**: `src/api/v1/jobs.py`

```python
# Replace mock classes with real storage
from ..database import get_storage_backend
from ...batch.batch_orchestrator import BatchOrchestrator

def get_batch_orchestrator() -> BatchOrchestrator:
    """Get batch orchestrator with real storage backend."""
    storage = get_storage_backend()
    return BatchOrchestrator(storage_backend=storage)
```

#### 1.4 CLI Integration (Day 4-5)

Update CLI to show database information:

**Update**: `src/cli.py`

```python
@app.command()
def info():
    """Show VideoAnnotator system information."""
    from .api.database import get_storage_backend

    storage = get_storage_backend()
    stats = storage.get_stats()

    typer.echo(f"VideoAnnotator v{__version__}")
    typer.echo(f"Database: {stats['backend']}")

    if stats['backend'] == 'sqlite':
        typer.echo(f"Database file: {stats['database_path']}")
        typer.echo(f"Database size: {stats['database_size_mb']} MB")

    typer.echo(f"Total jobs: {stats['total_jobs']}")
    typer.echo(f"  Pending: {stats['pending_jobs']}")
    typer.echo(f"  Running: {stats['running_jobs']}")
    typer.echo(f"  Completed: {stats['completed_jobs']}")
    typer.echo(f"  Failed: {stats['failed_jobs']}")
    typer.echo(f"Total annotations: {stats['total_annotations']}")

@app.command()
def backup(output_path: Path):
    """Backup database to specified location."""
    from .api.database import get_storage_backend
    import shutil

    storage = get_storage_backend()
    if hasattr(storage, 'database_path'):
        shutil.copy2(storage.database_path, output_path)
        typer.echo(f"Database backed up to: {output_path}")
    else:
        typer.echo("Backup only supported for SQLite databases", err=True)
```

### Phase 2: PostgreSQL Backend (Week 2-3)

**Goal**: Add enterprise-grade PostgreSQL support as optional backend

#### 2.1 PostgreSQL Backend Implementation

Create PostgreSQL implementation using same interface:

**File**: `src/storage/postgresql_backend.py`

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
import os

class PostgreSQLStorageBackend(StorageBackend):
    """PostgreSQL-based storage backend for enterprise installations."""

    def __init__(self, database_url: str):
        """Initialize PostgreSQL backend with connection URL."""
        self.database_url = database_url

        # Create engine with connection pooling
        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,  # Verify connections
            json_serializer=json.dumps,
            json_deserializer=json.loads
        )

        # Create tables (in production, use migrations)
        Base.metadata.create_all(self.engine)

        self.SessionLocal = sessionmaker(bind=self.engine)

    # Implementation follows same pattern as SQLite backend
    # but with PostgreSQL-specific optimizations
```

#### 2.2 Database Migrations System

Create simple migration system for schema updates:

**File**: `src/storage/migrations.py`

```python
from pathlib import Path
from typing import Dict, Callable
import logging

class DatabaseMigrations:
    """Simple database migration system."""

    def __init__(self, storage_backend: StorageBackend):
        self.storage = storage_backend
        self.migrations: Dict[int, Callable] = {}

    def register_migration(self, version: int):
        """Decorator to register migration functions."""
        def decorator(func: Callable):
            self.migrations[version] = func
            return func
        return decorator

    def get_current_version(self) -> int:
        """Get current database schema version."""
        # Implementation to track schema version
        pass

    def apply_migrations(self) -> None:
        """Apply any pending migrations."""
        current = self.get_current_version()
        target = max(self.migrations.keys())

        for version in range(current + 1, target + 1):
            if version in self.migrations:
                logging.info(f"Applying migration {version}")
                self.migrations[version](self.storage)

# Example migration
@migrations.register_migration(1)
def add_user_tables(storage: StorageBackend):
    """Add user management tables for v1.2.1"""
    # SQL to add user and api_key tables
    pass
```

#### 2.3 Docker Compose Integration

Add PostgreSQL to development docker-compose:

**Update**: `docker-compose.yml`

```yaml
services:
  # Add PostgreSQL service
  videoannotator-postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: videoannotator
      POSTGRES_USER: videoannotator
      POSTGRES_PASSWORD: videoannotator_dev
      POSTGRES_HOST_AUTH_METHOD: trust
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    profiles:
      - postgres

  # Update API service to use PostgreSQL when profile is active
  videoannotator-api-postgres:
    build:
      context: .
      target: development
    environment:
      - DATABASE_URL=postgresql://videoannotator:videoannotator_dev@videoannotator-postgres:5432/videoannotator
    depends_on:
      - videoannotator-postgres
    command: ["uv", "run", "python", "-m", "src.cli", "server"]
    profiles:
      - postgres

volumes:
  postgres_data:
```

### Phase 3: Testing & Documentation (Week 3)

**Goal**: Comprehensive testing and user documentation

#### 3.1 Database Backend Tests

Create comprehensive test suite:

**File**: `tests/unit/storage/test_sqlite_backend.py`

```python
import pytest
from pathlib import Path
import tempfile
from src.storage.sqlite_backend import SQLiteStorageBackend
from src.batch.types import BatchJob, JobStatus

class TestSQLiteBackend:
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
            db_path = Path(f.name)

        yield db_path

        # Cleanup
        if db_path.exists():
            db_path.unlink()

    @pytest.fixture
    def storage(self, temp_db):
        """Create storage backend with temporary database."""
        return SQLiteStorageBackend(temp_db)

    def test_job_crud_operations(self, storage):
        """Test basic CRUD operations for jobs."""
        # Create test job
        job = BatchJob(
            job_id="test-job-1",
            video_path=Path("test_video.mp4"),
            config={"test": True},
            status=JobStatus.PENDING
        )

        # Save job
        storage.save_job_metadata(job)

        # Load job
        loaded_job = storage.load_job_metadata("test-job-1")
        assert loaded_job.job_id == job.job_id
        assert loaded_job.video_path == job.video_path
        assert loaded_job.config == job.config
        assert loaded_job.status == job.status

        # Update job
        job.status = JobStatus.COMPLETED
        storage.save_job_metadata(job)

        # Verify update
        updated_job = storage.load_job_metadata("test-job-1")
        assert updated_job.status == JobStatus.COMPLETED

        # Delete job
        storage.delete_job("test-job-1")

        # Verify deletion
        with pytest.raises(FileNotFoundError):
            storage.load_job_metadata("test-job-1")

    def test_annotations_crud(self, storage):
        """Test annotation storage and retrieval."""
        # Setup job first
        job = BatchJob(job_id="test-job-2", video_path=Path("test.mp4"))
        storage.save_job_metadata(job)

        # Test annotations
        annotations = [
            {"type": "detection", "bbox": [100, 100, 200, 200]},
            {"type": "keypoint", "points": [[150, 150], [160, 160]]}
        ]

        # Save annotations
        result_path = storage.save_annotations("test-job-2", "person_tracking", annotations)
        assert "database://" in result_path

        # Load annotations
        loaded_annotations = storage.load_annotations("test-job-2", "person_tracking")
        assert loaded_annotations == annotations

        # Test existence check
        assert storage.annotation_exists("test-job-2", "person_tracking") == True
        assert storage.annotation_exists("test-job-2", "face_analysis") == False

    def test_database_stats(self, storage):
        """Test database statistics."""
        # Create some test data
        for i in range(3):
            job = BatchJob(
                job_id=f"job-{i}",
                video_path=Path(f"video{i}.mp4"),
                status=JobStatus.PENDING if i < 2 else JobStatus.COMPLETED
            )
            storage.save_job_metadata(job)

        # Get stats
        stats = storage.get_stats()

        assert stats["backend"] == "sqlite"
        assert stats["total_jobs"] == 3
        assert stats["pending_jobs"] == 2
        assert stats["completed_jobs"] == 1
        assert "database_path" in stats
        assert "database_size_mb" in stats
```

#### 3.2 Integration Tests

Test API with real database:

**File**: `tests/integration/test_api_with_database.py`

```python
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import os

@pytest.fixture
def temp_db_api():
    """Create API client with temporary database."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
        db_path = Path(f.name)

    # Set environment variable for test database
    os.environ["VIDEOANNOTATOR_DB_PATH"] = str(db_path)

    from src.api.main import create_app
    app = create_app()
    client = TestClient(app)

    yield client

    # Cleanup
    if db_path.exists():
        db_path.unlink()
    del os.environ["VIDEOANNOTATOR_DB_PATH"]

def test_api_job_lifecycle(temp_db_api):
    """Test complete job lifecycle through API."""
    client = temp_db_api

    # Test job submission
    with open("tests/data/sample_video.mp4", "rb") as video_file:
        response = client.post(
            "/api/v1/jobs/",
            files={"video": video_file},
            data={"selected_pipelines": "scene,person"}
        )

    assert response.status_code == 201
    job_data = response.json()
    job_id = job_data["id"]

    # Test job retrieval
    response = client.get(f"/api/v1/jobs/{job_id}")
    assert response.status_code == 200

    retrieved_job = response.json()
    assert retrieved_job["id"] == job_id
    assert retrieved_job["status"] == "pending"
    assert retrieved_job["selected_pipelines"] == ["scene", "person"]

    # Test job listing
    response = client.get("/api/v1/jobs/")
    assert response.status_code == 200

    jobs_list = response.json()
    assert jobs_list["total"] == 1
    assert len(jobs_list["jobs"]) == 1
    assert jobs_list["jobs"][0]["id"] == job_id
```

#### 3.3 User Documentation

Create user-facing documentation:

**File**: `docs/usage/database.md`

````markdown
# Database Configuration

VideoAnnotator v1.2.0 uses a database to store job information, processing results, and system state. The database backend is configurable to support different deployment scenarios.

## Default Configuration (Recommended for Most Users)

By default, VideoAnnotator uses a local SQLite database file:

```bash
# This creates videoannotator.db in your current directory
videoannotator server
```
````

### Database Location

The database file is created in your current working directory:

- **Database file**: `./videoannotator.db`
- **Backup**: Just copy the file: `cp videoannotator.db backup_2024.db`
- **Share project**: Send the `.db` file to collaborators
- **Archive**: Move `.db` file to archive folder when project complete

### Researcher Workflow

```bash
# Start new research project
mkdir infant_language_study
cd infant_language_study

# Start VideoAnnotator (creates videoannotator.db here)
videoannotator server

# Upload videos via web interface at http://localhost:8000
# Process videos and download results
# Database contains all job history and results

# Later: Backup project
cp videoannotator.db ../backups/infant_study_complete_2024.db
```

## Advanced Configuration

### Custom Database Location

```bash
# Use specific database file
export VIDEOANNOTATOR_DB_PATH="/path/to/my_project.db"
videoannotator server

# Or specify when running
VIDEOANNOTATOR_DB_PATH="/path/to/my_project.db" videoannotator server
```

### PostgreSQL (Multi-User Labs)

For research labs needing multi-user access:

```bash
# Start PostgreSQL (requires Docker)
docker run -d --name videoannotator-postgres \
  -e POSTGRES_DB=videoannotator \
  -e POSTGRES_USER=videoannotator \
  -e POSTGRES_PASSWORD=your_password \
  -p 5432:5432 postgres:15

# Configure VideoAnnotator to use PostgreSQL
export DATABASE_URL="postgresql://videoannotator:your_password@localhost:5432/videoannotator"
videoannotator server
```

## Database Management

### View Database Statistics

```bash
videoannotator info
```

### Backup Database

```bash
# Backup SQLite database
videoannotator backup my_project_backup.db

# Or just copy the file
cp videoannotator.db backups/project_$(date +%Y%m%d).db
```

### Migration Between Backends

VideoAnnotator can export/import projects between SQLite and PostgreSQL:

```bash
# Export from current database
videoannotator export project_data.json

# Import to different database
DATABASE_URL="postgresql://..." videoannotator import project_data.json
```

## Troubleshooting

### Database Locked Errors

If you see "database is locked" errors:

1. Make sure only one VideoAnnotator instance is running
2. Check that no other process has the database file open
3. Restart VideoAnnotator server

### Database Corruption

If database becomes corrupted:

1. Restore from backup: `cp backup.db videoannotator.db`
2. Or start fresh: `rm videoannotator.db` (loses all data!)
3. VideoAnnotator will create new empty database on next start

```

### Phase 4: Production Readiness (Week 4)
**Goal**: Polish and optimize for production deployment

#### 4.1 Performance Optimization
- Database connection pooling
- Query optimization
- Index creation for common queries
- Background task cleanup

#### 4.2 Security Hardening
- SQL injection prevention (SQLAlchemy handles this)
- Database file permissions
- Connection encryption for PostgreSQL

#### 4.3 Monitoring and Observability
- Database health checks
- Performance metrics
- Error tracking and logging

## Success Metrics

### Functional Requirements
- ✅ API can submit, track, and retrieve jobs using database
- ✅ Database persists across server restarts
- ✅ Zero configuration for individual researchers
- ✅ Optional PostgreSQL for enterprise users
- ✅ Data migration between backends

### Performance Requirements
- ✅ Job submission < 500ms
- ✅ Job status queries < 100ms
- ✅ Support 100+ concurrent jobs in SQLite
- ✅ Support 1000+ concurrent jobs in PostgreSQL

### User Experience Requirements
- ✅ Single command setup: `videoannotator server`
- ✅ Easy backup: Copy database file
- ✅ Clear error messages for database issues
- ✅ Graceful handling of database unavailability

## Risk Mitigation

### Technical Risks
- **SQLite limitations**: Mitigated by PostgreSQL option for high-load scenarios
- **Schema changes**: Mitigated by migration system
- **Data corruption**: Mitigated by backup/restore functionality
- **Performance degradation**: Mitigated by indexing and query optimization

### User Adoption Risks
- **Complexity**: Mitigated by zero-configuration default
- **Migration anxiety**: Mitigated by export/import tools
- **Data loss fears**: Mitigated by simple backup (copy file)

## Future Enhancements (Post v1.2.0)

### v1.3.0 Candidates
- **Cloud storage integration**: S3/Google Cloud for video and results
- **Real-time collaboration**: Multiple users on same project
- **Advanced analytics**: Query interface for research data
- **Data governance**: Retention policies, compliance features

### v2.0.0 Candidates
- **Distributed processing**: Multiple worker nodes
- **Advanced search**: Full-text search across annotations
- **API rate limiting**: User quotas and throttling
- **Audit logging**: Complete activity tracking

---

*This implementation plan prioritizes the 90% use case (individual researchers) while maintaining a clear path to enterprise features for the 10% who need them.*
```
