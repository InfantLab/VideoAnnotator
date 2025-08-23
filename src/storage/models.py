"""
SQLAlchemy database models for VideoAnnotator.

These models map to the existing BatchJob, PipelineResult, and related data structures
to provide database persistence while maintaining the same interfaces.
"""

from sqlalchemy import Column, String, DateTime, Text, Integer, JSON, ForeignKey, Boolean, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import uuid

Base = declarative_base()


class Job(Base):
    """
    Database model for BatchJob.
    
    Represents a video processing job with its metadata, status, and configuration.
    """
    __tablename__ = "jobs"
    
    # Core job fields
    id = Column(String, primary_key=True)  # job_id from BatchJob
    video_path = Column(String, nullable=False)
    output_dir = Column(String)
    config = Column(JSON, default=dict)  # SQLite 3.38+ supports JSON
    status = Column(String, nullable=False, default="pending")
    selected_pipelines = Column(JSON)  # List[str] as JSON
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Retry and error handling
    retry_count = Column(Integer, default=0)
    error_message = Column(Text)
    
    # Relationships
    pipeline_results = relationship("PipelineResult", back_populates="job", cascade="all, delete-orphan")
    annotations = relationship("Annotation", back_populates="job", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Job(id='{self.id}', status='{self.status}', video='{self.video_path}')>"


class PipelineResult(Base):
    """
    Database model for individual pipeline execution results.
    
    Each job can have multiple pipeline results (scene, person, face, audio, etc.)
    """
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

    def __repr__(self):
        return f"<PipelineResult(job='{self.job_id}', pipeline='{self.pipeline_name}', status='{self.status}')>"


class Annotation(Base):
    """
    Database model for storing pipeline annotation data.
    
    Stores the actual annotation content as JSON for flexible schema support.
    """
    __tablename__ = "annotations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String, ForeignKey("jobs.id"), nullable=False) 
    pipeline = Column(String, nullable=False)
    
    # Annotation data stored as JSON
    data = Column(JSON, nullable=False)  # The actual annotation content
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    job = relationship("Job", back_populates="annotations")

    def __repr__(self):
        return f"<Annotation(job='{self.job_id}', pipeline='{self.pipeline}')>"


# Optional tables for future multi-user support (v1.3.0+)
class User(Base):
    """
    User model for multi-user installations.
    
    Currently unused but defined for future enterprise features.
    """
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}')>"


class ApiKey(Base):
    """
    API key model for authentication.
    
    Currently unused but defined for future enterprise features.
    """
    __tablename__ = "api_keys"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"))
    name = Column(String)  # User-friendly name for the key
    key_hash = Column(String, nullable=False)  # bcrypt hash of actual key
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_used_at = Column(DateTime)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")

    def __repr__(self):
        return f"<ApiKey(name='{self.name}', user='{self.user_id}')>"


class BatchReport(Base):
    """
    Database model for batch processing reports.
    
    Stores summary information about batch processing operations.
    """
    __tablename__ = "batch_reports"
    
    id = Column(String, primary_key=True)  # batch_id
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    total_jobs = Column(Integer, default=0)
    completed_jobs = Column(Integer, default=0)
    failed_jobs = Column(Integer, default=0)
    cancelled_jobs = Column(Integer, default=0)
    total_processing_time = Column(Integer, default=0)  # milliseconds
    
    # Store additional report data as JSON
    report_data = Column(JSON)  # For jobs list, errors, etc.
    
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f"<BatchReport(id='{self.id}', total_jobs={self.total_jobs}, completed={self.completed_jobs})>"


# Database session management
def create_database_engine(database_url: str, echo: bool = False):
    """
    Create SQLAlchemy engine with appropriate configuration.
    
    Args:
        database_url: Database connection URL (sqlite:/// or postgresql://)
        echo: Whether to log SQL queries (useful for debugging)
    
    Returns:
        SQLAlchemy engine instance
    """
    if database_url.startswith("sqlite"):
        import json as json_lib
        # SQLite-specific configuration
        engine = create_engine(
            database_url,
            echo=echo,
            # Enable JSON support for SQLite 3.38+
            connect_args={"check_same_thread": False},  # Allow multiple threads
            json_serializer=json_lib.dumps,  # Proper JSON serialization
            json_deserializer=json_lib.loads  # Proper JSON deserialization
        )
    else:
        # PostgreSQL or other database configuration
        import json as json_lib
        from sqlalchemy.pool import QueuePool
        engine = create_engine(
            database_url,
            echo=echo,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,  # Verify connections before use
            json_serializer=json_lib.dumps,
            json_deserializer=json_lib.loads
        )
    
    return engine


def create_session_factory(engine):
    """
    Create SQLAlchemy session factory.
    
    Args:
        engine: SQLAlchemy engine
        
    Returns:
        Session factory (sessionmaker instance)
    """
    return sessionmaker(bind=engine, autoflush=True, autocommit=False)


def initialize_database(engine):
    """
    Initialize database by creating all tables.
    
    Args:
        engine: SQLAlchemy engine
    """
    Base.metadata.create_all(engine)


# Database schema version management
CURRENT_SCHEMA_VERSION = 1

class SchemaVersion(Base):
    """Track database schema version for migrations."""
    __tablename__ = "schema_version"
    
    version = Column(Integer, primary_key=True)
    applied_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    description = Column(String)

    def __repr__(self):
        return f"<SchemaVersion(version={self.version}, description='{self.description}')>"