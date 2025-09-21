"""
Database models for VideoAnnotator API server.
"""

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import Column, String, DateTime, Text, JSON, Boolean, Integer, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.types import TypeDecorator, CHAR
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from .database import Base


class GUID(TypeDecorator):
    """Platform-independent GUID type that uses PostgreSQL UUID or String for other databases."""
    
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(PG_UUID())
        else:
            return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return str(value)
        else:
            if not isinstance(value, uuid.UUID):
                return str(value)
            else:
                return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        else:
            if not isinstance(value, uuid.UUID):
                return uuid.UUID(value)
            return value


class User(Base):
    """User model for authentication and authorization."""
    
    __tablename__ = "users"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    full_name = Column(String(200), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    jobs = relationship("Job", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, email={self.email})>"


class APIKey(Base):
    """API key model for authentication."""
    
    __tablename__ = "api_keys"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4, index=True)
    key_name = Column(String(100), nullable=False)
    key_hash = Column(String(255), nullable=False, index=True)  # Hashed API key
    key_prefix = Column(String(10), nullable=False, index=True)  # First few chars for identification
    user_id = Column(GUID(), ForeignKey("users.id"), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_used = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")
    
    def __repr__(self):
        return f"<APIKey(id={self.id}, name={self.key_name}, prefix={self.key_prefix})>"


class Job(Base):
    """Job model for video processing tasks."""
    
    __tablename__ = "jobs"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(GUID(), ForeignKey("users.id"), nullable=True)  # Allow anonymous jobs
    
    # Video information
    video_path = Column(String(500), nullable=False)
    video_filename = Column(String(255), nullable=True)
    video_size_bytes = Column(Integer, nullable=True)
    video_duration_seconds = Column(Integer, nullable=True)
    
    # Processing configuration
    selected_pipelines = Column(JSON, nullable=True, default=list)  # List of pipeline names
    config = Column(JSON, nullable=True, default=dict)  # Processing configuration
    
    # Status and timing
    status = Column(String(50), nullable=False, default="pending", index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Results and errors
    result_path = Column(String(500), nullable=True)  # Path to result files
    error_message = Column(Text, nullable=True)
    progress_percentage = Column(Integer, default=0, nullable=False)
    
    # Metadata
    job_metadata = Column(JSON, nullable=True, default=dict)  # Additional job metadata
    
    # Relationships
    user = relationship("User", back_populates="jobs")
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate job duration in seconds."""
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            return delta.total_seconds()
        return None
    
    @property
    def is_complete(self) -> bool:
        """Check if job is in a completed state."""
        return self.status in ["completed", "failed", "cancelled"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary representation."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id) if self.user_id else None,
            "video_path": self.video_path,
            "video_filename": self.video_filename,
            "video_size_bytes": self.video_size_bytes,
            "video_duration_seconds": self.video_duration_seconds,
            "selected_pipelines": self.selected_pipelines or [],
            "config": self.config or {},
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "result_path": self.result_path,
            "error_message": self.error_message,
            "progress_percentage": self.progress_percentage,
            "duration_seconds": self.duration_seconds,
            "is_complete": self.is_complete,
            "metadata": self.job_metadata or {}
        }
    
    def __repr__(self):
        return f"<Job(id={self.id}, status={self.status}, video={self.video_filename})>"


class JobStatus:
    """Job status constants."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    
    ALL_STATUSES = [PENDING, QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED]
    ACTIVE_STATUSES = [PENDING, QUEUED, RUNNING]
    FINAL_STATUSES = [COMPLETED, FAILED, CANCELLED]