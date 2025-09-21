"""
CRUD operations for VideoAnnotator database models.
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc

from .models import User, APIKey, Job, JobStatus


class UserCRUD:
    """CRUD operations for User model."""
    
    @staticmethod
    def get_by_id(db: Session, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return db.query(User).filter(User.id == user_id).first()
    
    @staticmethod
    def get_by_email(db: Session, email: str) -> Optional[User]:
        """Get user by email."""
        return db.query(User).filter(User.email == email).first()
    
    @staticmethod
    def get_by_username(db: Session, username: str) -> Optional[User]:
        """Get user by username."""
        return db.query(User).filter(User.username == username).first()
    
    @staticmethod
    def create(db: Session, email: str, username: str, full_name: str = None) -> User:
        """Create a new user."""
        user = User(
            email=email,
            username=username,
            full_name=full_name
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    
    @staticmethod
    def update(db: Session, user_id: str, **kwargs) -> Optional[User]:
        """Update user by ID."""
        user = UserCRUD.get_by_id(db, user_id)
        if user:
            for key, value in kwargs.items():
                if hasattr(user, key):
                    setattr(user, key, value)
            db.commit()
            db.refresh(user)
        return user
    
    @staticmethod
    def delete(db: Session, user_id: str) -> bool:
        """Delete user by ID."""
        user = UserCRUD.get_by_id(db, user_id)
        if user:
            db.delete(user)
            db.commit()
            return True
        return False


class APIKeyCRUD:
    """CRUD operations for APIKey model."""
    
    @staticmethod
    def generate_api_key() -> tuple[str, str, str]:
        """
        Generate a new API key.
        
        Returns:
            Tuple of (raw_key, key_hash, key_prefix)
        """
        # Generate a secure random API key
        raw_key = f"va_{secrets.token_urlsafe(32)}"
        
        # Create hash for storage
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        # Create prefix for identification (first 8 chars after va_)
        key_prefix = raw_key[3:11]
        
        return raw_key, key_hash, key_prefix
    
    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """Hash an API key for comparison."""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    @staticmethod
    def get_by_hash(db: Session, key_hash: str) -> Optional[APIKey]:
        """Get API key by hash."""
        return db.query(APIKey).filter(
            and_(APIKey.key_hash == key_hash, APIKey.is_active == True)
        ).first()
    
    @staticmethod
    def get_by_prefix(db: Session, key_prefix: str) -> List[APIKey]:
        """Get API keys by prefix."""
        return db.query(APIKey).filter(APIKey.key_prefix == key_prefix).all()
    
    @staticmethod
    def create(db: Session, user_id: str, key_name: str, expires_days: int = None) -> tuple[APIKey, str]:
        """
        Create a new API key for a user.
        
        Returns:
            Tuple of (APIKey object, raw_key)
        """
        raw_key, key_hash, key_prefix = APIKeyCRUD.generate_api_key()
        
        expires_at = None
        if expires_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_days)
        
        api_key = APIKey(
            key_name=key_name,
            key_hash=key_hash,
            key_prefix=key_prefix,
            user_id=user_id,
            expires_at=expires_at
        )
        
        db.add(api_key)
        db.commit()
        db.refresh(api_key)
        
        return api_key, raw_key
    
    @staticmethod
    def authenticate(db: Session, api_key: str) -> Optional[User]:
        """
        Authenticate user by API key.
        
        Returns:
            User object if authentication successful, None otherwise
        """
        key_hash = APIKeyCRUD.hash_api_key(api_key)
        api_key_obj = APIKeyCRUD.get_by_hash(db, key_hash)
        
        if api_key_obj:
            # Check if key is expired
            if api_key_obj.expires_at and api_key_obj.expires_at < datetime.utcnow():
                return None
            
            # Update last used timestamp
            api_key_obj.last_used = datetime.utcnow()
            db.commit()
            
            return api_key_obj.user
        
        return None
    
    @staticmethod
    def revoke(db: Session, api_key_id: str) -> bool:
        """Revoke an API key."""
        api_key = db.query(APIKey).filter(APIKey.id == api_key_id).first()
        if api_key:
            api_key.is_active = False
            db.commit()
            return True
        return False


class JobCRUD:
    """CRUD operations for Job model."""
    
    @staticmethod
    def get_by_id(db: Session, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        return db.query(Job).filter(Job.id == job_id).first()
    
    @staticmethod
    def get_by_user(db: Session, user_id: str, limit: int = 100, offset: int = 0) -> List[Job]:
        """Get jobs for a user."""
        return (
            db.query(Job)
            .filter(Job.user_id == user_id)
            .order_by(desc(Job.created_at))
            .limit(limit)
            .offset(offset)
            .all()
        )
    
    @staticmethod
    def get_all(db: Session, limit: int = 100, offset: int = 0) -> List[Job]:
        """Get all jobs (admin only)."""
        return (
            db.query(Job)
            .order_by(desc(Job.created_at))
            .limit(limit)
            .offset(offset)
            .all()
        )
    
    @staticmethod
    def get_by_status(db: Session, status: str, limit: int = 100) -> List[Job]:
        """Get jobs by status."""
        return (
            db.query(Job)
            .filter(Job.status == status)
            .order_by(desc(Job.created_at))
            .limit(limit)
            .all()
        )
    
    @staticmethod
    def get_active_jobs(db: Session, limit: int = 100) -> List[Job]:
        """Get active (non-completed) jobs."""
        return (
            db.query(Job)
            .filter(Job.status.in_(JobStatus.ACTIVE_STATUSES))
            .order_by(desc(Job.created_at))
            .limit(limit)
            .all()
        )
    
    @staticmethod
    def create(
        db: Session,
        video_path: str,
        user_id: str = None,
        video_filename: str = None,
        selected_pipelines: List[str] = None,
        config: Dict[str, Any] = None,
        job_metadata: Dict[str, Any] = None
    ) -> Job:
        """Create a new job."""
        job = Job(
            user_id=user_id,
            video_path=video_path,
            video_filename=video_filename,
            selected_pipelines=selected_pipelines or [],
            config=config or {},
            job_metadata=job_metadata or {}
        )
        
        db.add(job)
        db.commit()
        db.refresh(job)
        return job
    
    @staticmethod
    def update_status(
        db: Session,
        job_id: str,
        status: str,
        error_message: str = None,
        progress_percentage: int = None
    ) -> Optional[Job]:
        """Update job status."""
        job = JobCRUD.get_by_id(db, job_id)
        if job:
            job.status = status
            
            if status == JobStatus.RUNNING and not job.started_at:
                job.started_at = datetime.utcnow()
            
            if status in JobStatus.FINAL_STATUSES and not job.completed_at:
                job.completed_at = datetime.utcnow()
            
            if error_message is not None:
                job.error_message = error_message
            
            if progress_percentage is not None:
                job.progress_percentage = progress_percentage
            
            db.commit()
            db.refresh(job)
        
        return job
    
    @staticmethod
    def update_results(
        db: Session,
        job_id: str,
        result_path: str,
        job_metadata: Dict[str, Any] = None
    ) -> Optional[Job]:
        """Update job results."""
        job = JobCRUD.get_by_id(db, job_id)
        if job:
            job.result_path = result_path
            if job_metadata:
                job.job_metadata = {**(job.job_metadata or {}), **job_metadata}
            
            db.commit()
            db.refresh(job)
        
        return job
    
    @staticmethod
    def delete(db: Session, job_id: str) -> bool:
        """Delete job by ID."""
        job = JobCRUD.get_by_id(db, job_id)
        if job:
            db.delete(job)
            db.commit()
            return True
        return False
    
    @staticmethod
    def cleanup_old_jobs(db: Session, days_old: int = 30) -> int:
        """
        Clean up jobs older than specified days.
        
        Returns:
            Number of jobs deleted
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        old_jobs = (
            db.query(Job)
            .filter(
                and_(
                    Job.created_at < cutoff_date,
                    Job.status.in_(JobStatus.FINAL_STATUSES)
                )
            )
            .all()
        )
        
        count = len(old_jobs)
        for job in old_jobs:
            db.delete(job)
        
        db.commit()
        return count