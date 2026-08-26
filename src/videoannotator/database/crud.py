"""CRUD operations for VideoAnnotator database models."""

import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import and_, desc
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from .models import APIKey, Job, JobStatus, SavedDataset, SavedPipelinePreset, User


class UserCRUD:
    """CRUD operations for User model."""

    @staticmethod
    def get_by_id(db: Session, user_id: str) -> User | None:
        """Get user by ID."""
        return db.query(User).filter(User.id == user_id).first()

    @staticmethod
    def get_by_email(db: Session, email: str) -> User | None:
        """Get user by email."""
        return db.query(User).filter(User.email == email).first()

    @staticmethod
    def get_by_username(db: Session, username: str) -> User | None:
        """Get user by username."""
        return db.query(User).filter(User.username == username).first()

    @staticmethod
    def create(
        db: Session, email: str, username: str, full_name: str | None = None
    ) -> User:
        """Create a new user."""
        user = User(email=email, username=username, full_name=full_name)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def count(db: Session) -> int:
        """Return the total number of users."""
        return db.query(User).count()

    @staticmethod
    def update(db: Session, user_id: str, **kwargs) -> User | None:
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
        """Generate a new API key.

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
    def get_by_token(db: Session, token: str) -> APIKey | None:
        """Get API key by raw token string (regardless of active status)."""
        key_hash = APIKeyCRUD.hash_api_key(token)
        return db.query(APIKey).filter(APIKey.key_hash == key_hash).first()

    @staticmethod
    def get_by_hash(db: Session, key_hash: str) -> APIKey | None:
        """Get API key by hash."""
        try:
            return (
                db.query(APIKey)
                .filter(and_(APIKey.key_hash == key_hash, APIKey.is_active))
                .first()
            )
        except SQLAlchemyError:
            # Database not initialized yet (e.g., api_keys table missing).
            # For authentication, treat this as "no matching key" rather than crashing the request.
            return None

    @staticmethod
    def get_by_prefix(db: Session, key_prefix: str) -> list[APIKey]:
        """Get API keys by prefix."""
        return db.query(APIKey).filter(APIKey.key_prefix == key_prefix).all()

    @staticmethod
    def create(
        db: Session, user_id: str, key_name: str, expires_days: int | None = None
    ) -> tuple[APIKey, str]:
        """Create a new API key for a user.

        Returns:
            Tuple of (APIKey object, raw_key)
        """
        raw_key, key_hash, key_prefix = APIKeyCRUD.generate_api_key()

        expires_at = None
        if expires_days:
            expires_at = datetime.now(UTC).replace(tzinfo=None) + timedelta(
                days=expires_days
            )

        api_key = APIKey(
            key_name=key_name,
            key_hash=key_hash,
            key_prefix=key_prefix,
            user_id=user_id,
            expires_at=expires_at,
        )

        db.add(api_key)
        db.commit()
        db.refresh(api_key)

        return api_key, raw_key

    @staticmethod
    def authenticate(db: Session, api_key: str) -> User | None:
        """Authenticate user by API key.

        Returns:
            User object if authentication successful, None otherwise
        """
        key_hash = APIKeyCRUD.hash_api_key(api_key)
        api_key_obj = APIKeyCRUD.get_by_hash(db, key_hash)

        if api_key_obj:
            # Check if key is expired
            now = datetime.now(UTC).replace(tzinfo=None)
            if api_key_obj.expires_at and api_key_obj.expires_at < now:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    f"API key expired: {api_key[:10]}... (expired at {api_key_obj.expires_at})"
                )
                return None

            # Update last used timestamp
            api_key_obj.last_used = datetime.now(UTC).replace(tzinfo=None)
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
    def get_by_id(db: Session, job_id: str) -> Job | None:
        """Get job by ID."""
        return db.query(Job).filter(Job.id == job_id).first()

    @staticmethod
    def get_by_user(
        db: Session, user_id: str, limit: int = 100, offset: int = 0
    ) -> list[Job]:
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
    def get_all(db: Session, limit: int = 100, offset: int = 0) -> list[Job]:
        """Get all jobs (admin only)."""
        return (
            db.query(Job)
            .order_by(desc(Job.created_at))
            .limit(limit)
            .offset(offset)
            .all()
        )

    @staticmethod
    def get_by_status(db: Session, status: str, limit: int = 100) -> list[Job]:
        """Get jobs by status."""
        return (
            db.query(Job)
            .filter(Job.status == status)
            .order_by(desc(Job.created_at))
            .limit(limit)
            .all()
        )

    @staticmethod
    def get_active_jobs(db: Session, limit: int = 100) -> list[Job]:
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
        user_id: str | None = None,
        video_filename: str | None = None,
        selected_pipelines: list[str] | None = None,
        config: dict[str, Any] | None = None,
        job_metadata: dict[str, Any] | None = None,
    ) -> Job:
        """Create a new job."""
        job = Job(
            user_id=user_id,
            video_path=video_path,
            video_filename=video_filename,
            selected_pipelines=selected_pipelines or [],
            config=config or {},
            job_metadata=job_metadata or {},
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
        error_message: str | None = None,
        progress_percentage: int | None = None,
    ) -> Job | None:
        """Update job status."""
        job = JobCRUD.get_by_id(db, job_id)
        if job:
            job.status = status

            if status == JobStatus.RUNNING and not job.started_at:
                job.started_at = datetime.now(UTC).replace(tzinfo=None)

            if status in JobStatus.FINAL_STATUSES and not job.completed_at:
                job.completed_at = datetime.now(UTC).replace(tzinfo=None)

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
        job_metadata: dict[str, Any] | None = None,
    ) -> Job | None:
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
        """Clean up jobs older than specified days.

        Returns:
            Number of jobs deleted
        """
        cutoff_date = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=days_old)

        old_jobs = (
            db.query(Job)
            .filter(
                and_(
                    Job.created_at < cutoff_date,
                    Job.status.in_(JobStatus.FINAL_STATUSES),
                )
            )
            .all()
        )

        count = len(old_jobs)
        for job in old_jobs:
            db.delete(job)

        db.commit()
        return count


class SavedDatasetCRUD:
    """CRUD operations for SavedDataset (spec 007). Visibility is shared-read
    (any authenticated user may list/get); mutation is owner-or-admin,
    enforced by the caller (route layer), not here."""

    @staticmethod
    def get_by_id(db: Session, dataset_id: str) -> SavedDataset | None:
        """Get a saved dataset by ID."""
        return db.query(SavedDataset).filter(SavedDataset.id == dataset_id).first()

    @staticmethod
    def list_all(db: Session, limit: int = 100, offset: int = 0) -> list[SavedDataset]:
        """List all saved datasets, newest first (shared-read visibility)."""
        return (
            db.query(SavedDataset)
            .order_by(desc(SavedDataset.created_at))
            .limit(limit)
            .offset(offset)
            .all()
        )

    @staticmethod
    def create(
        db: Session,
        owner_user_id: str,
        name: str,
        video_manifest: list[dict[str, Any]],
        description: str | None = None,
    ) -> SavedDataset | None:
        """Create a saved dataset. Returns None on a name collision for this
        owner (FR-007) rather than raising, so the route can turn that into
        a clean 409."""
        dataset = SavedDataset(
            owner_user_id=owner_user_id,
            name=name,
            description=description,
            video_manifest=video_manifest,
        )
        db.add(dataset)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            return None
        db.refresh(dataset)
        return dataset

    @staticmethod
    def update(
        db: Session,
        dataset_id: str,
        name: str | None = None,
        description: str | None = None,
        video_manifest: list[dict[str, Any]] | None = None,
    ) -> SavedDataset | None:
        """Update a saved dataset's fields (only those provided). Returns
        None if not found, or if a rename collides with an existing name
        for the same owner (FR-007)."""
        dataset = SavedDatasetCRUD.get_by_id(db, dataset_id)
        if not dataset:
            return None
        if name is not None:
            dataset.name = name
        if description is not None:
            dataset.description = description
        if video_manifest is not None:
            dataset.video_manifest = video_manifest
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            return None
        db.refresh(dataset)
        return dataset

    @staticmethod
    def delete(db: Session, dataset_id: str) -> bool:
        """Delete a saved dataset by ID. Does not touch any job that
        referenced it (FR-008) since jobs don't carry a foreign key to it."""
        dataset = SavedDatasetCRUD.get_by_id(db, dataset_id)
        if dataset:
            db.delete(dataset)
            db.commit()
            return True
        return False

    @staticmethod
    def touch_last_used(db: Session, dataset_id: str) -> None:
        """Record that a dataset was just applied to a job submission."""
        dataset = SavedDatasetCRUD.get_by_id(db, dataset_id)
        if dataset:
            dataset.last_used_at = datetime.now(UTC).replace(tzinfo=None)
            db.commit()


class SavedPipelinePresetCRUD:
    """CRUD operations for SavedPipelinePreset (spec 007). Same shared-read /
    owner-or-admin-mutation visibility model as SavedDatasetCRUD."""

    @staticmethod
    def get_by_id(db: Session, preset_id: str) -> SavedPipelinePreset | None:
        """Get a saved preset by ID."""
        return (
            db.query(SavedPipelinePreset)
            .filter(SavedPipelinePreset.id == preset_id)
            .first()
        )

    @staticmethod
    def list_all(
        db: Session, limit: int = 100, offset: int = 0
    ) -> list[SavedPipelinePreset]:
        """List all saved presets, newest first (shared-read visibility)."""
        return (
            db.query(SavedPipelinePreset)
            .order_by(desc(SavedPipelinePreset.created_at))
            .limit(limit)
            .offset(offset)
            .all()
        )

    @staticmethod
    def create(
        db: Session,
        owner_user_id: str,
        name: str,
        selected_pipelines: list[str],
        config: dict[str, Any],
        description: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> SavedPipelinePreset | None:
        """Create a saved preset. Returns None on a name collision for this
        owner (FR-007) rather than raising, so the route can turn that into
        a clean 409."""
        preset = SavedPipelinePreset(
            owner_user_id=owner_user_id,
            name=name,
            description=description,
            selected_pipelines=selected_pipelines,
            config=config,
            tags=tags or {},
        )
        db.add(preset)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            return None
        db.refresh(preset)
        return preset

    @staticmethod
    def update(
        db: Session,
        preset_id: str,
        name: str | None = None,
        description: str | None = None,
        selected_pipelines: list[str] | None = None,
        config: dict[str, Any] | None = None,
        tags: dict[str, Any] | None = None,
    ) -> SavedPipelinePreset | None:
        """Update a saved preset's fields (only those provided). Returns
        None if not found, or if a rename collides with an existing name
        for the same owner (FR-007)."""
        preset = SavedPipelinePresetCRUD.get_by_id(db, preset_id)
        if not preset:
            return None
        if name is not None:
            preset.name = name
        if description is not None:
            preset.description = description
        if selected_pipelines is not None:
            preset.selected_pipelines = selected_pipelines
        if config is not None:
            preset.config = config
        if tags is not None:
            preset.tags = tags
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            return None
        db.refresh(preset)
        return preset

    @staticmethod
    def delete(db: Session, preset_id: str) -> bool:
        """Delete a saved preset by ID. Does not touch any job that
        referenced it (FR-008) since jobs don't carry a foreign key to it."""
        preset = SavedPipelinePresetCRUD.get_by_id(db, preset_id)
        if preset:
            db.delete(preset)
            db.commit()
            return True
        return False

    @staticmethod
    def touch_last_used(db: Session, preset_id: str) -> None:
        """Record that a preset was just applied to a job submission."""
        preset = SavedPipelinePresetCRUD.get_by_id(db, preset_id)
        if preset:
            preset.last_used_at = datetime.now(UTC).replace(tzinfo=None)
            db.commit()
