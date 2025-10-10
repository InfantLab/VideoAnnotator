"""
SQLite-based storage backend for VideoAnnotator.

This backend provides persistent storage using SQLite, perfect for individual researchers
who need zero-configuration local installations. The database file is created automatically
in the current directory and contains all job data, annotations, and processing results.

Key features:
- Zero configuration: Database auto-created on first use
- Single file: Easy backup, sharing, and archiving
- Cross-platform: Works identically on Windows, Mac, Linux
- Research-friendly: Database lives with video files in project directory
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy.exc import SQLAlchemyError

from .base import StorageBackend
from .models import (
    CURRENT_SCHEMA_VERSION,
    Annotation,
    Job,
    PipelineResult,
    SchemaVersion,
    create_database_engine,
    create_session_factory,
    initialize_database,
)
from .models import BatchReport as BatchReportModel

if TYPE_CHECKING:
    from batch.types import BatchJob, BatchReport


class SQLiteStorageBackend(StorageBackend):
    """
    SQLite-based storage backend for local research installations.

    This backend creates and manages a local SQLite database file that contains
    all job data, processing results, and annotations. Perfect for individual
    researchers who want zero-configuration operation.
    """

    def __init__(self, database_path: Path | None = None, echo: bool = False):
        """
        Initialize SQLite storage backend.

        Args:
            database_path: Path to SQLite database file.
                          Defaults to ./videoannotator.db in current directory.
            echo: Whether to log SQL queries (useful for debugging)
        """
        if database_path is None:
            database_path = Path.cwd() / "videoannotator.db"

        self.database_path = Path(database_path)
        self.database_url = f"sqlite:///{self.database_path}"

        # Create database engine
        self.engine = create_database_engine(self.database_url, echo=echo)

        # Set up logging first
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter("[%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # Log database creation for first-time users
        database_exists = self.database_path.exists()
        if not database_exists:
            self.logger.info(
                f"[DATABASE] Creating new VideoAnnotator database: {self.database_path}"
            )
        else:
            self.logger.info(
                f"[DATABASE] Using existing database: {self.database_path}"
            )

        # Create session factory
        self.SessionLocal = create_session_factory(self.engine)

        # Initialize database (create tables if they don't exist)
        self._initialize_database()

    def _initialize_database(self):
        """Initialize database schema and handle migrations."""
        try:
            # Create all tables
            initialize_database(self.engine)

            # Check/update schema version
            with self.SessionLocal() as session:
                schema_version = session.query(SchemaVersion).first()

                if not schema_version:
                    # First time setup
                    version_record = SchemaVersion(
                        version=CURRENT_SCHEMA_VERSION,
                        description="Initial database schema for VideoAnnotator v1.2.0",
                    )
                    session.add(version_record)
                    session.commit()
                    self.logger.info(
                        f"[DATABASE] Initialized schema version {CURRENT_SCHEMA_VERSION}"
                    )

                elif schema_version.version < CURRENT_SCHEMA_VERSION:
                    # Future: Handle schema migrations
                    self.logger.warning(
                        f"[DATABASE] Schema version {schema_version.version} < {CURRENT_SCHEMA_VERSION} - migrations needed"
                    )

        except SQLAlchemyError as e:
            self.logger.error(f"[ERROR] Failed to initialize database: {e}")
            raise

    def _batch_job_to_db_job(self, batch_job: "BatchJob") -> Job:
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
            error_message=batch_job.error_message,
        )

    def _db_job_to_batch_job(self, db_job: Job) -> "BatchJob":
        """Convert database Job model to BatchJob."""
        from batch.types import BatchJob, JobStatus
        from batch.types import PipelineResult as BatchPipelineResult

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
            selected_pipelines=db_job.selected_pipelines,
        )

        # Load pipeline results
        for result in db_job.pipeline_results:
            pipeline_result = BatchPipelineResult(
                pipeline_name=result.pipeline_name,
                status=JobStatus(result.status),
                start_time=result.start_time,
                end_time=result.end_time,
                processing_time=result.processing_time / 1000.0
                if result.processing_time
                else None,  # Convert ms to seconds
                annotation_count=result.annotation_count,
                output_file=Path(result.output_file) if result.output_file else None,
                error_message=result.error_message,
            )
            batch_job.pipeline_results[result.pipeline_name] = pipeline_result

        return batch_job

    def save_job_metadata(self, job: "BatchJob") -> None:
        """Save job metadata to database."""
        try:
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
                    existing.selected_pipelines = job.selected_pipelines
                    existing.config = job.config

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
                            processing_time=int(result.processing_time * 1000)
                            if result.processing_time
                            else None,  # Convert to ms
                            annotation_count=result.annotation_count,
                            output_file=str(result.output_file)
                            if result.output_file
                            else None,
                            error_message=result.error_message,
                        )
                        session.add(db_result)

                    self.logger.debug(f"[DATABASE] Updated job {job.job_id}")
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
                            processing_time=int(result.processing_time * 1000)
                            if result.processing_time
                            else None,
                            annotation_count=result.annotation_count,
                            output_file=str(result.output_file)
                            if result.output_file
                            else None,
                            error_message=result.error_message,
                        )
                        session.add(db_result)

                    self.logger.debug(f"[DATABASE] Created job {job.job_id}")

                session.commit()

        except SQLAlchemyError as e:
            self.logger.error(
                f"[ERROR] Failed to save job metadata for {job.job_id}: {e}"
            )
            raise

    def load_job_metadata(self, job_id: str) -> "BatchJob":
        """Load job metadata from database."""
        try:
            with self.SessionLocal() as session:
                db_job = session.query(Job).filter_by(id=job_id).first()
                if not db_job:
                    raise FileNotFoundError(f"Job {job_id} not found in database")

                return self._db_job_to_batch_job(db_job)

        except SQLAlchemyError as e:
            self.logger.error(
                f"[ERROR] Failed to load job metadata for {job_id}: {e}", exc_info=True
            )
            raise

    def save_annotations(
        self, job_id: str, pipeline: str, annotations: list[dict[str, Any]]
    ) -> str:
        """Save pipeline annotations to database."""
        try:
            with self.SessionLocal() as session:
                # Delete existing annotations for this job+pipeline combination
                session.query(Annotation).filter_by(
                    job_id=job_id, pipeline=pipeline
                ).delete()

                # Save new annotations
                for annotation_data in annotations:
                    db_annotation = Annotation(
                        job_id=job_id,
                        pipeline=pipeline,
                        data=annotation_data,
                        created_at=datetime.utcnow(),
                    )
                    session.add(db_annotation)

                session.commit()

                self.logger.debug(
                    f"[DATABASE] Saved {len(annotations)} annotations for {job_id}/{pipeline}"
                )
                return f"database://annotations/{job_id}/{pipeline}"

        except SQLAlchemyError as e:
            self.logger.error(
                f"[ERROR] Failed to save annotations for {job_id}/{pipeline}: {e}"
            )
            raise

    def load_annotations(self, job_id: str, pipeline: str) -> list[dict[str, Any]]:
        """Load pipeline annotations from database."""
        try:
            with self.SessionLocal() as session:
                db_annotations = (
                    session.query(Annotation)
                    .filter_by(job_id=job_id, pipeline=pipeline)
                    .all()
                )

                if not db_annotations:
                    raise FileNotFoundError(
                        f"Annotations not found for {job_id}/{pipeline}"
                    )

                return [ann.data for ann in db_annotations]

        except SQLAlchemyError as e:
            self.logger.error(
                f"[ERROR] Failed to load annotations for {job_id}/{pipeline}: {e}"
            )
            raise

    def annotation_exists(self, job_id: str, pipeline: str) -> bool:
        """Check if annotations exist for job and pipeline."""
        try:
            with self.SessionLocal() as session:
                count = (
                    session.query(Annotation)
                    .filter_by(job_id=job_id, pipeline=pipeline)
                    .count()
                )
                return count > 0

        except SQLAlchemyError as e:
            self.logger.error(
                f"[ERROR] Failed to check annotations for {job_id}/{pipeline}: {e}"
            )
            return False

    def list_jobs(self, status_filter: str | None = None) -> list[str]:
        """List all job IDs, optionally filtered by status."""
        try:
            with self.SessionLocal() as session:
                query = session.query(Job.id)

                if status_filter:
                    query = query.filter(Job.status == status_filter)

                return [row[0] for row in query.order_by(Job.created_at.desc()).all()]

        except SQLAlchemyError as e:
            self.logger.error(f"[ERROR] Failed to list jobs: {e}")
            return []

    def delete_job(self, job_id: str) -> None:
        """Delete all data for a job."""
        try:
            with self.SessionLocal() as session:
                # Foreign key constraints will cascade delete annotations and pipeline_results
                deleted_count = session.query(Job).filter_by(id=job_id).delete()

                if deleted_count == 0:
                    self.logger.warning(
                        f"[WARNING] Job {job_id} not found for deletion"
                    )
                else:
                    self.logger.debug(f"[DATABASE] Deleted job {job_id}")

                session.commit()

        except SQLAlchemyError as e:
            self.logger.error(f"[ERROR] Failed to delete job {job_id}: {e}")
            raise

    def get_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        try:
            with self.SessionLocal() as session:
                # Job statistics
                total_jobs = session.query(Job).count()
                pending_jobs = session.query(Job).filter_by(status="pending").count()
                running_jobs = session.query(Job).filter_by(status="running").count()
                completed_jobs = (
                    session.query(Job).filter_by(status="completed").count()
                )
                failed_jobs = session.query(Job).filter_by(status="failed").count()
                cancelled_jobs = (
                    session.query(Job).filter_by(status="cancelled").count()
                )

                # Annotation statistics
                total_annotations = session.query(Annotation).count()

                # Pipeline statistics
                pipeline_results_count = session.query(PipelineResult).count()

                # Database file statistics
                database_size_mb = 0
                if self.database_path.exists():
                    database_size_mb = round(
                        self.database_path.stat().st_size / (1024 * 1024), 2
                    )

                return {
                    "backend": "sqlite",
                    "database_path": str(self.database_path),
                    "database_url": self.database_url,
                    "database_size_mb": database_size_mb,
                    "total_jobs": total_jobs,
                    "pending_jobs": pending_jobs,
                    "running_jobs": running_jobs,
                    "completed_jobs": completed_jobs,
                    "failed_jobs": failed_jobs,
                    "cancelled_jobs": cancelled_jobs,
                    "total_annotations": total_annotations,
                    "pipeline_results_count": pipeline_results_count,
                    "schema_version": CURRENT_SCHEMA_VERSION,
                }

        except SQLAlchemyError as e:
            self.logger.error(f"[ERROR] Failed to get database stats: {e}")
            # Return basic stats even if database query fails
            return {
                "backend": "sqlite",
                "database_path": str(self.database_path),
                "database_url": self.database_url,
                "error": str(e),
            }

    def save_report(self, report: "BatchReport") -> None:
        """Save batch report to database."""
        try:
            with self.SessionLocal() as session:
                # Convert BatchReport to database model
                db_report = BatchReportModel(
                    id=report.batch_id,
                    start_time=report.start_time,
                    end_time=report.end_time,
                    total_jobs=report.total_jobs,
                    completed_jobs=report.completed_jobs,
                    failed_jobs=report.failed_jobs,
                    cancelled_jobs=report.cancelled_jobs,
                    total_processing_time=int(report.total_processing_time * 1000)
                    if report.total_processing_time
                    else 0,
                    report_data=report.to_dict(),  # Store full report as JSON
                )

                # Use merge to handle both insert and update
                session.merge(db_report)
                session.commit()

                self.logger.debug(f"[DATABASE] Saved batch report {report.batch_id}")

        except SQLAlchemyError as e:
            self.logger.error(
                f"[ERROR] Failed to save batch report {report.batch_id}: {e}"
            )
            raise

    def load_report(self, batch_id: str) -> "BatchReport":
        """Load batch report from database."""
        try:
            with self.SessionLocal() as session:
                db_report = (
                    session.query(BatchReportModel).filter_by(id=batch_id).first()
                )

                if not db_report:
                    raise FileNotFoundError(f"Batch report {batch_id} not found")

                # Reconstruct BatchReport from stored JSON data
                from batch.types import BatchReport

                return BatchReport.from_dict(db_report.report_data)

        except SQLAlchemyError as e:
            self.logger.error(f"[ERROR] Failed to load batch report {batch_id}: {e}")
            raise

    def list_reports(self) -> list[str]:
        """List all batch report IDs."""
        try:
            with self.SessionLocal() as session:
                return [
                    row[0]
                    for row in session.query(BatchReportModel.id)
                    .order_by(BatchReportModel.start_time.desc())
                    .all()
                ]

        except SQLAlchemyError as e:
            self.logger.error(f"[ERROR] Failed to list batch reports: {e}")
            return []

    def cleanup_old_files(self, max_age_days: int) -> tuple[int, int]:
        """
        Clean up old jobs and reports.

        Args:
            max_age_days: Maximum age in days for jobs/reports to keep

        Returns:
            Tuple of (deleted_jobs, deleted_reports)
        """
        try:
            from datetime import timedelta

            cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)

            with self.SessionLocal() as session:
                # Delete old completed/failed jobs
                old_jobs = (
                    session.query(Job)
                    .filter(
                        Job.completed_at < cutoff_date,
                        Job.status.in_(["completed", "failed", "cancelled"]),
                    )
                    .all()
                )

                deleted_jobs = len(old_jobs)
                for job in old_jobs:
                    session.delete(job)

                # Delete old reports
                old_reports = (
                    session.query(BatchReportModel)
                    .filter(BatchReportModel.end_time < cutoff_date)
                    .all()
                )

                deleted_reports = len(old_reports)
                for report in old_reports:
                    session.delete(report)

                session.commit()

                self.logger.info(
                    f"[DATABASE] Cleaned up {deleted_jobs} old jobs and {deleted_reports} old reports"
                )
                return (deleted_jobs, deleted_reports)

        except SQLAlchemyError as e:
            self.logger.error(f"[ERROR] Failed to cleanup old files: {e}")
            return (0, 0)
