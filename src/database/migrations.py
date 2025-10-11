"""Database migration utilities for VideoAnnotator."""

import logging
from pathlib import Path

from .crud import APIKeyCRUD, UserCRUD
from .database import create_tables, drop_tables, engine
from .models import Job, User

logger = logging.getLogger(__name__)


def init_database(force: bool = False) -> bool:
    """Initialize the database with tables and default data.

    Args:
        force: If True, drop existing tables first

    Returns:
        True if successful, False otherwise
    """
    try:
        if force:
            logger.info("Dropping existing tables...")
            drop_tables()

        logger.info("Creating database tables...")
        create_tables()

        # Verify tables were created
        from sqlalchemy import inspect

        inspector = inspect(engine)
        tables = inspector.get_table_names()

        expected_tables = {"users", "api_keys", "jobs"}
        created_tables = set(tables)

        if not expected_tables.issubset(created_tables):
            missing = expected_tables - created_tables
            logger.error(f"Missing tables: {missing}")
            return False

        logger.info(f"Successfully created tables: {created_tables}")

        # Align legacy jobs table schema with storage backend expectations
        # The storage backend defines additional columns (output_dir, retry_count)
        # that might not exist if database was initialized via legacy models.
        if "jobs" in created_tables:
            from sqlalchemy import text

            with engine.connect() as conn:
                try:
                    # SQLite pragma to introspect columns
                    result = conn.execute(text("PRAGMA table_info('jobs')"))
                    existing_cols = {row[1] for row in result}  # row[1] is column name
                    alter_performed = False
                    if "output_dir" not in existing_cols:
                        logger.warning(
                            "[MIGRATION] Adding missing column jobs.output_dir"
                        )
                        conn.execute(
                            text("ALTER TABLE jobs ADD COLUMN output_dir VARCHAR")
                        )
                        alter_performed = True
                    if "retry_count" not in existing_cols:
                        logger.warning(
                            "[MIGRATION] Adding missing column jobs.retry_count"
                        )
                        conn.execute(
                            text(
                                "ALTER TABLE jobs ADD COLUMN retry_count INTEGER DEFAULT 0"
                            )
                        )
                        alter_performed = True
                    if alter_performed:
                        logger.info(
                            "[MIGRATION] Jobs table schema updated for storage backend compatibility"
                        )
                except Exception as mig_e:
                    logger.error(
                        f"[MIGRATION] Failed to align jobs table schema: {mig_e}"
                    )
        return True

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False


def create_admin_user(
    username: str = "admin",
    email: str = "admin@videoannotator.com",
    full_name: str = "Administrator",
) -> tuple[User, str] | None:
    """Create an admin user with API key.

    Returns:
        Tuple of (User, api_key) if successful, None otherwise
    """
    from .database import SessionLocal

    db = SessionLocal()
    try:
        # Check if admin user already exists
        existing_user = UserCRUD.get_by_username(db, username)
        if existing_user:
            logger.info(f"Admin user '{username}' already exists")
            return existing_user, None  # type: ignore[return-value]

        # Create admin user
        user = UserCRUD.create(
            db=db, email=email, username=username, full_name=full_name
        )

        # Make user admin
        user.is_admin = True
        db.commit()

        # Create API key for admin
        api_key_obj, raw_key = APIKeyCRUD.create(
            db=db,
            user_id=str(user.id),
            key_name="admin_default",
            expires_days=None,  # Never expires
        )

        logger.info(f"Created admin user '{username}' with API key")
        logger.info(f"API Key: {raw_key}")
        logger.warning("Save this API key securely - it won't be shown again!")

        return user, raw_key

    except Exception as e:
        logger.error(f"Failed to create admin user: {e}")
        db.rollback()
        return None
    finally:
        db.close()


def migrate_from_memory_jobs(memory_jobs: dict) -> int:
    """Migrate jobs from in-memory storage to database.

    Args:
        memory_jobs: Dictionary of jobs from the old in-memory system

    Returns:
        Number of jobs migrated
    """
    from .database import SessionLocal

    db = SessionLocal()
    migrated_count = 0

    try:
        for job_id, job_obj in memory_jobs.items():
            try:
                # Convert memory job to database job
                db_job = Job(
                    id=job_id,
                    video_path=getattr(job_obj, "video_path", ""),
                    video_filename=getattr(job_obj, "video_path", "").split("/")[-1],
                    selected_pipelines=getattr(job_obj, "selected_pipelines", []),
                    config=getattr(job_obj, "config", {}),
                    status=getattr(job_obj, "status", "pending"),
                    created_at=getattr(job_obj, "created_at", None),
                    completed_at=getattr(job_obj, "completed_at", None),
                    error_message=getattr(job_obj, "error_message", None),
                )

                db.add(db_job)
                migrated_count += 1

            except Exception as job_error:
                logger.warning(f"Failed to migrate job {job_id}: {job_error}")
                continue

        db.commit()
        logger.info(f"Successfully migrated {migrated_count} jobs to database")

    except Exception as e:
        logger.error(f"Failed to migrate jobs: {e}")
        db.rollback()
    finally:
        db.close()

    return migrated_count


def backup_database(backup_path: Path | None = None) -> bool:
    """Create a backup of the database.

    Args:
        backup_path: Path for backup file. If None, auto-generate

    Returns:
        True if successful, False otherwise
    """
    try:
        if backup_path is None:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = Path(f"videoannotator_backup_{timestamp}.db")

        # For SQLite, simply copy the database file
        database_url = str(engine.url)
        if database_url.startswith("sqlite"):
            import shutil

            db_path = database_url.replace("sqlite:///", "").replace("sqlite://", "")
            shutil.copy2(db_path, backup_path)
            logger.info(f"Database backed up to: {backup_path}")
            return True
        else:
            # For other databases, would need more complex backup logic
            logger.warning("Database backup not implemented for non-SQLite databases")
            return False

    except Exception as e:
        logger.error(f"Failed to backup database: {e}")
        return False


def get_database_info() -> dict:
    """Get information about the current database.

    Returns:
        Dictionary with database information
    """
    from sqlalchemy import inspect, text

    from .database import SessionLocal

    db = SessionLocal()
    try:
        inspector = inspect(engine)

        info = {
            "database_url": str(engine.url),
            "tables": inspector.get_table_names(),
            "table_info": {},
        }

        # Get row counts for each table
        for table_name in info["tables"]:
            try:
                result = db.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                count = result.scalar()
                info["table_info"][table_name] = {"row_count": count}
            except Exception as e:
                info["table_info"][table_name] = {"error": str(e)}

        return info

    except Exception as e:
        return {"error": str(e)}
    finally:
        db.close()


if __name__ == "__main__":
    """Command-line interface for database migrations."""
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="VideoAnnotator Database Migration")
    parser.add_argument(
        "command",
        choices=["init", "admin", "info", "backup"],
        help="Migration command to run",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force operation (recreate tables)"
    )
    parser.add_argument("--username", default="admin", help="Admin username")
    parser.add_argument(
        "--email", default="admin@videoannotator.com", help="Admin email"
    )

    args = parser.parse_args()

    if args.command == "init":
        success = init_database(force=args.force)
        sys.exit(0 if success else 1)

    elif args.command == "admin":
        result = create_admin_user(username=args.username, email=args.email)
        sys.exit(0 if result else 1)

    elif args.command == "info":
        info = get_database_info()
        print("Database Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        sys.exit(0)

    elif args.command == "backup":
        success = backup_database()
        sys.exit(0 if success else 1)
