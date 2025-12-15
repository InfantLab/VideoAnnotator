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

        from sqlalchemy import text

        with engine.connect() as conn:

            def ensure_columns(table: str, required: dict[str, str]) -> list[str]:
                try:
                    result = conn.execute(text(f"PRAGMA table_info('{table}')"))
                    existing_cols = {row[1] for row in result}
                    added: list[str] = []
                    for column, ddl in required.items():
                        if column in existing_cols:
                            continue
                        logger.warning(
                            f"[MIGRATION] Adding missing column {table}.{column}"
                        )
                        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {ddl}"))
                        added.append(column)
                    return added
                except Exception as mig_e:
                    logger.error(
                        f"[MIGRATION] Failed to align {table} table schema: {mig_e}"
                    )
                    return []

            if "jobs" in created_tables:
                added = ensure_columns(
                    "jobs",
                    {
                        "output_dir": "VARCHAR",
                        "retry_count": "INTEGER DEFAULT 0",
                    },
                )
                if added:
                    logger.info(
                        "[MIGRATION] Jobs table schema updated for storage backend compatibility"
                    )

            if "users" in created_tables:
                added = ensure_columns(
                    "users",
                    {
                        "full_name": "VARCHAR(200)",
                    },
                )
                if added:
                    logger.info(
                        "[MIGRATION] Users table schema updated for authentication compatibility"
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
        _api_key_obj, raw_key = APIKeyCRUD.create(
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


def migrate_to_v1_3_0() -> bool:
    """Migrate database schema from v1.2.x to v1.3.0.

    Adds:
    - jobs.cancelled_at (DateTime): Timestamp when job was cancelled
    - jobs.storage_path (String): Path to persistent job storage directory

    Returns:
        True if successful, False otherwise
    """
    from sqlalchemy import inspect, text

    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        if "jobs" not in tables:
            logger.warning(
                "[MIGRATION] Jobs table does not exist, skipping v1.3.0 migration"
            )
            return False

        with engine.connect() as conn:
            # Get existing columns
            result = conn.execute(text("PRAGMA table_info('jobs')"))
            existing_cols = {row[1] for row in result}  # row[1] is column name

            migrations_applied = []

            # Add cancelled_at column if missing
            if "cancelled_at" not in existing_cols:
                logger.info("[MIGRATION] Adding jobs.cancelled_at column")
                conn.execute(text("ALTER TABLE jobs ADD COLUMN cancelled_at DATETIME"))
                migrations_applied.append("cancelled_at")
                logger.info("[MIGRATION] ✓ Added jobs.cancelled_at column")

            # Add storage_path column if missing
            if "storage_path" not in existing_cols:
                logger.info("[MIGRATION] Adding jobs.storage_path column")
                conn.execute(
                    text("ALTER TABLE jobs ADD COLUMN storage_path VARCHAR(500)")
                )
                migrations_applied.append("storage_path")
                logger.info("[MIGRATION] ✓ Added jobs.storage_path column")

            # Add output_dir column if missing (Storage Backend compatibility)
            if "output_dir" not in existing_cols:
                logger.info("[MIGRATION] Adding jobs.output_dir column")
                conn.execute(
                    text("ALTER TABLE jobs ADD COLUMN output_dir VARCHAR(500)")
                )
                migrations_applied.append("output_dir")
                logger.info("[MIGRATION] ✓ Added jobs.output_dir column")

            # Add retry_count column if missing (Storage Backend compatibility)
            if "retry_count" not in existing_cols:
                logger.info("[MIGRATION] Adding jobs.retry_count column")
                conn.execute(
                    text("ALTER TABLE jobs ADD COLUMN retry_count INTEGER DEFAULT 0")
                )
                migrations_applied.append("retry_count")
                logger.info("[MIGRATION] ✓ Added jobs.retry_count column")

            conn.commit()

            if migrations_applied:
                logger.info(
                    f"[MIGRATION] v1.3.0 migration complete. Applied: {', '.join(migrations_applied)}"
                )
            else:
                logger.info("[MIGRATION] v1.3.0 schema already up to date")

            return True

    except Exception as e:
        logger.error(f"[MIGRATION] Failed to migrate to v1.3.0: {e}")
        return False


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
