"""Database configuration and session management for VideoAnnotator API."""

from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool

from videoannotator.database_location import database_url

DATABASE_URL = database_url()

# Create engine with appropriate settings
if DATABASE_URL.startswith("sqlite"):
    # SQLite creates the file but not its directory (the per-user default may not exist).
    Path(DATABASE_URL.removeprefix("sqlite:///")).parent.mkdir(
        parents=True, exist_ok=True
    )
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,  # Set to True for SQL debugging
    )
else:
    # PostgreSQL/MySQL settings
    engine = create_engine(
        DATABASE_URL,
        echo=False,  # Set to True for SQL debugging
    )

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class for models
Base = declarative_base()


def get_db() -> Generator:
    """Dependency to get database session.

    Yields:
        Database session that will be closed after use
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)


def drop_tables():
    """Drop all database tables."""
    Base.metadata.drop_all(bind=engine)
