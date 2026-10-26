"""Database configuration and session management for VideoAnnotator API."""

from collections.abc import Generator
from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import NullPool, StaticPool

from videoannotator.database_location import database_url

DATABASE_URL = database_url()


def make_engine(url: str) -> Engine:
    """The app's engine for ``url``.

    A file database gets a connection per checkout, never one shared between
    threads: request threads and run workers interleaving on one sqlite3
    connection returned other queries' rows, "another row available" and
    "bad parameter or other API misuse" (UAT #55). The 30 s timeout waits out
    another connection's write instead of failing with "database is locked".
    Only an in-memory database, which exists only on its one connection,
    keeps a shared connection.
    """
    if not url.startswith("sqlite"):
        return create_engine(url, echo=False)
    path = url.removeprefix("sqlite:///")
    if url in ("sqlite://", "sqlite:///") or path == ":memory:":
        return create_engine(
            url, connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
    # SQLite creates the file but not its directory (the per-user default may not exist).
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(
        url,
        connect_args={"check_same_thread": False, "timeout": 30},
        poolclass=NullPool,
    )


engine = make_engine(DATABASE_URL)

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
