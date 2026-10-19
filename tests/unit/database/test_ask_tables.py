"""Spec 027's tables are created on a fresh database by create_all."""

from sqlalchemy import create_engine, inspect

from videoannotator.database import models  # noqa: F401  (registers the tables)
from videoannotator.database.database import Base


def test_ask_tables_created_on_a_fresh_database():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(bind=engine)
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    assert {"ask_entries", "endpoint_confirmations"} <= tables
    indexed = {
        col for ix in inspector.get_indexes("ask_entries") for col in ix["column_names"]
    }
    assert {"created_at", "video_path", "user_id", "job_id"} <= indexed


def test_video_notes_table_created_on_a_fresh_database():
    """Spec 028: notes on a video, looked up by its path."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(bind=engine)
    inspector = inspect(engine)
    assert "video_notes" in inspector.get_table_names()
    indexed = {
        col for ix in inspector.get_indexes("video_notes") for col in ix["column_names"]
    }
    assert "video_path" in indexed
