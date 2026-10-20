"""Spec 029's additive schema: a database from before it gains the columns and
the preferences table at start-up, and its rows are untouched."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.pool import StaticPool

OLD_SCHEMA = [
    """CREATE TABLE jobs (id CHAR(36) PRIMARY KEY, video_path VARCHAR(500) NOT NULL,
       cancelled_at DATETIME, storage_path VARCHAR(500), output_dir VARCHAR(500),
       retry_count INTEGER DEFAULT 0)""",
    """CREATE TABLE saved_datasets (id CHAR(36) PRIMARY KEY, name VARCHAR(200) NOT NULL,
       description TEXT, owner_user_id CHAR(36) NOT NULL, video_manifest JSON NOT NULL,
       server_folder VARCHAR(1000), server_folder_recursive BOOLEAN NOT NULL DEFAULT 0,
       server_selection BOOLEAN NOT NULL DEFAULT 0, created_at DATETIME NOT NULL,
       updated_at DATETIME, last_used_at DATETIME)""",
    """CREATE TABLE saved_pipeline_presets (id CHAR(36) PRIMARY KEY,
       name VARCHAR(200) NOT NULL, description TEXT, owner_user_id CHAR(36) NOT NULL,
       selected_pipelines JSON NOT NULL, config JSON NOT NULL, tags JSON,
       created_at DATETIME NOT NULL, updated_at DATETIME, last_used_at DATETIME)""",
    """INSERT INTO saved_datasets (id, name, owner_user_id, video_manifest, created_at)
       VALUES ('d1', 'Peekaboo', 'u1', '[]', '2026-01-01')""",
    """INSERT INTO saved_pipeline_presets (id, name, owner_user_id, selected_pipelines,
       config, created_at) VALUES ('p1', 'Faces', 'u1', '["face_analysis"]', '{}',
       '2026-01-01')""",
]


@pytest.fixture
def old_db(monkeypatch, tmp_path):
    import videoannotator.database.database as db_module
    import videoannotator.database.migrations as migrations

    engine = create_engine(
        f"sqlite:///{tmp_path / 'old.db'}",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.begin() as conn:
        for statement in OLD_SCHEMA:
            conn.execute(text(statement))
    monkeypatch.setattr(db_module, "engine", engine)
    monkeypatch.setattr(migrations, "engine", engine)
    return engine


def _columns(engine, table: str) -> set[str]:
    return {c["name"] for c in inspect(engine).get_columns(table)}


def test_an_old_database_gains_the_columns_and_table(old_db):
    from videoannotator.database.database import create_tables
    from videoannotator.database.migrations import migrate_to_v1_3_0

    create_tables()
    assert migrate_to_v1_3_0() is True

    assert "current_settings" in _columns(old_db, "saved_datasets")
    assert {"export_choices", "source"} <= _columns(old_db, "saved_pipeline_presets")
    assert {"user_key", "name", "value", "updated_at"} <= _columns(
        old_db, "user_preferences"
    )
    with old_db.connect() as conn:
        dataset = conn.execute(
            text("SELECT name, current_settings FROM saved_datasets")
        ).one()
        preset = conn.execute(
            text("SELECT name, export_choices, source FROM saved_pipeline_presets")
        ).one()
    assert tuple(dataset) == ("Peekaboo", None)
    assert tuple(preset) == ("Faces", None, None)


def test_running_it_twice_changes_nothing(old_db):
    from videoannotator.database.database import create_tables
    from videoannotator.database.migrations import migrate_to_v1_3_0

    create_tables()
    assert migrate_to_v1_3_0() is True
    assert migrate_to_v1_3_0() is True
