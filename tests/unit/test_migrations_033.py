"""Spec 033's additive schema: a database from before it gains the columns and
tables, keeps its rows, and its confirmations count as made before the remote
switch existed."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.pool import StaticPool

OLD_SCHEMA = [
    """CREATE TABLE jobs (id CHAR(36) PRIMARY KEY, video_path VARCHAR(500) NOT NULL,
       cancelled_at DATETIME, storage_path VARCHAR(500), output_dir VARCHAR(500),
       retry_count INTEGER DEFAULT 0)""",
    """CREATE TABLE prompts (sha256 VARCHAR(64) PRIMARY KEY, text TEXT NOT NULL,
       name VARCHAR(200), tags JSON NOT NULL, starred BOOLEAN NOT NULL,
       hidden BOOLEAN NOT NULL, first_used_at DATETIME NOT NULL,
       last_used_at DATETIME NOT NULL, first_user_id VARCHAR(64),
       updated_at DATETIME, updated_by VARCHAR(64))""",
    """CREATE TABLE ask_entries (id CHAR(32) PRIMARY KEY, kind VARCHAR(8) NOT NULL,
       status VARCHAR(12) NOT NULL, prompt_sha256 VARCHAR(64) NOT NULL,
       model VARCHAR(200) NOT NULL, base_url VARCHAR(500) NOT NULL,
       endpoint_class VARCHAR(8) NOT NULL, sampling JSON NOT NULL,
       think BOOLEAN NOT NULL, video_path VARCHAR(1000) NOT NULL,
       video_name VARCHAR(255) NOT NULL, video_sha256 VARCHAR(64), moments JSON,
       answers JSON, job_id VARCHAR(64), user_id VARCHAR(64), name VARCHAR(200),
       starred BOOLEAN NOT NULL, created_at DATETIME NOT NULL, finished_at DATETIME)""",
    """CREATE TABLE endpoint_confirmations (base_url VARCHAR(500) PRIMARY KEY,
       confirmed_by VARCHAR(64), confirmed_at DATETIME NOT NULL)""",
    """INSERT INTO prompts VALUES ('s1', 'Touching?', NULL, '[]', 0, 0,
       '2026-01-01', '2026-01-01', NULL, NULL, NULL)""",
    """INSERT INTO ask_entries (id, kind, status, prompt_sha256, model, base_url,
       endpoint_class, sampling, think, video_path, video_name, starred, created_at)
       VALUES ('e1', 'try', 'done', 's1', 'qwen3.5:9b', 'http://127.0.0.1:11434',
       'local', '{}', 0, '/v/a.mp4', 'a.mp4', 0, '2026-01-01')""",
    """INSERT INTO endpoint_confirmations VALUES ('http://gpu.example.org:11434',
       'u1', '2026-01-01')""",
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


def _upgrade() -> None:
    from videoannotator.database.database import create_tables
    from videoannotator.database.migrations import migrate_to_v1_3_0

    create_tables()
    assert migrate_to_v1_3_0() is True


def test_gains_the_tables_and_columns(old_db):
    _upgrade()
    tables = set(inspect(old_db).get_table_names())
    assert {"model_servers", "server_settings"} <= tables
    assert {"backend", "answer_shape"} <= _columns(old_db, "ask_entries")
    assert "answer_shape" in _columns(old_db, "prompts")
    assert "before_switch" in _columns(old_db, "endpoint_confirmations")


def test_keeps_rows_and_marks_old_confirmations(old_db):
    _upgrade()
    with old_db.connect() as conn:
        entry = conn.execute(text("SELECT model, backend FROM ask_entries")).one()
        assert tuple(entry) == ("qwen3.5:9b", "ollama")
        confirmation = conn.execute(
            text("SELECT base_url, before_switch FROM endpoint_confirmations")
        ).one()
        assert tuple(confirmation) == ("http://gpu.example.org:11434", 1)


def test_new_confirmations_default_to_after_the_switch(old_db):
    _upgrade()
    with old_db.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO endpoint_confirmations (base_url, confirmed_at) "
                "VALUES ('http://new.example.org', '2026-10-22')"
            )
        )
    _upgrade()
    with old_db.connect() as conn:
        rows = dict(
            conn.execute(
                text("SELECT base_url, before_switch FROM endpoint_confirmations")
            ).all()
        )
    assert rows == {"http://gpu.example.org:11434": 1, "http://new.example.org": 0}
