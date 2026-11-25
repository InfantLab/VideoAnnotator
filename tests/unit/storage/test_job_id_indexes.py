"""A job's results and annotations are found by index, not a table scan (UAT #74)."""

import sqlite3

from videoannotator.storage.sqlite_backend import SQLiteStorageBackend

INDEXES = {"ix_pipeline_results_job_id", "ix_annotations_job_id"}


def _indexes(path):
    with sqlite3.connect(path) as conn:
        return {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}


def test_a_new_database_has_them(tmp_path):
    SQLiteStorageBackend(tmp_path / "new.db").close()
    assert _indexes(tmp_path / "new.db") >= INDEXES


def test_an_older_database_gains_them(tmp_path):
    path = tmp_path / "old.db"
    SQLiteStorageBackend(path).close()
    with sqlite3.connect(path) as conn:
        for name in INDEXES:
            conn.execute(f"DROP INDEX {name}")
    assert not INDEXES & _indexes(path)

    SQLiteStorageBackend(path).close()
    assert _indexes(path) >= INDEXES


def test_loading_a_job_uses_the_index(tmp_path):
    path = tmp_path / "va.db"
    SQLiteStorageBackend(path).close()
    with sqlite3.connect(path) as conn:
        plan = conn.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM pipeline_results WHERE job_id = 'x'"
        ).fetchall()
    assert "ix_pipeline_results_job_id" in str(plan)
