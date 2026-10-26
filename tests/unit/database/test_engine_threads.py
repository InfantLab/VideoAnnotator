"""The app's engine under concurrent threads (UAT #55).

One sqlite3 connection shared between request threads and run workers
returned other queries' rows and "another row available" under ordinary use.
"""

import threading
import uuid

from sqlalchemy import text
from sqlalchemy.pool import NullPool, StaticPool

from videoannotator.database.database import make_engine


def test_file_database_does_not_share_one_connection(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'sub' / 'va.db'}")
    assert isinstance(engine.pool, NullPool)
    assert (tmp_path / "sub").is_dir()


def test_memory_database_keeps_its_one_connection():
    assert isinstance(make_engine("sqlite://").pool, StaticPool)
    assert isinstance(make_engine("sqlite:///:memory:").pool, StaticPool)


def test_threads_reading_and_writing_at_once_get_their_own_rows(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'va.db'}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE t (id TEXT PRIMARY KEY, n INTEGER)"))
        conn.execute(
            text("INSERT INTO t VALUES (:id, :n)"),
            [{"id": f"r{n}", "n": n} for n in range(50)],
        )

    errors: list[BaseException] = []

    def reader(n: int) -> None:
        try:
            for _ in range(10):
                with engine.connect() as conn:
                    rows = conn.execute(
                        text("SELECT n FROM t WHERE id = :id"), {"id": f"r{n}"}
                    ).all()
                    assert rows == [(n,)]
                    assert len(conn.execute(text("SELECT * FROM t")).all()) >= 50
        except BaseException as e:
            errors.append(e)

    def writer() -> None:
        try:
            for _ in range(10):
                with engine.begin() as conn:
                    conn.execute(
                        text("INSERT INTO t VALUES (:id, -1)"),
                        {"id": str(uuid.uuid4())},
                    )
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=reader, args=(n,)) for n in range(4)]
    threads.append(threading.Thread(target=writer))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
