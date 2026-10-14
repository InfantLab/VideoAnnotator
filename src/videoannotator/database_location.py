"""Where the database lives: one setting for both database layers.

The storage backend (`api/database.py`) and the SQLAlchemy layer
(`database/database.py`) used to default to `./videoannotator.db` under whatever
directory the server started in, and only the first honoured
`VIDEOANNOTATOR_DB_PATH`. Standard library only, so `config_env` can use it
without creating an engine.
"""

from __future__ import annotations

import os
from pathlib import Path

from videoannotator.models_dir import user_data_dir

DB_PATH_ENV = "VIDEOANNOTATOR_DB_PATH"


def database_path() -> Path:
    """The SQLite database file, as an absolute path.

    `VIDEOANNOTATOR_DB_PATH`, defaulting to `videoannotator.db` in the per-user
    data directory (next to the models directory's default).
    """
    configured = os.environ.get(DB_PATH_ENV)
    path = (
        Path(configured).expanduser()
        if configured
        else user_data_dir() / "videoannotator.db"
    )
    return path.resolve()


def database_url() -> str:
    """`DATABASE_URL` if set, else the SQLite URL for `database_path()`."""
    return os.environ.get("DATABASE_URL") or f"sqlite:///{database_path()}"
