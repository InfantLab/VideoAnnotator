"""One database location for both database layers, not ./videoannotator.db."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from videoannotator import database_location as dl


def test_db_path_setting_wins(tmp_path, monkeypatch):
    monkeypatch.setenv(dl.DB_PATH_ENV, str(tmp_path / "jobs.db"))
    assert dl.database_path() == (tmp_path / "jobs.db").resolve()


def test_default_is_in_the_per_user_data_dir(monkeypatch):
    monkeypatch.delenv(dl.DB_PATH_ENV, raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", "/data-home")
    monkeypatch.setattr(sys, "platform", "linux")
    assert (
        dl.database_path()
        == Path("/data-home/videoannotator/videoannotator.db").resolve()
    )


def test_url_follows_the_path(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv(dl.DB_PATH_ENV, str(tmp_path / "jobs.db"))
    assert dl.database_url() == f"sqlite:///{(tmp_path / 'jobs.db').resolve()}"


def test_database_url_wins(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://db/va")
    monkeypatch.setenv(dl.DB_PATH_ENV, "/ignored.db")
    assert dl.database_url() == "postgresql://db/va"


@pytest.mark.parametrize("setting", ["db_path", "default"])
def test_both_layers_use_one_file(tmp_path, setting):
    """The SQLAlchemy layer used to read only DATABASE_URL, so setting
    VIDEOANNOTATOR_DB_PATH alone split the data across two files."""
    cwd = tmp_path / "start-here"
    cwd.mkdir()
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("DATABASE_URL", dl.DB_PATH_ENV, "XDG_DATA_HOME")
    }
    if setting == "db_path":
        expected = tmp_path / "chosen" / "jobs.db"
        env[dl.DB_PATH_ENV] = str(expected)
    else:
        # Every platform's per-user base (XDG on Linux, ~/Library on macOS,
        # LOCALAPPDATA on Windows) pointed into tmp_path.
        home = tmp_path / "home"
        env.update(
            XDG_DATA_HOME=str(home / ".local" / "share"),
            LOCALAPPDATA=str(home / "AppData" / "Local"),
            HOME=str(home),
            USERPROFILE=str(home),
        )
        expected = None
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json\n"
            "from videoannotator.database import database as sa\n"
            "from videoannotator.api.database import get_storage_backend\n"
            "sa.create_tables()\n"
            "print(json.dumps([sa.DATABASE_URL,"
            " str(get_storage_backend().database_path)]))",
        ],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    sa_url, backend_path = json.loads(out.strip().splitlines()[-1])
    sa_path = Path(sa_url.removeprefix("sqlite:///"))
    assert sa_path.exists()
    assert sa_path.samefile(backend_path)
    if expected is not None:
        assert sa_path.samefile(expected)
    else:
        assert sa_path.resolve().is_relative_to((tmp_path / "home").resolve())
        assert sa_path.name == "videoannotator.db"
    assert not (cwd / "videoannotator.db").exists()
