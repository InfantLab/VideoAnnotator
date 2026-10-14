"""Logs go to one per-user directory, not ./logs (v1.6.0 Phase 1)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from videoannotator.utils import logging_config as lc


def test_setting_wins(tmp_path, monkeypatch):
    monkeypatch.setenv(lc.LOG_DIR_ENV, str(tmp_path / "mine"))
    assert lc.logs_dir() == (tmp_path / "mine").resolve()


def test_setting_expands_home(monkeypatch):
    monkeypatch.setenv(lc.LOG_DIR_ENV, "~/va-logs")
    assert lc.logs_dir() == (Path.home() / "va-logs").resolve()


@pytest.mark.parametrize(
    ("platform", "expected"),
    [
        ("linux", Path("state-home") / "videoannotator" / "logs"),
        ("darwin", Path.home() / "Library" / "Logs" / "videoannotator"),
        ("win32", Path("local-app-data") / "videoannotator" / "logs"),
    ],
)
def test_default_is_per_user(monkeypatch, platform, expected):
    monkeypatch.delenv(lc.LOG_DIR_ENV, raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", "state-home")
    monkeypatch.setenv("LOCALAPPDATA", "local-app-data")
    monkeypatch.setattr(sys, "platform", platform)
    assert lc.logs_dir() == expected.resolve()


def test_linux_default_without_xdg(monkeypatch):
    monkeypatch.delenv(lc.LOG_DIR_ENV, raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    assert lc.logs_dir() == Path.home() / ".local" / "state" / "videoannotator" / "logs"


def test_logging_does_not_write_to_the_working_directory(tmp_path):
    """The old default was ./logs, so each start directory got its own logs."""
    logs = tmp_path / "logs-here"
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    env = {**os.environ, lc.LOG_DIR_ENV: str(logs)}
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from videoannotator.utils.logging_config import get_logger; "
            "get_logger('api').info('hello')",
        ],
        cwd=cwd,
        env=env,
        check=True,
    )
    assert not (cwd / "logs").exists()
    assert "hello" in (logs / "api_server.log").read_text()
