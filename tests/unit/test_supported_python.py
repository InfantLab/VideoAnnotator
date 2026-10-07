"""The supported Python range is declared in four places; keep them in step."""

import logging
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from videoannotator import version

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _project() -> dict:
    return tomllib.loads(PYPROJECT.read_text())["project"]


def test_matches_requires_python():
    spec = _project()["requires-python"]
    lower = re.search(r">=\s*3\.(\d+)", spec)
    upper = re.search(r"<\s*3\.(\d+)", spec)
    assert lower and upper, spec
    declared = tuple((3, minor) for minor in range(int(lower[1]), int(upper[1])))
    assert declared == version.SUPPORTED_PYTHON


def test_matches_classifiers():
    prefix = "Programming Language :: Python :: 3."
    classified = {
        (3, int(c.removeprefix(prefix)))
        for c in _project()["classifiers"]
        if c.startswith(prefix) and c.removeprefix(prefix).isdigit()
    }
    assert classified == set(version.SUPPORTED_PYTHON)


@pytest.fixture
def fresh_warning_state(monkeypatch):
    monkeypatch.setattr(version, "_unsupported_python_warned", False)


def _warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "is not supported" in r.getMessage()
    ]


@pytest.mark.parametrize("supported", version.SUPPORTED_PYTHON)
def test_no_warning_on_supported(monkeypatch, caplog, fresh_warning_state, supported):
    monkeypatch.setattr(sys, "version_info", (*supported, 0, "final", 0))
    with caplog.at_level(logging.WARNING):
        version.warn_if_unsupported_python()
    assert _warnings(caplog) == []


def test_warns_once_on_unsupported(monkeypatch, caplog, fresh_warning_state):
    monkeypatch.setattr(sys, "version_info", (3, 99, 0, "final", 0))
    with caplog.at_level(logging.WARNING):
        version.warn_if_unsupported_python()
        version.warn_if_unsupported_python()
    messages = _warnings(caplog)
    assert len(messages) == 1
    assert "3.99" in messages[0]
    assert "3.12, 3.13" in messages[0]


def _triton_cache_dir_after_import(env: dict[str, str]) -> str:
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, videoannotator; print(os.environ['TRITON_CACHE_DIR'])",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def test_triton_cache_is_per_python_version():
    env = {k: v for k, v in os.environ.items() if k != "TRITON_CACHE_DIR"}
    expected = f"py{sys.version_info.major}.{sys.version_info.minor}"
    assert Path(_triton_cache_dir_after_import(env)).name == expected


def test_triton_cache_respects_user_setting(tmp_path):
    env = {**os.environ, "TRITON_CACHE_DIR": str(tmp_path)}
    assert _triton_cache_dir_after_import(env) == str(tmp_path)


def _env_after_import(env: dict[str, str], var: str) -> str:
    return subprocess.run(
        [
            sys.executable,
            "-c",
            f"import os, videoannotator; print(os.environ[{var!r}])",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def test_pyannote_telemetry_off_by_default():
    env = {k: v for k, v in os.environ.items() if k != "PYANNOTE_METRICS_ENABLED"}
    assert _env_after_import(env, "PYANNOTE_METRICS_ENABLED") == "0"


def test_pyannote_telemetry_respects_user_choice():
    env = {**os.environ, "PYANNOTE_METRICS_ENABLED": "1"}
    assert _env_after_import(env, "PYANNOTE_METRICS_ENABLED") == "1"
