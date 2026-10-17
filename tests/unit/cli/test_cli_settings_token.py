"""`videoannotator settings huggingface-token` (spec 031)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from videoannotator import secrets_store
from videoannotator.api import readiness
from videoannotator.cli import app

runner = CliRunner()
TOKEN = "hf_SECRETsecretSECRET123"


def _hub(whoami: int):
    return lambda path, token: (whoami, b'{"name": "caspar"}')


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEOANNOTATOR_DB_PATH", str(tmp_path / "videoannotator.db"))
    for name in ("HUGGINGFACE_TOKEN", "HF_AUTH_TOKEN", "HF_TOKEN"):
        monkeypatch.delenv(name, raising=False)


def _run(*args, stdin=None):
    return runner.invoke(app, ["settings", "huggingface-token", *args], input=stdin)


def test_saves_a_token_from_stdin_without_echoing_it():
    with patch.object(readiness, "_hf_get", side_effect=_hub(200)):
        result = _run(stdin=f"{TOKEN}\n")
    assert result.exit_code == 0, result.output
    assert "accepts it (account: caspar)" in result.output
    assert TOKEN not in result.output
    assert secrets_store.read_saved_huggingface_token() == TOKEN


def test_rejected_token_is_not_saved():
    with patch.object(readiness, "_hf_get", side_effect=_hub(401)):
        result = _run(stdin=f"{TOKEN}\n")
    assert result.exit_code == 1
    assert "Nothing saved" in result.output
    assert secrets_store.read_saved_huggingface_token() is None


def test_empty_input_saves_nothing():
    result = _run(stdin="\n")
    assert result.exit_code == 1
    assert secrets_store.read_saved_huggingface_token() is None


def test_status_and_remove():
    assert "Not set." in _run("--status").output
    secrets_store.save_huggingface_token(TOKEN)
    with patch.object(readiness, "_hf_get", side_effect=_hub(200)):
        out = _run("--status").output
    assert "Set, saved in" in out
    assert "works (account: caspar)" in out
    assert TOKEN not in out
    assert "Removed the saved token." in _run("--remove").output
    assert secrets_store.read_saved_huggingface_token() is None


def test_status_names_the_environment_variable(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    with patch.object(readiness, "_hf_get", side_effect=_hub(200)):
        out = _run("--status").output
    assert "from HF_TOKEN in the environment" in out
