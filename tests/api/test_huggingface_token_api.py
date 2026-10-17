"""`/api/v1/system/huggingface-token` (spec 031, contracts/api.md)."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from videoannotator import secrets_store
from videoannotator.api import readiness
from videoannotator.api.main import create_app
from videoannotator.api.middleware.auth import (
    validate_api_key,
    validate_required_api_key,
)

ADMIN = {"id": "admin-1", "username": "admin", "is_admin": True}
USER = {"id": "user-1", "username": "alice", "is_admin": False}
TOKEN = "hf_SECRETsecretSECRET123"
URL = "/api/v1/system/huggingface-token"


def _hub(whoami: int):
    def get(path, token):
        assert path == "/api/whoami-v2"
        return whoami, b'{"name": "caspar"}'

    return get


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEOANNOTATOR_DB_PATH", str(tmp_path / "videoannotator.db"))
    for name in ("HUGGINGFACE_TOKEN", "HF_AUTH_TOKEN", "HF_TOKEN"):
        monkeypatch.delenv(name, raising=False)


def _client(user):
    app = create_app()
    app.dependency_overrides[validate_required_api_key] = lambda: user
    app.dependency_overrides[validate_api_key] = lambda: user
    return TestClient(app)


@pytest.fixture
def admin():
    return _client(ADMIN)


@pytest.fixture
def user():
    return _client(USER)


def test_not_set(user):
    assert user.get(URL).json() == {
        "set": False,
        "source": None,
        "env_var": None,
        "account": None,
        "check": None,
    }


def test_save_checks_and_reports_account(admin):
    with patch.object(readiness, "_hf_get", side_effect=_hub(200)):
        resp = admin.put(URL, json={"token": f"  {TOKEN}  "})
    assert resp.status_code == 200
    assert resp.json() == {
        "set": True,
        "source": "saved",
        "env_var": None,
        "account": "caspar",
        "check": "ok",
    }
    assert secrets_store.read_saved_huggingface_token() == TOKEN


def test_rejected_token_is_not_saved(admin):
    with patch.object(readiness, "_hf_get", side_effect=_hub(401)):
        resp = admin.put(URL, json={"token": TOKEN})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "HF_TOKEN_REJECTED"
    assert secrets_store.read_saved_huggingface_token() is None


def test_unreachable_hub_saves_unchecked(admin):
    with patch.object(readiness, "_hf_get", return_value=(None, b"")):
        resp = admin.put(URL, json={"token": TOKEN})
    assert resp.status_code == 200
    assert resp.json()["check"] == "unknown"
    assert secrets_store.read_saved_huggingface_token() == TOKEN


def test_empty_token(admin):
    resp = admin.put(URL, json={"token": "   "})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "HF_TOKEN_EMPTY"


def test_remove(admin):
    secrets_store.save_huggingface_token(TOKEN)
    resp = admin.delete(URL)
    assert resp.status_code == 200
    assert resp.json()["set"] is False
    assert secrets_store.read_saved_huggingface_token() is None


def test_environment_token_wins_and_is_named(admin, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    with patch.object(readiness, "_hf_get", side_effect=_hub(200)):
        put = admin.put(URL, json={"token": TOKEN}).json()
        after_delete = admin.delete(URL).json()
    assert (put["source"], put["env_var"]) == ("environment", "HF_TOKEN")
    assert after_delete["set"] is True
    assert after_delete["source"] == "environment"


def test_non_admin_cannot_change_it(user):
    assert user.put(URL, json={"token": TOKEN}).status_code == 403
    assert user.delete(URL).status_code == 403
    assert secrets_store.read_saved_huggingface_token() is None


def test_saved_token_is_used_without_restart(admin):
    assert not readiness._secret_is_set("HUGGINGFACE_TOKEN", [])
    with patch.object(readiness, "_hf_get", side_effect=_hub(200)):
        admin.put(URL, json={"token": TOKEN})
    assert readiness._secret_is_set("HUGGINGFACE_TOKEN", [])


def test_token_never_in_responses_or_logs(admin, caplog):
    """SC-002."""
    bodies = []
    with (
        caplog.at_level(logging.DEBUG),
        patch.object(readiness, "_hf_get", side_effect=_hub(200)),
    ):
        bodies.append(admin.put(URL, json={"token": TOKEN}).text)
        bodies.append(admin.get(URL).text)
        bodies.append(admin.get("/api/v1/pipelines/speaker_diarization").text)
        bodies.append(admin.delete(URL).text)
    with patch.object(readiness, "_hf_get", side_effect=_hub(401)):
        bodies.append(admin.put(URL, json={"token": TOKEN}).text)
    assert all(TOKEN not in b for b in bodies)
    assert TOKEN not in caplog.text
