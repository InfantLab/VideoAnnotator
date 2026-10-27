"""`/api/v1/system/locations`: where the server keeps things (UAT #16)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from videoannotator.api.main import create_app
from videoannotator.api.middleware.auth import validate_api_key


def test_locations(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEOANNOTATOR_DB_PATH", str(tmp_path / "va.db"))
    monkeypatch.setenv("VIDEOANNOTATOR_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("VIDEOANNOTATOR_LOG_DIR", str(tmp_path / "logs"))
    app = create_app()
    app.dependency_overrides[validate_api_key] = lambda: None
    body = TestClient(app).get("/api/v1/system/locations").json()
    assert body["database"]["path"] == str((tmp_path / "va.db").resolve())
    assert body["models"]["path"] == str((tmp_path / "models").resolve())
    assert body["logs"]["path"].startswith(str(tmp_path / "logs"))
    assert set(body) == {"results", "models", "logs", "database"}
