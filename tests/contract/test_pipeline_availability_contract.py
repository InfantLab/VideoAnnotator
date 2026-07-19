"""Contract test: unavailable pipelines are omitted from GET /api/v1/pipelines
by default (FR-005), and included with `available: false` + `install_hint`
when `?include_unavailable=true` is passed.

Contract: specs/004-extras-based-install/contracts/unavailable-pipeline-error.md
"""

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import videoannotator.api.v1.pipelines as pipelines_module
from videoannotator.api.database import reset_storage_backend, set_database_path
from videoannotator.api.main import create_app


@pytest.fixture
def client():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = Path(f.name)
    set_database_path(db_path)
    reset_storage_backend()

    app = create_app()
    yield TestClient(app)

    reset_storage_backend()
    if db_path.exists():
        db_path.unlink()


def _make_unavailable(monkeypatch, unavailable_extras: set[str]):
    """Force any pipeline requiring one of `unavailable_extras` to appear
    unavailable, regardless of what's actually installed in this test
    environment."""

    def fake_extras_available(requires_extras):
        return not any(extra in unavailable_extras for extra in requires_extras)

    monkeypatch.setattr(pipelines_module, "extras_available", fake_extras_available)


class TestPipelineAvailabilityContract:
    def test_default_listing_omits_unavailable_pipelines(self, client, monkeypatch):
        _make_unavailable(monkeypatch, {"face-laion"})

        response = client.get("/api/v1/pipelines/")

        assert response.status_code == 200
        names = {p["name"] for p in response.json()["pipelines"]}
        assert "face_laion_clip" not in names

    def test_default_listing_keeps_available_pipelines(self, client, monkeypatch):
        _make_unavailable(monkeypatch, {"face-laion"})

        response = client.get("/api/v1/pipelines/")

        names = {p["name"] for p in response.json()["pipelines"]}
        # Pipelines that don't need the unavailable extra are unaffected.
        assert "scene_detection" in names

    def test_include_unavailable_shows_availability_and_install_hint(
        self, client, monkeypatch
    ):
        _make_unavailable(monkeypatch, {"face-laion"})

        response = client.get(
            "/api/v1/pipelines/", params={"include_unavailable": "true"}
        )

        assert response.status_code == 200
        pipelines_by_name = {p["name"]: p for p in response.json()["pipelines"]}
        assert "face_laion_clip" in pipelines_by_name

        entry = pipelines_by_name["face_laion_clip"]
        assert entry["available"] is False
        assert entry["install_hint"] == "pip install videoannotator[face-laion]"

    def test_available_pipelines_report_available_true(self, client, monkeypatch):
        _make_unavailable(monkeypatch, {"face-laion"})

        response = client.get(
            "/api/v1/pipelines/", params={"include_unavailable": "true"}
        )

        pipelines_by_name = {p["name"]: p for p in response.json()["pipelines"]}
        entry = pipelines_by_name["scene_detection"]
        assert entry["available"] is True
        assert entry["install_hint"] is None
