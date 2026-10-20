"""An API client on a server of its own for spec 029's route tests: its own
jobs database and storage root, one shared folder, and a signed-in user.

The `api` fixture is registered for `tests/api` in its conftest.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.fixtures.video_index import JobsDb


@dataclass
class VideoApi:
    client: TestClient
    shared: Path
    jobs: JobsDb
    user: dict[str, Any]

    def as_user(self, user: dict[str, Any]) -> None:
        from videoannotator.api.main import app
        from videoannotator.api.middleware.auth import validate_api_key

        self.user = user
        app.dependency_overrides[validate_api_key] = lambda: user

    def project(self, name: str, **fields: Any) -> dict[str, Any]:
        r = self.client.post("/api/v1/datasets/", json={"name": name} | fields)
        assert r.status_code == 201, r.text
        return r.json()

    def videos(self, **params: Any) -> dict[str, Any]:
        r = self.client.get("/api/v1/videos", params=params)
        assert r.status_code == 200, r.text
        return r.json()


@pytest.fixture
def api(tmp_path, monkeypatch):
    import videoannotator.database.database as db_module
    from videoannotator import projects, video_index
    from videoannotator.api.database import get_storage_backend, reset_storage_backend
    from videoannotator.api.main import app
    from videoannotator.api.middleware.auth import validate_api_key
    from videoannotator.api.v1 import ingest as ingest_module
    from videoannotator.api.v1 import video_list

    shared = tmp_path / "shared"
    shared.mkdir()
    db_path = tmp_path / "server.db"
    monkeypatch.setenv("VIDEOANNOTATOR_DB_PATH", str(db_path))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setattr(ingest_module, "INGEST_ROOTS", str(shared))
    monkeypatch.setattr(video_list, "_places_cache", None)
    reset_storage_backend()
    video_index.clear_cache()
    projects.clear_cache()
    db_module.Base.metadata.create_all(bind=db_module.engine)
    # The datasets database is the session's: earlier tests' projects would
    # count here, and a user of its own keeps last-used settings apart.
    from videoannotator.database.models import SavedDataset, SavedPipelinePreset

    with db_module.SessionLocal() as db:
        db.query(SavedDataset).delete()
        db.query(SavedPipelinePreset).delete()
        db.commit()
    user = {"id": str(uuid.uuid4()), "username": "caspar", "is_admin": False}
    app.dependency_overrides[validate_api_key] = lambda: user

    jobs = JobsDb(tmp_path, storage=get_storage_backend())
    yield VideoApi(TestClient(app), shared, jobs, user)

    app.dependency_overrides.clear()
    reset_storage_backend()
    video_index.clear_cache()
    projects.clear_cache()
