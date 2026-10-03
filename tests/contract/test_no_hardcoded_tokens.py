"""No hard-coded tokens; artifacts and debug endpoints follow AUTH_REQUIRED.

These endpoints used an older auth dependency that ignored AUTH_REQUIRED and
accepted "dev-token" / "test-token" on any server, so anyone could download a
job's artifacts (videos, annotations) with a well-known string.
"""

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from videoannotator.api.database import reset_storage_backend, set_database_path
from videoannotator.api.main import create_app

PROTECTED = [
    "/api/v1/jobs/some-job/artifacts",
    "/api/v1/debug/token-info",
    "/api/v1/debug/jobs/some-job",
    "/api/v1/debug/request-log",
]


@pytest.fixture
def client():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = Path(f.name)
    set_database_path(db_path)
    reset_storage_backend()
    yield TestClient(create_app())
    reset_storage_backend()
    db_path.unlink(missing_ok=True)


@pytest.mark.parametrize("token", ["dev-token", "test-token"])
@pytest.mark.parametrize("path", PROTECTED)
def test_well_known_tokens_rejected_when_auth_required(
    client, monkeypatch, path, token
):
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    response = client.get(path, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401, (path, response.status_code)


def test_artifacts_reachable_without_token_when_auth_off(client, monkeypatch):
    monkeypatch.setenv("AUTH_REQUIRED", "false")
    response = client.get("/api/v1/jobs/no-such-job/artifacts")
    assert response.status_code != 401
