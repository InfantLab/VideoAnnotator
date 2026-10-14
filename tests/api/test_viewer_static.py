"""The bundled viewer is a single-page app: unknown routes get the app shell,
but a missing build asset stays a 404. A tab opened before an upgrade asks for
the old build's hashed files; answering with index.html (HTML, status 200)
crashes it with "Failed to fetch dynamically imported module".
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from videoannotator.api.main import SPAStaticFiles


@pytest.fixture
def client(tmp_path):
    (tmp_path / "index.html").write_text("<html>shell</html>")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "Datasets-NEW.js").write_text("export default 1")
    app = FastAPI()
    app.mount("/viewer", SPAStaticFiles(directory=tmp_path, html=True))
    return TestClient(app)


def test_client_route_gets_app_shell(client):
    response = client.get("/viewer/jobs/abc")

    assert response.status_code == 200
    assert "shell" in response.text
    assert response.headers["cache-control"] == "no-cache"


def test_missing_build_asset_is_404(client):
    assert client.get("/viewer/assets/Datasets-OLD.js").status_code == 404


def test_existing_build_asset_is_served_cacheable(client):
    response = client.get("/viewer/assets/Datasets-NEW.js")

    assert response.status_code == 200
    assert "cache-control" not in response.headers


def test_app_shell_is_revalidated(client):
    assert client.get("/viewer/").headers["cache-control"] == "no-cache"
