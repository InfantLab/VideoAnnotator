"""The viewer only works from 127.0.0.1: it pins its API URL there and the
browser keeps the API token per origin. Browsers that open it at `localhost`
are redirected, and every link the CLI prints points at 127.0.0.1.
"""

import pytest
from fastapi.testclient import TestClient

from videoannotator.cli import _viewer_connect_url


@pytest.fixture
def client():
    from videoannotator.api.main import app

    return TestClient(app, base_url="http://localhost:18011")


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/viewer/", "http://127.0.0.1:18011/viewer/"),
        ("/viewer/jobs/abc", "http://127.0.0.1:18011/viewer/jobs/abc"),
        (
            "/viewer-connect?token=va_x",
            "http://127.0.0.1:18011/viewer-connect?token=va_x",
        ),
    ],
)
def test_localhost_viewer_redirects_to_loopback_ip(client, path, expected):
    response = client.get(path, follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == expected


def test_localhost_api_is_not_redirected(client):
    response = client.get("/api/v1/health", follow_redirects=False)

    assert response.status_code == 200


def test_loopback_ip_viewer_connect_is_served():
    from videoannotator.api.main import app

    response = TestClient(app, base_url="http://127.0.0.1:18011").get(
        "/viewer-connect?token=va_x", follow_redirects=False
    )

    assert response.status_code == 200
    assert "va_x" in response.text


def test_viewer_connect_url_uses_loopback_ip():
    assert _viewer_connect_url("va_x", port=18011).startswith(
        "http://127.0.0.1:18011/viewer-connect?token="
    )
