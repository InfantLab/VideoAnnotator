"""Tests for GET /api/v1/auth/me.

This endpoint exists specifically so a client (e.g. the viewer) that hits a
403 from an admin-only endpoint (like the extras-install action) has a way
to self-diagnose: check whether it's even authenticated as an admin.
"""

import pytest
from fastapi.testclient import TestClient

from videoannotator.api.main import create_app
from videoannotator.api.middleware.auth import validate_required_api_key

ADMIN_USER = {
    "id": "admin-1",
    "username": "admin",
    "email": "admin@example.com",
    "is_admin": True,
}
NON_ADMIN_USER = {
    "id": "user-1",
    "username": "alice",
    "email": "alice@example.com",
    "is_admin": False,
}


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


class TestAuthMe:
    def test_unauthenticated_returns_401(self, client):
        resp = client.get("/api/v1/auth/me")
        assert resp.status_code == 401

    def test_admin_user_sees_is_admin_true(self, client):
        client.app.dependency_overrides[validate_required_api_key] = lambda: ADMIN_USER
        resp = client.get("/api/v1/auth/me")
        assert resp.status_code == 200
        body = resp.json()
        assert body["is_admin"] is True
        assert body["username"] == "admin"
        assert body["email"] == "admin@example.com"

    def test_non_admin_user_sees_is_admin_false(self, client):
        client.app.dependency_overrides[validate_required_api_key] = (
            lambda: NON_ADMIN_USER
        )
        resp = client.get("/api/v1/auth/me")
        assert resp.status_code == 200
        body = resp.json()
        assert body["is_admin"] is False
        assert body["username"] == "alice"

    def test_missing_is_admin_key_defaults_false(self, client):
        # TokenManager-backed users (dev/test tokens) have no is_admin key at
        # all in their user dict -- must default to false, not error.
        client.app.dependency_overrides[validate_required_api_key] = lambda: {
            "id": "u1",
            "username": "devtoken",
        }
        resp = client.get("/api/v1/auth/me")
        assert resp.status_code == 200
        assert resp.json()["is_admin"] is False
