"""Unit tests for the require_admin auth dependency (api/middleware/auth.py).

Covers the guard require_admin adds on top of validate_required_api_key:
unauthenticated -> 401 (existing behaviour, exercised here as the
foundation require_admin builds on), authenticated non-admin -> 403,
authenticated admin -> passes through unchanged.
"""

import pytest
from fastapi import HTTPException

from videoannotator.api.middleware.auth import require_admin, validate_required_api_key


class TestValidateRequiredApiKey:
    async def test_no_credentials_raises_401(self):
        with pytest.raises(HTTPException) as exc_info:
            await validate_required_api_key(None)
        assert exc_info.value.status_code == 401


class TestRequireAdmin:
    async def test_non_admin_user_raises_403(self):
        user = {
            "id": "u1",
            "username": "alice",
            "email": "alice@example.com",
            "is_admin": False,
        }
        with pytest.raises(HTTPException) as exc_info:
            await require_admin(user=user)
        assert exc_info.value.status_code == 403
        assert (
            exc_info.value.detail
            == "Administrator privileges required for this action."
        )

    async def test_user_missing_is_admin_key_raises_403(self):
        # TokenManager-backed users (dev/test tokens) have no admin concept at
        # all -- missing the key must be treated the same as False, never as
        # an implicit grant.
        user = {"id": "u1", "username": "alice", "email": "alice@example.com"}
        with pytest.raises(HTTPException) as exc_info:
            await require_admin(user=user)
        assert exc_info.value.status_code == 403

    async def test_admin_user_passes_through_unchanged(self):
        user = {
            "id": "u1",
            "username": "admin",
            "email": "admin@example.com",
            "is_admin": True,
        }
        result = await require_admin(user=user)
        assert result is user
