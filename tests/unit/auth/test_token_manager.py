"""Unit tests for auth/token_manager.py.

Focuses on JWT generation/validation (no DB needed) and mocks DB operations
for API-key paths.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import jwt
import pytest

from videoannotator.auth.token_manager import (
    SecureTokenManager,
    TokenInfo,
    TokenType,
    get_token_manager,
    initialize_token_manager,
)

# ---------------------------------------------------------------------------
# TokenType enum
# ---------------------------------------------------------------------------


class TestTokenType:
    def test_all_values(self):
        assert TokenType.API_KEY.value == "api_key"
        assert TokenType.SESSION.value == "session"
        assert TokenType.REFRESH.value == "refresh"
        assert TokenType.CLIENT_APP.value == "client_app"


# ---------------------------------------------------------------------------
# TokenInfo dataclass
# ---------------------------------------------------------------------------


class TestTokenInfo:
    def test_creation(self):
        info = TokenInfo(
            token_id="t1",
            user_id="u1",
            username="alice",
            email="alice@example.com",
            token_type=TokenType.SESSION,
            scopes=["read"],
            created_at=datetime(2025, 1, 1),
            expires_at=datetime(2025, 1, 2),
            last_used_at=None,
            is_active=True,
            metadata={},
        )
        assert info.token_id == "t1"
        assert info.token_type == TokenType.SESSION
        assert info.is_active is True


# ---------------------------------------------------------------------------
# SecureTokenManager — constructor & helpers
# ---------------------------------------------------------------------------


class TestSecureTokenManagerInit:
    def test_default_secret_key_generated(self):
        mgr = SecureTokenManager()
        assert mgr.secret_key is not None
        assert len(mgr.secret_key) > 20

    def test_custom_secret_key(self):
        mgr = SecureTokenManager(secret_key="my-secret")
        assert mgr.secret_key == "my-secret"

    def test_deprecated_params_ignored(self):
        mgr = SecureTokenManager(tokens_file="/tmp/f", encryption_key=b"key")
        assert mgr.secret_key is not None

    def test_persist_token_state_is_noop(self):
        mgr = SecureTokenManager()
        mgr.persist_token_state()  # should not raise


# ---------------------------------------------------------------------------
# JWT session tokens — generate & validate (no DB)
# ---------------------------------------------------------------------------


class TestJWTSessionTokens:
    def setup_method(self):
        self.mgr = SecureTokenManager(secret_key="test-secret-key")

    def test_generate_session_token_returns_jwt(self):
        token, info = self.mgr.generate_session_token(
            user_id="u1", username="alice", email="alice@test.com"
        )
        assert isinstance(token, str)
        assert info.token_type == TokenType.SESSION
        assert info.username == "alice"
        assert info.is_active is True

    def test_session_token_decodable(self):
        token, _ = self.mgr.generate_session_token(
            user_id="u1", username="bob", email="bob@test.com"
        )
        payload = jwt.decode(token, "test-secret-key", algorithms=["HS256"])
        assert payload["username"] == "bob"
        assert payload["token_type"] == "session"

    def test_session_token_custom_scopes(self):
        token, info = self.mgr.generate_session_token(
            user_id="u1",
            username="carol",
            email="carol@test.com",
            scopes=["read"],
        )
        assert info.scopes == ["read"]
        payload = jwt.decode(token, "test-secret-key", algorithms=["HS256"])
        assert payload["scopes"] == ["read"]

    def test_session_token_custom_expiry(self):
        _, info = self.mgr.generate_session_token(
            user_id="u1",
            username="dave",
            email="d@test.com",
            expires_in_hours=1,
        )
        assert info.expires_at is not None
        delta = info.expires_at - info.created_at
        assert delta.total_seconds() == pytest.approx(3600, abs=5)

    def test_validate_valid_jwt(self):
        token, _ = self.mgr.generate_session_token(
            user_id="u1", username="eve", email="eve@test.com"
        )
        result = self.mgr.validate_token(token)
        assert result is not None
        assert result.username == "eve"
        assert result.token_type == TokenType.SESSION

    def test_validate_expired_jwt_returns_none(self):
        payload = {
            "token_id": "t1",
            "user_id": "u1",
            "username": "frank",
            "email": "frank@test.com",
            "scopes": ["read"],
            "token_type": "session",
            "iat": (datetime.now() - timedelta(hours=2)).timestamp(),
            "exp": (datetime.now() - timedelta(hours=1)).timestamp(),
        }
        expired_token = jwt.encode(payload, "test-secret-key", algorithm="HS256")
        result = self.mgr.validate_token(expired_token)
        assert result is None

    def test_validate_wrong_secret_returns_none(self):
        other_mgr = SecureTokenManager(secret_key="other-secret")
        token, _ = other_mgr.generate_session_token(
            user_id="u1", username="gina", email="gina@test.com"
        )
        result = self.mgr.validate_token(token)
        assert result is None

    def test_validate_malformed_jwt_returns_none(self):
        result = self.mgr.validate_token("not-a-jwt")
        assert result is None

    def test_validate_empty_string_returns_none(self):
        result = self.mgr.validate_token("")
        assert result is None


# ---------------------------------------------------------------------------
# API key operations (DB-dependent — mocked)
# ---------------------------------------------------------------------------


class TestAPIKeyOperations:
    def setup_method(self):
        self.mgr = SecureTokenManager(secret_key="test-secret-key")

    @patch("videoannotator.auth.token_manager.db_module")
    @patch("videoannotator.auth.token_manager.UserCRUD")
    @patch("videoannotator.auth.token_manager.APIKeyCRUD")
    def test_generate_api_key(self, mock_api_crud, mock_user_crud, mock_db):
        mock_session = MagicMock()
        mock_db.SessionLocal.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db.SessionLocal.return_value.__exit__ = MagicMock(return_value=False)

        mock_user = MagicMock()
        mock_user.id = "uid-1"
        mock_user.username = "alice"
        mock_user.email = "alice@test.com"
        mock_user_crud.get_by_username.return_value = mock_user

        mock_api_key = MagicMock()
        mock_api_key.id = "key-1"
        mock_api_key.created_at = datetime(2025, 1, 1)
        mock_api_key.expires_at = None
        mock_api_key.last_used = None
        mock_api_key.is_active = True
        mock_api_crud.create.return_value = (mock_api_key, "va_test123")

        token_string, info = self.mgr.generate_api_key(
            user_id="ext-1", username="alice", email="alice@test.com"
        )

        assert token_string == "va_test123"
        assert info.token_type == TokenType.API_KEY
        assert info.username == "alice"
        assert info.is_active is True

    @patch("videoannotator.auth.token_manager.db_module")
    @patch("videoannotator.auth.token_manager.UserCRUD")
    @patch("videoannotator.auth.token_manager.APIKeyCRUD")
    def test_generate_api_key_creates_user_if_missing(
        self, mock_api_crud, mock_user_crud, mock_db
    ):
        mock_session = MagicMock()
        mock_db.SessionLocal.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db.SessionLocal.return_value.__exit__ = MagicMock(return_value=False)

        mock_user_crud.get_by_username.return_value = None  # user not found

        new_user = MagicMock()
        new_user.id = "uid-new"
        new_user.username = "newuser"
        new_user.email = "new@test.com"
        mock_user_crud.create.return_value = new_user

        mock_api_key = MagicMock()
        mock_api_key.id = "key-2"
        mock_api_key.created_at = datetime(2025, 1, 1)
        mock_api_key.expires_at = None
        mock_api_key.last_used = None
        mock_api_key.is_active = True
        mock_api_crud.create.return_value = (mock_api_key, "va_new456")

        token_string, info = self.mgr.generate_api_key(
            user_id="ext-2", username="newuser", email="new@test.com"
        )

        mock_user_crud.create.assert_called_once()
        assert token_string == "va_new456"

    @patch("videoannotator.auth.token_manager.db_module")
    @patch("videoannotator.auth.token_manager.APIKeyCRUD")
    def test_validate_api_key_valid(self, mock_api_crud, mock_db):
        mock_session = MagicMock()
        mock_db.SessionLocal.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db.SessionLocal.return_value.__exit__ = MagicMock(return_value=False)

        mock_user = MagicMock()
        mock_user.username = "alice"
        mock_user.email = "alice@test.com"

        mock_api_key = MagicMock()
        mock_api_key.id = "key-1"
        mock_api_key.user_id = "uid-1"
        mock_api_key.is_active = True
        mock_api_key.expires_at = None
        mock_api_key.created_at = datetime(2025, 1, 1)
        mock_api_key.last_used = None
        mock_api_key.user = mock_user
        mock_api_crud.get_by_token.return_value = mock_api_key

        result = self.mgr.validate_token("va_validtoken123")
        assert result is not None
        assert result.token_type == TokenType.API_KEY
        assert result.username == "alice"

    @patch("videoannotator.auth.token_manager.db_module")
    @patch("videoannotator.auth.token_manager.APIKeyCRUD")
    def test_validate_api_key_not_found(self, mock_api_crud, mock_db):
        mock_session = MagicMock()
        mock_db.SessionLocal.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db.SessionLocal.return_value.__exit__ = MagicMock(return_value=False)
        mock_api_crud.get_by_token.return_value = None

        result = self.mgr.validate_token("va_invalid")
        assert result is None

    @patch("videoannotator.auth.token_manager.db_module")
    @patch("videoannotator.auth.token_manager.APIKeyCRUD")
    def test_validate_api_key_inactive(self, mock_api_crud, mock_db):
        mock_session = MagicMock()
        mock_db.SessionLocal.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db.SessionLocal.return_value.__exit__ = MagicMock(return_value=False)

        mock_api_key = MagicMock()
        mock_api_key.is_active = False
        mock_api_crud.get_by_token.return_value = mock_api_key

        result = self.mgr.validate_token("va_inactive")
        assert result is None

    @patch("videoannotator.auth.token_manager.db_module")
    @patch("videoannotator.auth.token_manager.APIKeyCRUD")
    def test_validate_api_key_expired(self, mock_api_crud, mock_db):
        mock_session = MagicMock()
        mock_db.SessionLocal.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db.SessionLocal.return_value.__exit__ = MagicMock(return_value=False)

        mock_api_key = MagicMock()
        mock_api_key.is_active = True
        mock_api_key.expires_at = datetime(2020, 1, 1)  # past
        mock_api_crud.get_by_token.return_value = mock_api_key

        result = self.mgr.validate_token("va_expired")
        assert result is None

    @patch("videoannotator.auth.token_manager.db_module")
    @patch("videoannotator.auth.token_manager.APIKeyCRUD")
    def test_revoke_api_key_success(self, mock_api_crud, mock_db):
        mock_session = MagicMock()
        mock_db.SessionLocal.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db.SessionLocal.return_value.__exit__ = MagicMock(return_value=False)

        mock_api_key = MagicMock()
        mock_api_key.id = "key-1"
        mock_api_crud.get_by_token.return_value = mock_api_key
        mock_api_crud.revoke.return_value = True

        assert self.mgr.revoke_token("va_torevoke") is True

    @patch("videoannotator.auth.token_manager.db_module")
    @patch("videoannotator.auth.token_manager.APIKeyCRUD")
    def test_revoke_api_key_not_found(self, mock_api_crud, mock_db):
        mock_session = MagicMock()
        mock_db.SessionLocal.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db.SessionLocal.return_value.__exit__ = MagicMock(return_value=False)
        mock_api_crud.get_by_token.return_value = None

        assert self.mgr.revoke_token("va_gone") is False

    def test_revoke_jwt_returns_false(self):
        """JWT revocation is not supported (no blacklist)."""
        assert self.mgr.revoke_token("eyJhbGciOi.jwt.token") is False

    @patch("videoannotator.auth.token_manager.db_module")
    def test_list_all_tokens(self, mock_db):
        mock_session = MagicMock()
        mock_db.SessionLocal.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db.SessionLocal.return_value.__exit__ = MagicMock(return_value=False)

        mock_user = MagicMock()
        mock_user.username = "alice"
        mock_user.email = "alice@test.com"

        mock_key = MagicMock()
        mock_key.user_id = "uid-1"
        mock_key.user = mock_user
        mock_key.created_at = datetime(2025, 1, 1)
        mock_key.expires_at = None
        mock_key.last_used = None
        mock_key.is_active = True
        mock_key.id = "key-1"
        mock_key.key_prefix = "abcdefgh"

        mock_session.query.return_value.filter.return_value.all.return_value = [
            mock_key
        ]

        tokens = self.mgr.list_all_tokens()
        assert len(tokens) == 1
        assert tokens[0].token_id == "<hidden>"
        assert tokens[0].username == "alice"

    @patch("videoannotator.auth.token_manager.db_module")
    def test_get_token_stats(self, mock_db):
        mock_session = MagicMock()
        mock_db.SessionLocal.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db.SessionLocal.return_value.__exit__ = MagicMock(return_value=False)

        mock_query = MagicMock()
        mock_query.count.return_value = 5
        mock_query.filter.return_value.count.return_value = 3
        mock_session.query.return_value = mock_query

        stats = self.mgr.get_token_stats()
        assert stats["total_tokens"] == 5
        assert stats["active_tokens"] == 3
        assert stats["expired_tokens"] == 2

    @patch("videoannotator.auth.token_manager.db_module")
    def test_cleanup_expired_tokens(self, mock_db):
        mock_session = MagicMock()
        mock_db.SessionLocal.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db.SessionLocal.return_value.__exit__ = MagicMock(return_value=False)

        mock_key = MagicMock()
        mock_key.is_active = True
        mock_session.query.return_value.filter.return_value.all.return_value = [
            mock_key
        ]

        count = self.mgr.cleanup_expired_tokens()
        assert count == 1
        assert mock_key.is_active is False
        mock_session.commit.assert_called_once()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


class TestModuleLevelHelpers:
    def teardown_method(self):
        # Reset global state
        import videoannotator.auth.token_manager as mod

        mod._token_manager = None

    def test_get_token_manager_creates_singleton(self):
        mgr1 = get_token_manager()
        mgr2 = get_token_manager()
        assert mgr1 is mgr2

    def test_initialize_token_manager_with_key(self):
        mgr = initialize_token_manager(secret_key="custom-key")
        assert mgr.secret_key == "custom-key"
        assert get_token_manager() is mgr

    def test_initialize_replaces_existing(self):
        mgr1 = initialize_token_manager(secret_key="key1")
        mgr2 = initialize_token_manager(secret_key="key2")
        assert mgr2.secret_key == "key2"
        assert get_token_manager() is mgr2
        assert mgr1 is not mgr2
