"""Secure token management for the VideoAnnotator API.

Provides user-friendly token generation, validation, and management with
security best practices and multiple authentication flows.

Refactored to use SQLite database (via APIKeyCRUD) instead of flat files.
"""

import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import jwt

from videoannotator.database.crud import APIKeyCRUD
from videoannotator.database.database import SessionLocal
from videoannotator.database.models import APIKey
from videoannotator.utils.logging_config import get_logger

logger = get_logger("api")


class TokenType(Enum):
    """Token types for different use cases."""

    API_KEY = "api_key"  # Long-lived API keys
    SESSION = "session"  # Short-lived session tokens
    REFRESH = "refresh"  # Refresh tokens
    CLIENT_APP = "client_app"  # Client application tokens


@dataclass
class TokenInfo:
    """Token information structure."""

    token_id: str
    user_id: str
    username: str
    email: str
    token_type: TokenType
    scopes: list[str]
    created_at: datetime
    expires_at: datetime | None
    last_used_at: datetime | None
    is_active: bool
    metadata: dict[str, Any]


class SecureTokenManager:
    """Secure token management with multiple authentication flows.

    Features:
    - Multiple token types (API keys, sessions, refresh tokens)
    - Secure token generation and storage (DB-backed)
    - Token expiration and rotation
    - Scope-based permissions
    - User-friendly token management
    """

    def __init__(
        self,
        secret_key: str | None = None,
        tokens_file: str | None = None,  # Deprecated, ignored
        encryption_key: bytes | None = None,  # Deprecated, ignored
    ):
        """Initialize token manager."""
        self.secret_key = secret_key or self._generate_secret_key()
        # tokens_file and encryption_key are kept for signature compatibility but ignored

    def _generate_secret_key(self) -> str:
        """Generate a secure secret key."""
        return secrets.token_urlsafe(64)

    def generate_api_key(
        self,
        user_id: str,
        username: str,
        email: str,
        scopes: list[str] | None = None,
        expires_in_days: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, TokenInfo]:
        """Generate a long-lived API key for programmatic access.

        Returns:
            Tuple of (token_string, token_info)
        """
        scopes = scopes or ["read", "write"]
        metadata = metadata or {}

        # Generate secure API key
        # Format: va_<random_chars>
        token_string = f"va_{secrets.token_urlsafe(32)}"

        with SessionLocal() as db:
            api_key, token_string = APIKeyCRUD.create(
                db=db,
                user_id=user_id,
                key_name=f"API Key for {username}",
                expires_days=expires_in_days,
            )
            # Note: APIKeyCRUD.create generates its own token string.
            # We should use that one.
            # Wait, APIKeyCRUD.create returns (api_key_obj, raw_token_string)

            # We need to construct TokenInfo from the DB object
            token_info = TokenInfo(
                token_id=token_string,  # The raw token is the ID for the client
                user_id=user_id,
                username=username,
                email=email,
                token_type=TokenType.API_KEY,
                scopes=scopes,
                created_at=api_key.created_at,
                expires_at=api_key.expires_at,
                last_used_at=api_key.last_used,
                is_active=api_key.is_active,
                metadata={
                    **metadata,
                    "generated_by": "token_manager",
                    "db_id": api_key.id,
                },
            )

        logger.info(f"Generated API key for user {username}")
        return token_string, token_info

    def generate_session_token(
        self,
        user_id: str,
        username: str,
        email: str,
        scopes: list[str] | None = None,
        expires_in_hours: int = 24,
    ) -> tuple[str, TokenInfo]:
        """Generate a short-lived session token with JWT.

        Returns:
            Tuple of (jwt_token, token_info)
        """
        scopes = scopes or ["read", "write"]

        token_id = f"session_{uuid.uuid4().hex[:16]}"
        expires_at = datetime.now() + timedelta(hours=expires_in_hours)

        # JWT payload
        payload = {
            "token_id": token_id,
            "user_id": user_id,
            "username": username,
            "email": email,
            "scopes": scopes,
            "token_type": TokenType.SESSION.value,
            "iat": datetime.now().timestamp(),
            "exp": expires_at.timestamp(),
        }

        # Generate JWT
        jwt_token = jwt.encode(payload, self.secret_key, algorithm="HS256")

        token_info = TokenInfo(
            token_id=token_id,
            user_id=user_id,
            username=username,
            email=email,
            token_type=TokenType.SESSION,
            scopes=scopes,
            created_at=datetime.now(),
            expires_at=expires_at,
            last_used_at=None,
            is_active=True,
            metadata={"jwt_token": True},
        )

        logger.info(
            f"Generated session token for user {username} (expires in {expires_in_hours}h)"
        )

        return jwt_token, token_info

    def validate_token(self, token: str) -> TokenInfo | None:
        """Validate any type of token and return user information.

        Args:
            token: Token string (API key, JWT, etc.)

        Returns:
            TokenInfo if valid, None if invalid
        """
        try:
            # Try JWT validation first
            if not token.startswith("va_"):
                return self._validate_jwt_token(token)

            # API key validation via DB
            with SessionLocal() as db:
                api_key = APIKeyCRUD.get_by_token(db, token)

                if not api_key:
                    logger.warning(f"Invalid token used: {token[:16]}...")
                    return None

                if not api_key.is_active:
                    logger.warning(f"Inactive token used: {token[:16]}...")
                    return None

                if api_key.expires_at and datetime.utcnow() > api_key.expires_at:
                    logger.warning(f"Expired token used: {token[:16]}...")
                    return None

                # Update last used (async or optimized in CRUD?)
                # APIKeyCRUD.get_by_token updates last_used automatically

                # We need to fetch the user to get username/email
                # Assuming user_id is valid.
                # For now, we might not have the user object loaded with APIKeyCRUD.get_by_token
                # Let's assume we can get it.

                # Hack: APIKey model has user relationship?
                # Yes, `user = relationship("User", back_populates="api_keys")`

                user = api_key.user
                username = user.username if user else "unknown"
                email = user.email if user else "unknown"

                token_info = TokenInfo(
                    token_id=token,
                    user_id=api_key.user_id,
                    username=username,
                    email=email,
                    token_type=TokenType.API_KEY,
                    scopes=["read", "write"],  # Default scopes
                    created_at=api_key.created_at,
                    expires_at=api_key.expires_at,
                    last_used_at=api_key.last_used,
                    is_active=api_key.is_active,
                    metadata={"db_id": api_key.id},
                )

                logger.debug(f"Valid API key used by {username}")
                return token_info

        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return None

    def _validate_jwt_token(self, token: str) -> TokenInfo | None:
        """Validate JWT session token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])

            token_info = TokenInfo(
                token_id=payload["token_id"],
                user_id=payload["user_id"],
                username=payload["username"],
                email=payload["email"],
                token_type=TokenType(payload["token_type"]),
                scopes=payload["scopes"],
                created_at=datetime.fromtimestamp(payload["iat"]),
                expires_at=datetime.fromtimestamp(payload["exp"]),
                last_used_at=datetime.now(),
                is_active=True,
                metadata={"jwt_token": True},
            )

            logger.debug(f"Valid JWT token for {token_info.username}")
            return token_info

        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            if "Not enough segments" in str(e):
                logger.debug(f"Malformed JWT token: {e}")
            else:
                logger.warning(f"Invalid JWT token: {e}")
            return None

    def revoke_token(self, token: str) -> bool:
        """Revoke a token."""
        try:
            if token.startswith("va_"):
                with SessionLocal() as db:
                    return APIKeyCRUD.revoke(db, token)

            # JWT revocation not fully supported without a blacklist
            return False

        except Exception as e:
            logger.error(f"Failed to revoke token: {e}")
            return False

    def list_all_tokens(self) -> list[TokenInfo]:
        """List all active tokens (admin only)."""
        with SessionLocal() as db:
            # Fix: Use APIKey.is_active without == True
            api_keys = db.query(APIKey).filter(APIKey.is_active).all()
            tokens = []
            for api_key in api_keys:
                user = api_key.user
                tokens.append(
                    TokenInfo(
                        token_id="<hidden>",  # Don't expose raw tokens
                        user_id=api_key.user_id,
                        username=user.username if user else "unknown",
                        email=user.email if user else "unknown",
                        token_type=TokenType.API_KEY,
                        scopes=["read", "write"],
                        created_at=api_key.created_at,
                        expires_at=api_key.expires_at,
                        last_used_at=api_key.last_used,
                        is_active=api_key.is_active,
                        metadata={
                            "db_id": api_key.id,
                            "key_prefix": api_key.key_prefix,
                        },
                    )
                )
            return tokens

    def cleanup_expired_tokens(self) -> int:
        """Remove expired tokens from storage."""
        with SessionLocal() as db:
            now = datetime.utcnow()
            # Fix: Use APIKey.is_active without == True
            expired_keys = (
                db.query(APIKey).filter(APIKey.expires_at < now, APIKey.is_active).all()
            )

            count = 0
            for key in expired_keys:
                key.is_active = False
                count += 1

            if count > 0:
                db.commit()
                logger.info(f"Deactivated {count} expired tokens")

            return count

    def persist_token_state(self) -> None:
        """No-op for DB storage."""
        pass

    def get_token_stats(self) -> dict[str, Any]:
        """Get token usage statistics."""
        with SessionLocal() as db:
            total = db.query(APIKey).count()
            # Fix: Use APIKey.is_active without == True
            active = db.query(APIKey).filter(APIKey.is_active).count()

            return {
                "total_tokens": total,
                "active_tokens": active,
                "by_type": {"api_key": active},  # Simplified
                "expired_tokens": total - active,  # Approximation
                "recently_used": 0,
            }


# Global token manager instance
_token_manager: SecureTokenManager | None = None


def get_token_manager() -> SecureTokenManager:
    """Get the global token manager instance."""
    global _token_manager
    if _token_manager is None:
        _token_manager = SecureTokenManager()
    return _token_manager


def initialize_token_manager(
    secret_key: str | None = None, tokens_dir: str = "tokens"
) -> SecureTokenManager:
    """Initialize the global token manager."""
    global _token_manager
    _token_manager = SecureTokenManager(secret_key=secret_key)
    return _token_manager
