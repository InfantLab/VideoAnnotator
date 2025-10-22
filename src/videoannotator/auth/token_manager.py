"""Secure Token Management System for VideoAnnotator API v1.2.0.

Provides user-friendly token generation, validation, and management with
security best practices and multiple authentication flows.
"""

import json
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import jwt
from cryptography.fernet import Fernet

from .utils.logging_config import get_logger

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
    - Secure token generation and storage
    - Token expiration and rotation
    - Scope-based permissions
    - User-friendly token management
    """

    def __init__(
        self,
        secret_key: str | None = None,
        tokens_file: str = "tokens/tokens.json",
        encryption_key: bytes | None = None,
    ):
        """Initialize token storage and encryption configuration."""
        self.secret_key = secret_key or self._generate_secret_key()
        self.tokens_file = Path(tokens_file)
        self.tokens_file.parent.mkdir(exist_ok=True)

        # Initialize encryption
        if encryption_key:
            self.cipher = Fernet(encryption_key)
        else:
            key_file = self.tokens_file.parent / "encryption.key"
            self.cipher = self._init_encryption(key_file)

        # In-memory token cache for performance
        self._token_cache: dict[str, TokenInfo] = {}
        self._load_tokens()

    def _generate_secret_key(self) -> str:
        """Generate a secure secret key."""
        return secrets.token_urlsafe(64)

    def _init_encryption(self, key_file: Path) -> Fernet:
        """Initialize encryption with persistent key."""
        if key_file.exists():
            with open(key_file, "rb") as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, "wb") as f:
                f.write(key)
            # Secure the key file on Unix systems
            try:
                key_file.chmod(0o600)
            except OSError:
                pass  # Windows doesn't support chmod

        return Fernet(key)

    def _load_tokens(self) -> None:
        """Load tokens from persistent storage."""
        if not self.tokens_file.exists():
            return

        try:
            with open(self.tokens_file, "rb") as f:
                encrypted_data = f.read()

            if encrypted_data:
                decrypted_data = self.cipher.decrypt(encrypted_data)
                tokens_data = json.loads(decrypted_data.decode())

                for token_id, data in tokens_data.items():
                    token_info = TokenInfo(
                        token_id=data["token_id"],
                        user_id=data["user_id"],
                        username=data["username"],
                        email=data["email"],
                        token_type=TokenType(data["token_type"]),
                        scopes=data["scopes"],
                        created_at=datetime.fromisoformat(data["created_at"]),
                        expires_at=datetime.fromisoformat(data["expires_at"])
                        if data["expires_at"]
                        else None,
                        last_used_at=datetime.fromisoformat(data["last_used_at"])
                        if data["last_used_at"]
                        else None,
                        is_active=data["is_active"],
                        metadata=data["metadata"],
                    )
                    self._token_cache[token_id] = token_info

                logger.info(f"Loaded {len(self._token_cache)} tokens from storage")

        except Exception as e:
            logger.error(f"Failed to load tokens: {e}")

    def _save_tokens(self) -> None:
        """Save tokens to persistent storage."""
        try:
            # Convert to serializable format
            tokens_data = {}
            for token_id, token_info in self._token_cache.items():
                tokens_data[token_id] = {
                    "token_id": token_info.token_id,
                    "user_id": token_info.user_id,
                    "username": token_info.username,
                    "email": token_info.email,
                    "token_type": token_info.token_type.value,
                    "scopes": token_info.scopes,
                    "created_at": token_info.created_at.isoformat(),
                    "expires_at": token_info.expires_at.isoformat()
                    if token_info.expires_at
                    else None,
                    "last_used_at": token_info.last_used_at.isoformat()
                    if token_info.last_used_at
                    else None,
                    "is_active": token_info.is_active,
                    "metadata": token_info.metadata,
                }

            # Encrypt and save
            data_json = json.dumps(tokens_data, indent=2)
            encrypted_data = self.cipher.encrypt(data_json.encode())

            with open(self.tokens_file, "wb") as f:
                f.write(encrypted_data)

            logger.debug(f"Saved {len(tokens_data)} tokens to storage")

        except Exception as e:
            logger.error(f"Failed to save tokens: {e}")

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
        token_id = f"va_api_{secrets.token_urlsafe(32)}"

        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)

        token_info = TokenInfo(
            token_id=token_id,
            user_id=user_id,
            username=username,
            email=email,
            token_type=TokenType.API_KEY,
            scopes=scopes,
            created_at=datetime.now(),
            expires_at=expires_at,
            last_used_at=None,
            is_active=True,
            metadata={
                **metadata,
                "generated_by": "token_manager",
                "client_info": "VideoAnnotator API",
            },
        )

        # Store token
        self._token_cache[token_id] = token_info
        self._save_tokens()

        logger.info(f"Generated API key for user {username} (expires: {expires_at})")

        return token_id, token_info

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

        # Store for tracking (but don't persist JWT tokens)
        self._token_cache[token_id] = token_info

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

            # API key validation
            if token in self._token_cache:
                token_info = self._token_cache[token]

                # Check if token is active
                if not token_info.is_active:
                    logger.warning(f"Inactive token used: {token[:16]}...")
                    return None

                # Check expiration
                if token_info.expires_at and datetime.now() > token_info.expires_at:
                    logger.warning(f"Expired token used: {token[:16]}...")
                    self.revoke_token(token)
                    return None

                # Update last used time
                token_info.last_used_at = datetime.now()
                self._save_tokens()

                logger.debug(f"Valid API key used by {token_info.username}")
                return token_info

            logger.warning(f"Unknown token format: {token[:16]}...")
            return None

        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return None

    def _validate_jwt_token(self, token: str) -> TokenInfo | None:
        """Validate JWT session token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])

            token_id = payload["token_id"]

            # Check if token is in cache and active
            if token_id in self._token_cache:
                token_info = self._token_cache[token_id]
                if not token_info.is_active:
                    return None

                # Update last used
                token_info.last_used_at = datetime.now()
                logger.debug(f"Valid JWT token used by {token_info.username}")
                return token_info

            # Create token info from JWT payload (for stateless operation)
            token_info = TokenInfo(
                token_id=token_id,
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
            logger.warning(f"Invalid JWT token: {e}")
            return None

    def revoke_token(self, token: str) -> bool:
        """Revoke a token."""
        try:
            if token.startswith("va_") and token in self._token_cache:
                self._token_cache[token].is_active = False
                self._save_tokens()
                logger.info(f"Revoked API key: {token[:16]}...")
                return True

            # For JWT tokens, try to extract token_id
            try:
                payload = jwt.decode(
                    token,
                    self.secret_key,
                    algorithms=["HS256"],
                    options={"verify_exp": False},
                )
                token_id = payload["token_id"]
                if token_id in self._token_cache:
                    self._token_cache[token_id].is_active = False
                    logger.info(f"Revoked JWT token: {token_id}")
                    return True
            except:
                pass

            return False

        except Exception as e:
            logger.error(f"Failed to revoke token: {e}")
            return False

    def list_user_tokens(self, user_id: str) -> list[TokenInfo]:
        """List all tokens for a user."""
        return [
            token_info
            for token_info in self._token_cache.values()
            if token_info.user_id == user_id and token_info.is_active
        ]

    def cleanup_expired_tokens(self) -> int:
        """Remove expired tokens from storage."""
        count = 0
        now = datetime.now()

        expired_tokens = []
        for token_id, token_info in self._token_cache.items():
            if token_info.expires_at and now > token_info.expires_at:
                expired_tokens.append(token_id)

        for token_id in expired_tokens:
            del self._token_cache[token_id]
            count += 1

        if count > 0:
            self._save_tokens()
            logger.info(f"Cleaned up {count} expired tokens")

        return count

    def get_token_stats(self) -> dict[str, Any]:
        """Get token usage statistics."""
        now = datetime.now()
        stats = {
            "total_tokens": len(self._token_cache),
            "active_tokens": sum(1 for t in self._token_cache.values() if t.is_active),
            "by_type": {},
            "expired_tokens": 0,
            "recently_used": 0,  # Used in last 24 hours
        }

        for token_info in self._token_cache.values():
            # Count by type
            token_type = token_info.token_type.value
            stats["by_type"][token_type] = stats["by_type"].get(token_type, 0) + 1  # type: ignore[index,attr-defined]

            # Count expired
            if token_info.expires_at and now > token_info.expires_at:
                stats["expired_tokens"] += 1  # type: ignore[operator]

            # Count recently used
            if token_info.last_used_at and (now - token_info.last_used_at).days < 1:
                stats["recently_used"] += 1  # type: ignore[operator]

        return stats


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
    _token_manager = SecureTokenManager(
        secret_key=secret_key, tokens_file=f"{tokens_dir}/tokens.json"
    )
    return _token_manager
