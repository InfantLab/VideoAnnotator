"""API dependencies for VideoAnnotator."""

from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from ..auth.token_manager import get_token_manager
from ..database.crud import APIKeyCRUD
from ..database.database import SessionLocal  # Provides DB session factory

# Security scheme for Bearer tokens
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict[str, Any]:
    """Get current user from API token using secure token manager."""
    token = credentials.credentials

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization token is required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Use token manager for validation
    token_manager = get_token_manager()
    token_info = token_manager.validate_token(token)

    if not token_info:
        # Fallback for development tokens
        if token in ["dev-token", "test-token"]:
            return {
                "id": "test-user-123",
                "username": "test_user",
                "email": "test@example.com",
                "is_active": True,
                "scopes": ["read", "write", "debug"],
            }

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {
        "id": token_info.user_id,
        "username": token_info.username,
        "email": token_info.email,
        "is_active": token_info.is_active,
        "scopes": token_info.scopes,
        "token_type": token_info.token_type.value,
        "token_id": token_info.token_id,
    }


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(
        HTTPBearer(auto_error=False)
    ),
) -> dict[str, Any] | None:
    """Get current user if token is provided, otherwise return None.

    Used for endpoints that work with or without authentication.
    """
    if not credentials:
        return None

    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


def _validate_api_key_header(raw: str, db: Session) -> dict[str, Any] | None:
    """Validate API key headers that use the va_ prefix.

    Uses database storage for token validation (v1.3.0+ standard).

    Returns user dict if valid else None.
    """
    if not raw.startswith("va_"):
        return None

    # Validate using database CRUD
    user = APIKeyCRUD.authenticate(db, raw)
    if not user:
        return None

    return {
        "id": str(user.id),
        "username": user.username,
        "email": user.email,
        "is_active": True,
        "scopes": ["read", "write"],
        "token_type": "api_key",
    }


async def validate_optional_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(
        HTTPBearer(auto_error=False)
    ),
) -> dict[str, Any] | None:
    """Validate API key if Authorization header present.

    Behavior:
    - No header: return None (anonymous allowed on some endpoints)
    - Bearer value starts with va_: treat as legacy API key and validate via DB
    - Other tokens: defer to get_current_user for structured token validation
    - Invalid key provided: raise 401 with consistent detail
    """
    if not credentials:
        return None

    token = credentials.credentials
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization token is required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Legacy API key path
    if token.startswith("va_"):
        db = SessionLocal()
        try:
            user = _validate_api_key_header(token, db)
            if user:
                return user
            # If not found in DB, fall through to TokenManager
            # This handles cases where TokenManager generates keys starting with va_
            import logging

            logger = logging.getLogger(__name__)
            logger.info(
                f"Token {token[:10]}... not found in DB, falling back to TokenManager"
            )
        finally:
            db.close()

    # Non-legacy: attempt token manager path
    return await get_current_user(credentials)
