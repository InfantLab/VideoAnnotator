"""Authentication middleware and configuration for VideoAnnotator API.

Provides secure-by-default authentication with configurable behavior:
- AUTH_REQUIRED=true (default): All endpoints require valid API key
- AUTH_REQUIRED=false: Endpoints accept optional API keys (development mode)

Environment Variables:
    AUTH_REQUIRED: Enable/disable authentication requirement (default: true)
    SECURITY_ENABLED: Alias for AUTH_REQUIRED (for backward compatibility)
"""

import os
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..dependencies import validate_optional_api_key


def is_auth_required() -> bool:
    """Check if authentication is required based on environment variables.

    Returns:
        True if authentication is required, False otherwise

    Environment Variables:
        AUTH_REQUIRED: Primary configuration (true/false, default: true)
        SECURITY_ENABLED: Fallback for backward compatibility

    Examples:
        >>> os.environ['AUTH_REQUIRED'] = 'false'
        >>> is_auth_required()
        False

        >>> os.environ['AUTH_REQUIRED'] = 'true'
        >>> is_auth_required()
        True
    """
    # Check AUTH_REQUIRED first (primary)
    auth_required = os.environ.get("AUTH_REQUIRED", "").lower()
    if auth_required in ("true", "1", "yes"):
        return True
    if auth_required in ("false", "0", "no"):
        return False

    # Check SECURITY_ENABLED as fallback
    security_enabled = os.environ.get("SECURITY_ENABLED", "").lower()
    if security_enabled in ("true", "1", "yes"):
        return True

    # Default: authentication REQUIRED (secure by default) unless explicitly disabled
    return security_enabled not in ("false", "0", "no")


async def validate_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(
        HTTPBearer(auto_error=False)
    ),
) -> dict[str, Any] | None:
    """Validate API key with configurable requirement.

    Behavior depends on AUTH_REQUIRED environment variable:
    - AUTH_REQUIRED=true (default): Authentication required, returns user dict or raises 401
    - AUTH_REQUIRED=false: Authentication optional, returns user dict or None

    Args:
        credentials: HTTP Authorization header credentials

    Returns:
        User dictionary if authenticated, None if optional and not provided

    Raises:
        HTTPException: 401 if AUTH_REQUIRED=true and credentials invalid/missing

    Examples:
        # Required mode (default)
        >>> os.environ['AUTH_REQUIRED'] = 'true'
        >>> await validate_api_key(None)  # Raises 401

        # Optional mode (development)
        >>> os.environ['AUTH_REQUIRED'] = 'false'
        >>> await validate_api_key(None)  # Returns None
    """
    auth_required = is_auth_required()

    # Get user from credentials if provided
    user = await validate_optional_api_key(credentials)

    # If authentication is required and no valid user, reject
    if auth_required and user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Set AUTH_REQUIRED=false to disable authentication.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


async def validate_required_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(
        HTTPBearer(auto_error=False)
    ),
) -> dict[str, Any]:
    """Validate API key - always required regardless of AUTH_REQUIRED setting.

    Use this dependency for endpoints that must always require authentication,
    even when AUTH_REQUIRED=false (e.g., admin endpoints, sensitive operations).

    Args:
        credentials: HTTP Authorization header credentials

    Returns:
        User dictionary

    Raises:
        HTTPException: 401 if credentials invalid or missing

    Examples:
        >>> await validate_required_api_key(None)  # Always raises 401
    """
    user = await validate_optional_api_key(credentials)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required for this endpoint",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


__all__ = [
    "is_auth_required",
    "validate_api_key",
    "validate_required_api_key",
]
