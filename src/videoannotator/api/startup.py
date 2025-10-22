"""Startup utilities for VideoAnnotator API.

Handles first-run initialization including automatic API key generation.
"""

import os
from pathlib import Path

from .auth.token_manager import get_token_manager
from .utils.logging_config import get_logger

logger = get_logger("api")


def ensure_api_key_exists() -> tuple[str | None, bool]:
    """Ensure at least one API key exists, generating one if needed.

    Checks for existing tokens and generates a new API key on first startup
    if none exist. Prints the key to console for user access.

    Returns:
        Tuple of (api_key, is_new_key)
        - api_key: The generated API key if new, None if keys already exist
        - is_new_key: True if a new key was generated, False if keys exist

    Environment Variables:
        AUTO_GENERATE_API_KEY: Enable/disable auto-generation (default: true)

    Examples:
        >>> api_key, is_new = ensure_api_key_exists()
        >>> if is_new:
        ...     print(f"New API key: {api_key}")
    """
    # Check if auto-generation is disabled
    auto_generate = os.environ.get("AUTO_GENERATE_API_KEY", "true").lower()
    if auto_generate in ("false", "0", "no"):
        logger.debug("Auto API key generation disabled via AUTO_GENERATE_API_KEY")
        return None, False

    # Get token manager
    token_manager = get_token_manager()

    # Check if tokens file exists and has content
    tokens_file = Path("tokens/tokens.json")
    if tokens_file.exists() and tokens_file.stat().st_size > 0:
        # Tokens exist, count active API keys
        try:
            active_api_keys = [
                t
                for t in token_manager._token_cache.values()
                if t.token_type.value == "api_key" and t.is_active
            ]
            if active_api_keys:
                logger.info(f"Found {len(active_api_keys)} existing API key(s)")
                return None, False
        except Exception as e:
            logger.warning(f"Could not check existing tokens: {e}")

    # No active API keys found - generate one
    logger.info("[STARTUP] No API keys found - generating first API key")

    # Generate a strong API key
    user_id = "admin"
    username = "admin"
    email = "admin@localhost"
    scopes = ["read", "write", "admin"]

    try:
        api_key, _token_info = token_manager.generate_api_key(
            user_id=user_id,
            username=username,
            email=email,
            scopes=scopes,
            expires_in_days=None,  # Never expires
            metadata={
                "auto_generated": True,
                "purpose": "First startup initialization",
            },
        )

        # Print to console with clear instructions
        print("\n" + "=" * 80)
        print("[API KEY] VIDEOANNOTATOR API KEY GENERATED")
        print("=" * 80)
        print(f"\nYour API key: {api_key}")
        print("\nSave this key securely - it will NOT be shown again!")
        print("\nUsage:")
        print(
            f"  curl -H 'Authorization: Bearer {api_key}' http://localhost:8000/api/v1/jobs"
        )
        print("\nTo disable authentication (development only):")
        print("  export AUTH_REQUIRED=false")
        print("\nTo generate additional keys:")
        print("  python -m scripts.manage_tokens create")
        print("=" * 80 + "\n")

        logger.info(f"Generated first API key for user '{username}'")
        return api_key, True

    except Exception as e:
        logger.error(f"Failed to generate API key: {e}")
        print("\n[ERROR] Failed to generate API key - check server logs")
        print(
            "You may need to disable authentication with: export AUTH_REQUIRED=false\n"
        )
        return None, False


def initialize_security() -> None:
    """Initialize security settings on startup.

    Performs all security-related startup tasks:
    - Ensures API keys exist (generates if needed)
    - Validates security configuration
    - Logs security status
    """
    # Ensure API key exists
    _api_key, is_new = ensure_api_key_exists()  # Check authentication status
    from api.middleware.auth import is_auth_required

    auth_required = is_auth_required()
    if auth_required:
        if is_new:
            logger.info("[SECURITY] Authentication REQUIRED - API key generated above")
        else:
            logger.info("[SECURITY] Authentication REQUIRED - using existing API keys")
    else:
        logger.warning(
            "[SECURITY] Authentication DISABLED - set AUTH_REQUIRED=true for production"
        )

    # Log CORS configuration
    cors_origins = os.environ.get("CORS_ORIGINS", "http://localhost:3000")
    logger.info(f"[SECURITY] CORS allowed origins: {cors_origins}")


__all__ = ["ensure_api_key_exists", "initialize_security"]
