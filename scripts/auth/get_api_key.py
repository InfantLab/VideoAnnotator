#!/usr/bin/env python3
"""Retrieve the admin API key metadata for testing.

Note: the full API key is not stored (only a hash). This script helps you find
the admin user and any active API key prefixes.

Run:
  uv run python scripts/auth/get_api_key.py
"""

from __future__ import annotations

from videoannotator.database.database import SessionLocal, create_tables
from videoannotator.database.models import APIKey, User


def get_admin_api_key() -> APIKey | None:
    """Get the admin user's first active API key (metadata only)."""
    create_tables()
    db = SessionLocal()
    try:
        admin_user = db.query(User).filter(User.username == "admin").first()
        if not admin_user:
            print("[ERROR] No admin user found")
            return None

        api_key = (
            db.query(APIKey)
            .filter(APIKey.user_id == admin_user.id, APIKey.is_active)
            .first()
        )

        if not api_key:
            print("[ERROR] No active API key found for admin")
            return None

        print(f"Admin User: {admin_user.username} ({admin_user.email})")
        print(f"API Key Prefix: va_{api_key.key_prefix}")
        print(f"Created: {api_key.created_at}")
        print(f"Last Used: {api_key.last_used or 'Never'}")
        print("")
        print("To test the API, use this as a Bearer token:")
        print("Authorization: Bearer va_<FULL_KEY>")
        print("")
        print("[WARNING] The full API key is not stored in the database (only a hash)")
        print("          Use the key from creation output or create a new one")

        return api_key
    finally:
        db.close()


if __name__ == "__main__":
    get_admin_api_key()
