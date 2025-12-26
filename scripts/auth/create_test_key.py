#!/usr/bin/env python3
"""Create a new API key for testing purposes.

Run:
  uv run python scripts/auth/create_test_key.py
"""

from __future__ import annotations

from videoannotator.database.crud import APIKeyCRUD
from videoannotator.database.database import SessionLocal, create_tables
from videoannotator.database.models import User


def create_test_api_key() -> str | None:
    """Create a new API key for the admin user."""
    create_tables()
    db = SessionLocal()
    try:
        admin_user = db.query(User).filter(User.username == "admin").first()
        if not admin_user:
            print("[ERROR] No admin user found")
            return None

        api_key_obj, raw_key = APIKeyCRUD.create(
            db=db,
            user_id=str(admin_user.id),
            key_name="test_key",
            expires_days=None,
        )

        print(f"Admin User: {admin_user.username} ({admin_user.email})")
        print(f"New API Key: {raw_key}")
        print(f"Key Prefix: {api_key_obj.key_prefix}")
        print(f"Created: {api_key_obj.created_at}")
        print("")
        print("To test the API, use this as a Bearer token:")
        print(f"Authorization: Bearer {raw_key}")
        print("")
        print("Save this key - it won't be shown again!")

        return raw_key
    finally:
        db.close()


if __name__ == "__main__":
    create_test_api_key()
