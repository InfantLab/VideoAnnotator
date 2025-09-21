#!/usr/bin/env python3
"""
Create a new API key for testing purposes.
"""

from src.database.database import SessionLocal
from src.database.models import User, APIKey
from src.database.crud import UserCRUD, APIKeyCRUD

def create_test_api_key():
    """Create a new API key for the admin user."""
    db = SessionLocal()
    try:
        # Find admin user
        admin_user = db.query(User).filter(User.username == "admin").first()
        if not admin_user:
            print("No admin user found")
            return None
        
        # Create new API key
        api_key_obj, raw_key = APIKeyCRUD.create(
            db=db,
            user_id=str(admin_user.id),
            key_name="test_key",
            expires_days=None  # Never expires
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