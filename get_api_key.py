#!/usr/bin/env python3
"""
Simple script to retrieve the admin API key for testing purposes.
"""

from src.database.database import SessionLocal
from src.database.models import User, APIKey

def get_admin_api_key():
    """Get the admin user's API key."""
    db = SessionLocal()
    try:
        # Find admin user
        admin_user = db.query(User).filter(User.username == "admin").first()
        if not admin_user:
            print("❌ No admin user found")
            return None
        
        # Get first active API key for admin
        api_key = db.query(APIKey).filter(
            APIKey.user_id == admin_user.id,
            APIKey.is_active == True
        ).first()
        
        if not api_key:
            print("❌ No active API key found for admin")
            return None
        
        print(f"Admin User: {admin_user.username} ({admin_user.email})")
        print(f"API Key Prefix: va_{api_key.key_prefix}")
        print(f"Created: {api_key.created_at}")
        print(f"Last Used: {api_key.last_used or 'Never'}")
        print("\n🔑 To test the API, use this as a Bearer token:")
        print("   Authorization: Bearer va_<FULL_KEY>")
        print("\n⚠️  The full API key is not stored in database (only hash)")
        print("   Use the key from migration output or create a new one")
        
        return api_key
        
    finally:
        db.close()

if __name__ == "__main__":
    get_admin_api_key()