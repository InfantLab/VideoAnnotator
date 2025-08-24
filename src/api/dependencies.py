"""
API dependencies for VideoAnnotator v1.2.0
"""

from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Any, Optional

# Security scheme for Bearer tokens
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """
    Get current user from API token.
    
    This is a simplified implementation for v1.2.0 development.
    In production, this should validate against the database.
    """
    token = credentials.credentials
    
    # For development, accept any non-empty token
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization token is required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Simple token validation for development
    if token in ["dev-token", "test-token"]:
        return {
            "id": "test-user-123",
            "username": "test_user",
            "email": "test@example.com",
            "is_active": True
        }
    
    # TODO: Implement proper token validation against database
    # For now, accept any token as valid
    return {
        "id": f"user-{hash(token) % 1000}",
        "username": "api_user", 
        "email": "user@example.com",
        "is_active": True
    }


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> Optional[Dict[str, Any]]:
    """
    Get current user if token is provided, otherwise return None.
    
    Used for endpoints that work with or without authentication.
    """
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None