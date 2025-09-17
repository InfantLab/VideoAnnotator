"""Standard API error envelope utilities (v1.2.1 minimal).

Provides a lightweight error response structure to establish consistency:
{
  "error": { "code": "PIPELINE_NOT_FOUND", "message": "...", "hint": "..." }
}

Usage:
    raise APIError(status_code=404, code="PIPELINE_NOT_FOUND", message="Pipeline 'x' not found", hint="Run 'videoannotator pipelines --detailed'")

Register the exception handler by calling register_error_handlers(app) in API startup.
(If already imported in src/api/main.py, just add the call.)
"""
from __future__ import annotations
from typing import Optional, Any, Dict
from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

class APIError(Exception):
    def __init__(self, status_code: int, code: str, message: str, hint: Optional[str] = None):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.hint = hint
        super().__init__(message)

class APIErrorModel(BaseModel):
    error: Dict[str, Any]

    class Config:
        extra = "allow"

async def api_error_handler(request: Request, exc: APIError):  # type: ignore[override]
    payload = {"error": {"code": exc.code, "message": exc.message}}
    if exc.hint:
        payload["error"]["hint"] = exc.hint
    return JSONResponse(status_code=exc.status_code, content=payload)

async def generic_http_exception_handler(request: Request, exc):  # fallback for non-APIError HTTPException
    # Preserve FastAPI's status code but wrap structure
    status_code = getattr(exc, "status_code", 500)
    detail = getattr(exc, "detail", "Internal Server Error")
    payload = {"error": {"code": "HTTP_ERROR", "message": str(detail)}}
    return JSONResponse(status_code=status_code, content=payload)

def register_error_handlers(app):
    from fastapi import HTTPException
    app.add_exception_handler(APIError, api_error_handler)
    # Optionally wrap raw HTTPException to maintain envelope
    app.add_exception_handler(HTTPException, generic_http_exception_handler)

__all__ = ["APIError", "register_error_handlers"]
