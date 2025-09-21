#!/usr/bin/env python3
"""
VideoAnnotator API Server v1.2.0

This is the standalone API server that can be run independently to serve
the VideoAnnotator REST API with database persistence.

Usage:
    python api_server.py                    # Start server on port 18011
    python api_server.py --port 18011       # Start on custom port
    uvicorn api_server:app --reload         # Development mode with auto-reload
    
    # Or use the CLI (recommended):
    uv run python -m src.cli server         # Uses same underlying API
"""

import argparse
import logging
from pathlib import Path
import sys

import uvicorn

# Import the main API application
from src.api.main import create_app
from src.utils.logging_config import setup_videoannotator_logging, get_logger

def setup_logging(level: str = "INFO", logs_dir: str = "logs"):
    """Set up enhanced logging configuration."""
    setup_videoannotator_logging(logs_dir=logs_dir, log_level=level)

def main():
    """Main entry point for the API server."""
    parser = argparse.ArgumentParser(description="VideoAnnotator API Server v1.2.0")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=18011, help="Port to bind to (default: 18011)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"], help="Log level")
    parser.add_argument("--logs-dir", default="logs", help="Directory for log files (default: logs)")
    
    args = parser.parse_args()
    
    # Set up enhanced logging
    setup_logging(args.log_level, args.logs_dir)
    logger = get_logger("api")
    
    # Create the FastAPI app
    app = create_app()
    
    # Display startup information
    print("=" * 60)
    print("[START] VideoAnnotator API Server v1.2.0")
    print("=" * 60)
    print(f"[INFO] Server starting on http://{args.host}:{args.port}")
    print(f"[INFO] API Documentation: http://{args.host}:{args.port}/docs")
    print(f"[INFO] Health Check: http://{args.host}:{args.port}/health")
    print(f"[INFO] Database: SQLite (auto-created in current directory)")
    
    if args.reload:
        print("[INFO] Development mode: Auto-reload enabled")
    
    print("=" * 60)
    
    # Start the server
    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,  # Reload doesn't work with multiple workers
            log_level=args.log_level,
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("[STOP] Server stopped by user")
    except Exception as e:
        logger.error(f"[ERROR] Server failed to start: {e}")
        sys.exit(1)

# Create the FastAPI app instance for direct import
# This allows: uvicorn api_server:app --reload
app = create_app()

if __name__ == "__main__":
    main()