#!/usr/bin/env python3
"""VideoAnnotator API server compatibility wrapper.

This repository's primary entrypoint is the `videoannotator` CLI.

This file is kept for backwards compatibility with older docs and workflows.

Usage:
    uv run python api_server.py                       # Start server on port 18011
    uv run python api_server.py --port 18011          # Start on custom port
    uv run uvicorn api_server:app --reload            # Development mode (auto-reload)

Recommended:
    uv run videoannotator server --host 0.0.0.0 --port 18011
"""

import argparse
import sys

import uvicorn

from videoannotator.api.main import create_app
from videoannotator.utils.logging_config import get_logger, setup_videoannotator_logging
from videoannotator.version import __version__


def setup_logging(level: str = "INFO", logs_dir: str = "logs"):
    """Set up enhanced logging configuration."""
    # Capture Python warnings by default so compatibility warnings surface in logs.
    setup_videoannotator_logging(
        logs_dir=logs_dir,
        log_level=level,
        capture_warnings=True,
        capture_stdstreams=(level.upper() == "DEBUG"),
    )


def main():
    """Main entry point for the API server."""
    parser = argparse.ArgumentParser(
        description=f"VideoAnnotator API Server v{__version__}"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=18011, help="Port to bind to (default: 18011)"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes"
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level",
    )
    parser.add_argument(
        "--logs-dir", default="logs", help="Directory for log files (default: logs)"
    )

    args = parser.parse_args()

    # Set up enhanced logging
    setup_logging(args.log_level, args.logs_dir)
    logger = get_logger("api")

    # Create the FastAPI app
    app = create_app()

    # Display startup information
    print("=" * 60)
    print(f"[START] VideoAnnotator API Server v{__version__}")
    print("=" * 60)
    print(f"[INFO] Server starting on http://{args.host}:{args.port}")
    print(f"[INFO] API Documentation: http://{args.host}:{args.port}/docs")
    print(f"[INFO] Health Check: http://{args.host}:{args.port}/health")
    print("[INFO] Database: SQLite (auto-created in current directory)")

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
            workers=args.workers
            if not args.reload
            else 1,  # Reload doesn't work with multiple workers
            log_level=args.log_level,
            access_log=True,
        )
    except KeyboardInterrupt:
        logger.info("[STOP] Server stopped by user")
    except Exception as e:
        logger.error(f"[ERROR] Server failed to start: {e}")
        sys.exit(1)


# Create the FastAPI app instance for direct import.
# This allows: uvicorn api_server:app --reload
app = create_app()

if __name__ == "__main__":
    main()
