"""Configuration settings for VideoAnnotator.

This module centralizes configuration management with environment variable support.
Values can be overridden via environment variables or .env files.

v1.3.0: Added concurrent job limiting configuration.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()


def get_int_env(key: str, default: int) -> int:
    """Get integer from environment variable with fallback to default.

    Args:
        key: Environment variable name
        default: Default value if not set or invalid

    Returns:
        Integer value from environment or default
    """
    try:
        value = os.getenv(key)
        return int(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def get_bool_env(key: str, default: bool) -> bool:
    """Get boolean from environment variable with fallback to default.

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        Boolean value from environment or default
    """
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes", "on")


def get_str_env(key: str, default: str) -> str:
    """Get string from environment variable with fallback to default.

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        String value from environment or default
    """
    return os.getenv(key, default)


# =============================================================================
# Worker Configuration
# =============================================================================

# Maximum number of jobs to process concurrently
# Lower values reduce GPU memory pressure but decrease throughput
# Recommendation: 1-2 for 6GB GPU, 2-4 for 8-12GB GPU
MAX_CONCURRENT_JOBS = get_int_env("MAX_CONCURRENT_JOBS", 2)

# Poll interval for checking new jobs (seconds)
WORKER_POLL_INTERVAL = get_int_env("WORKER_POLL_INTERVAL", 5)

# Maximum retry attempts for failed jobs
MAX_JOB_RETRIES = get_int_env("MAX_JOB_RETRIES", 3)

# Base delay for exponential backoff (seconds)
RETRY_DELAY_BASE = float(os.getenv("RETRY_DELAY_BASE", "2.0"))


# =============================================================================
# Storage Configuration
# =============================================================================

# Base directory for job storage (results, logs, temp files)
STORAGE_BASE_DIR = Path(get_str_env("STORAGE_BASE_DIR", "./batch_results"))

# Days to retain completed job data (null = never delete)
# Only applies to terminal states (COMPLETED, FAILED, CANCELLED)
STORAGE_RETENTION_DAYS = int(d) if (d := os.getenv("STORAGE_RETENTION_DAYS")) else None


# =============================================================================
# Security Configuration
# =============================================================================

# Require API key authentication
AUTH_REQUIRED = get_bool_env("AUTH_REQUIRED", True)

# Path to token storage directory
TOKEN_DIR = Path(get_str_env("TOKEN_DIR", "./tokens"))

# Auto-generate API key on first startup if none exist
AUTO_GENERATE_KEY = get_bool_env("AUTO_GENERATE_KEY", True)


# =============================================================================
# API Server Configuration
# =============================================================================

# Host to bind to
API_HOST = get_str_env("API_HOST", "0.0.0.0")

# Port to listen on
API_PORT = get_int_env("API_PORT", 18011)

# CORS allowed origins (comma-separated)
# Default: localhost only (secure-by-default)
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS", "http://localhost,http://localhost:18011"
).split(",")

# Enable CORS credentials support
CORS_ALLOW_CREDENTIALS = get_bool_env("CORS_ALLOW_CREDENTIALS", True)


# =============================================================================
# Database Configuration
# =============================================================================

# Database URL (defaults to SQLite)
DATABASE_URL = get_str_env("DATABASE_URL", "sqlite:///./videoannotator.db")

# Enable database connection pool
DB_POOL_ENABLED = get_bool_env("DB_POOL_ENABLED", True)

# Pool size for database connections
DB_POOL_SIZE = get_int_env("DB_POOL_SIZE", 5)


# =============================================================================
# Logging Configuration
# =============================================================================

# Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL = get_str_env("LOG_LEVEL", "INFO")

# Log directory
LOG_DIR = Path(get_str_env("LOG_DIR", "./logs"))

# Enable structured JSON logging
LOG_JSON = get_bool_env("LOG_JSON", False)


# =============================================================================
# Model Configuration
# =============================================================================

# Cache directory for downloaded models
MODEL_CACHE_DIR = Path(get_str_env("MODEL_CACHE_DIR", "./models"))

# Device to use for inference (cpu, cuda, auto)
DEVICE = get_str_env("DEVICE", "auto")

# Use FP16 precision when available
USE_FP16 = get_bool_env("USE_FP16", True)


def print_config() -> None:
    """Print current configuration (for debugging)."""
    print("VideoAnnotator Configuration")
    print("=" * 50)
    print("Worker:")
    print(f"  MAX_CONCURRENT_JOBS: {MAX_CONCURRENT_JOBS}")
    print(f"  WORKER_POLL_INTERVAL: {WORKER_POLL_INTERVAL}s")
    print(f"  MAX_JOB_RETRIES: {MAX_JOB_RETRIES}")
    print(f"  RETRY_DELAY_BASE: {RETRY_DELAY_BASE}s")
    print("\nStorage:")
    print(f"  STORAGE_BASE_DIR: {STORAGE_BASE_DIR}")
    print(f"  STORAGE_RETENTION_DAYS: {STORAGE_RETENTION_DAYS}")
    print("\nSecurity:")
    print(f"  AUTH_REQUIRED: {AUTH_REQUIRED}")
    print(f"  TOKEN_DIR: {TOKEN_DIR}")
    print(f"  AUTO_GENERATE_KEY: {AUTO_GENERATE_KEY}")
    print("\nAPI Server:")
    print(f"  API_HOST: {API_HOST}")
    print(f"  API_PORT: {API_PORT}")
    print(f"  CORS_ORIGINS: {', '.join(CORS_ORIGINS)}")
    print("\nDatabase:")
    print(f"  DATABASE_URL: {DATABASE_URL}")
    print(f"  DB_POOL_SIZE: {DB_POOL_SIZE}")
    print("\nLogging:")
    print(f"  LOG_LEVEL: {LOG_LEVEL}")
    print(f"  LOG_DIR: {LOG_DIR}")
    print("\nModels:")
    print(f"  MODEL_CACHE_DIR: {MODEL_CACHE_DIR}")
    print(f"  DEVICE: {DEVICE}")
    print(f"  USE_FP16: {USE_FP16}")
    print("=" * 50)


if __name__ == "__main__":
    print_config()
