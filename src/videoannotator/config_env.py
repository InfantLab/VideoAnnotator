"""Configuration settings for VideoAnnotator.

This module centralizes configuration management with environment variable support.
Values can be overridden via environment variables or .env files.

v1.3.0: Added concurrent job limiting configuration.
"""

import os
from pathlib import Path

from dotenv import dotenv_values, find_dotenv, load_dotenv

from videoannotator.database_location import database_url


def load_env_file() -> None:
    """Load .env into the environment without overriding values already set.

    A variable that is set but blank still takes its .env value: the
    devcontainer forwards host variables as `${localEnv:NAME}`, which defines
    them as "" when the host doesn't set them, and that would otherwise mask
    .env entirely (python-dotenv never overrides an existing variable).
    """
    path = find_dotenv(usecwd=True)
    if not path:
        return
    load_dotenv(path)
    for key, value in dotenv_values(path).items():
        if value and not os.environ.get(key, "").strip():
            os.environ[key] = value


load_env_file()


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

# CORS (Cross-Origin Resource Sharing) Configuration
# Default allows the official web client (video-annotation-viewer) and server's own port
# Server runs on 18011-18020, official client runs on 19011-19020
# Allowing port ranges enables running multiple instances for testing/development
# Also includes common frontend development ports (3000, 5173, 8080, etc.)
DEFAULT_CORS_ORIGINS = ",".join(
    [
        # Server ports
        *[f"http://localhost:{port}" for port in range(18011, 18021)],
        *[f"http://127.0.0.1:{port}" for port in range(18011, 18021)],
        # Client ports
        *[f"http://localhost:{port}" for port in range(19011, 19021)],
        *[f"http://127.0.0.1:{port}" for port in range(19011, 19021)],
        # Common development ports
        "http://localhost:3000",
        "http://127.0.0.1:3000",  # React/Next.js
        "http://localhost:5173",
        "http://127.0.0.1:5173",  # Vite
        "http://localhost:4200",
        "http://127.0.0.1:4200",  # Angular
        "http://localhost:8080",
        "http://127.0.0.1:8080",  # Vue/Generic
        "http://localhost:8000",
        "http://127.0.0.1:8000",  # Common backend
    ]
)

CORS_ORIGINS: str = get_str_env("CORS_ORIGINS", DEFAULT_CORS_ORIGINS)

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

# Serve the bundled Video Annotation Viewer static build at /viewer.
# Disable if you don't want the companion review UI exposed on this server.
ENABLE_VIEWER = get_bool_env("VIDEOANNOTATOR_ENABLE_VIEWER", True)

# Enable CORS credentials support
CORS_ALLOW_CREDENTIALS = get_bool_env("CORS_ALLOW_CREDENTIALS", True)

# Directories the ingest API (POST /api/v1/ingest) may read videos from, as an
# os.pathsep-separated list. Empty means "the server user's home directory",
# which is where a single-user research install keeps its data. Ingest never
# copies these files -- jobs reference them where they are -- so this is the
# boundary of what an admin on this machine can turn into a job.
INGEST_ROOTS = get_str_env("VIDEOANNOTATOR_INGEST_ROOTS", "")

# Where every run's results go (spec 022): one visible folder, by run then
# video. Not ~/Documents, which many Windows installs sync to OneDrive by
# default. Read per call, like the Ollama URL below, so it follows the
# environment the server is running in.
RESULTS_DIR_ENV = "VIDEOANNOTATOR_RESULTS_DIR"

# Docker: set when the port is published on the host's loopback only, so every
# caller that can reach the server is on this machine (spec 022, R6).
PUBLISHED_LOCALLY_ENV = "VIDEOANNOTATOR_PUBLISHED_LOCALLY"

# Docker: `container=host` path prefix pairs, `;`-separated, so locations are
# shown as the researcher's own paths rather than the container's.
HOST_PATHS_ENV = "VIDEOANNOTATOR_HOST_PATHS"


def results_dir() -> Path:
    """`$VIDEOANNOTATOR_RESULTS_DIR`, else `~/VideoAnnotator`, resolved."""
    raw = os.environ.get(RESULTS_DIR_ENV, "").strip()
    path = Path(raw).expanduser() if raw else Path.home() / "VideoAnnotator"
    return path.resolve()


def published_locally() -> bool:
    """Whether every caller counts as being on this machine (Docker, R6)."""
    return get_bool_env(PUBLISHED_LOCALLY_ENV, False)


def host_paths() -> list[tuple[str, str]]:
    """`(container prefix, host prefix)` pairs, longest container prefix first."""
    pairs = []
    for entry in os.environ.get(HOST_PATHS_ENV, "").split(";"):
        container, sep, host = entry.partition("=")
        if sep and container.strip() and host.strip():
            pairs.append((container.strip().rstrip("/"), host.strip().rstrip("/")))
    return sorted(pairs, key=lambda pair: len(pair[0]), reverse=True)


# =============================================================================
# Database Configuration
# =============================================================================

# DATABASE_URL, else SQLite at VIDEOANNOTATOR_DB_PATH or the per-user default
DATABASE_URL = database_url()

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

# Hugging Face token for gated models (pyannote). HUGGINGFACE_TOKEN is the
# documented name; HF_AUTH_TOKEN (the pre-v1.5 name) and HF_TOKEN
# (huggingface_hub's own) are still read, in that order.
HUGGINGFACE_TOKEN_ENV = "HUGGINGFACE_TOKEN"
HUGGINGFACE_TOKEN_ALIASES = ("HF_AUTH_TOKEN", "HF_TOKEN")


def huggingface_token() -> str | None:
    """The configured Hugging Face token, or None."""
    for name in (HUGGINGFACE_TOKEN_ENV, *HUGGINGFACE_TOKEN_ALIASES):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


# Ollama server for the vlm_annotation pipeline when a job doesn't name one.
# In a container with Ollama on the host: http://host.docker.internal:11434
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_BASE_URL_ENV = "OLLAMA_BASE_URL"


def default_ollama_base_url() -> str:
    """`$OLLAMA_BASE_URL`, else `http://127.0.0.1:11434`. Read per call so it
    follows the server's current environment.

    Not `OLLAMA_HOST`: that is Ollama's own *listen* address (commonly
    `0.0.0.0:11434` on the host), not something to connect to.
    """
    return os.environ.get(OLLAMA_BASE_URL_ENV, "").strip() or DEFAULT_OLLAMA_BASE_URL


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
    print(f"  ENABLE_VIEWER: {ENABLE_VIEWER}")
    print(f"  CORS_ORIGINS: {CORS_ORIGINS}")
    print("\nVideos and results:")
    print(f"  VIDEOANNOTATOR_INGEST_ROOTS: {INGEST_ROOTS or '(home folder)'}")
    print(f"  {RESULTS_DIR_ENV}: {results_dir()}")
    print(f"  {PUBLISHED_LOCALLY_ENV}: {published_locally()}")
    print(f"  {HOST_PATHS_ENV}: {host_paths() or '(none)'}")
    print("\nDatabase:")
    print(f"  DATABASE_URL: {DATABASE_URL}")
    print(f"  DB_POOL_SIZE: {DB_POOL_SIZE}")
    print("\nLogging:")
    print(f"  LOG_LEVEL: {LOG_LEVEL}")
    from videoannotator.utils.logging_config import logs_dir

    print(f"  VIDEOANNOTATOR_LOG_DIR: {logs_dir()}")
    print("\nModels:")
    print(f"  MODEL_CACHE_DIR: {MODEL_CACHE_DIR}")
    print(f"  DEVICE: {DEVICE}")
    print(f"  USE_FP16: {USE_FP16}")
    print(f"  OLLAMA_BASE_URL: {default_ollama_base_url()}")
    print("=" * 50)


if __name__ == "__main__":
    print_config()
