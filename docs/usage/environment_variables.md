# Environment Variable Configuration

**Version**: v1.3.0+
**Status**: Production Ready

VideoAnnotator supports comprehensive configuration via environment variables, allowing you to customize behavior without modifying code or config files.

## Quick Start

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and set desired values

3. Start VideoAnnotator (environment variables are automatically loaded):
   ```bash
   videoannotator server
   # or
   videoannotator worker
   ```

## Configuration Categories

### Worker Configuration

Control concurrent job processing and retry behavior:

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CONCURRENT_JOBS` | `2` | Maximum jobs to process simultaneously |
| `WORKER_POLL_INTERVAL` | `5` | Seconds between database polls for new jobs |
| `MAX_JOB_RETRIES` | `3` | Maximum retry attempts for failed jobs |
| `RETRY_DELAY_BASE` | `2.0` | Base delay (seconds) for exponential backoff |

**Tuning `MAX_CONCURRENT_JOBS`**:
- **1-2**: 6GB GPU (safe for most models)
- **2-4**: 8-12GB GPU (balanced performance)
- **4+**: 24GB+ GPU or CPU-only workloads
- **Note**: Higher values increase throughput but risk OOM errors

### Storage Configuration

Control where job data is stored and how long it's retained:

| Variable | Default | Description |
|----------|---------|-------------|
| `STORAGE_BASE_DIR` | `./batch_results` | Base directory for job outputs |
| `STORAGE_RETENTION_DAYS` | _(none)_ | Days to retain completed jobs (empty = never delete) |

### Security Configuration

Control authentication and token management:

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTH_REQUIRED` | `true` | Require API key authentication |
| `TOKEN_DIR` | `./tokens` | Directory for API token storage |
| `AUTO_GENERATE_KEY` | `true` | Auto-generate key on first startup |

### API Server Configuration

Control network binding and CORS:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | Host to bind to (`0.0.0.0` = all interfaces) |
| `API_PORT` | `18011` | Port to listen on |
| `CORS_ORIGINS` | `http://localhost,...` | Comma-separated allowed origins |
| `CORS_ALLOW_CREDENTIALS` | `true` | Enable CORS credentials |

### Database Configuration

Control database connection:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///./videoannotator.db` | Database connection string |
| `DB_POOL_ENABLED` | `true` | Enable connection pooling |
| `DB_POOL_SIZE` | `5` | Connection pool size |

**Database URL Examples**:
```bash
# SQLite (default)
DATABASE_URL=sqlite:///./videoannotator.db

# PostgreSQL
DATABASE_URL=postgresql://user:password@localhost/dbname

# MySQL
DATABASE_URL=mysql://user:password@localhost/dbname
```

### Logging Configuration

Control log output and verbosity:

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `LOG_DIR` | `./logs` | Directory for log files |
| `LOG_JSON` | `false` | Enable structured JSON logging |

### Model Configuration

Control model downloads and inference:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_CACHE_DIR` | `./models` | Directory for downloaded models |
| `DEVICE` | `auto` | Inference device (`cpu`, `cuda`, `auto`) |
| `USE_FP16` | `true` | Use half-precision (reduces memory) |

## Usage Examples

### Example 1: High-Performance Server

For a 24GB GPU with fast storage:

```bash
# .env
MAX_CONCURRENT_JOBS=8
WORKER_POLL_INTERVAL=2
LOG_LEVEL=WARNING
USE_FP16=true
DEVICE=cuda
```

### Example 2: Memory-Constrained Environment

For a 6GB GPU or shared system:

```bash
# .env
MAX_CONCURRENT_JOBS=1
WORKER_POLL_INTERVAL=10
USE_FP16=true
MAX_JOB_RETRIES=5
```

### Example 3: Development/Testing

For local development with verbose logging:

```bash
# .env
MAX_CONCURRENT_JOBS=1
LOG_LEVEL=DEBUG
AUTH_REQUIRED=false
API_HOST=127.0.0.1
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
```

### Example 4: Production Deployment

For production with PostgreSQL and retention policy:

```bash
# .env
MAX_CONCURRENT_JOBS=4
WORKER_POLL_INTERVAL=3
STORAGE_RETENTION_DAYS=30
DATABASE_URL=postgresql://videoannotator:password@db.example.com/prod
AUTH_REQUIRED=true
AUTO_GENERATE_KEY=false
LOG_LEVEL=INFO
LOG_JSON=true
API_HOST=0.0.0.0
API_PORT=18011
```

## Command-Line Override

Environment variables can be overridden via command-line options:

```bash
# Environment variable sets default
export MAX_CONCURRENT_JOBS=2

# CLI option overrides environment
videoannotator worker --max-concurrent 4

# Result: uses 4 concurrent jobs
```

## Programmatic Access

Access configuration in Python code:

```python
from videoannotator.config_env import (
    MAX_CONCURRENT_JOBS,
    WORKER_POLL_INTERVAL,
    print_config,
)

print(f"Max concurrent jobs: {MAX_CONCURRENT_JOBS}")

# Print all configuration
print_config()
```

## Viewing Current Configuration

Print the active configuration:

```bash
uv run python -m videoannotator.config_env
```

Output:
```
VideoAnnotator Configuration
==================================================
Worker:
  MAX_CONCURRENT_JOBS: 2
  WORKER_POLL_INTERVAL: 5s
  MAX_JOB_RETRIES: 3
  ...
```

## Migration from v1.2.x

In v1.2.x, concurrency was only configurable via CLI arguments. Now it can be set three ways (in priority order):

1. **CLI argument** (highest priority): `--max-concurrent 4`
2. **Environment variable**: `MAX_CONCURRENT_JOBS=4`
3. **Default value** (lowest priority): `2`

**No breaking changes**: Existing CLI usage continues to work unchanged.

## Best Practices

1. **Use `.env` for local development**: Store per-machine settings
2. **Use system env vars for production**: Set via systemd, Docker, etc.
3. **Never commit `.env` files**: Add to `.gitignore`
4. **Document custom values**: Comment why non-default values are needed
5. **Monitor GPU memory**: Tune `MAX_CONCURRENT_JOBS` based on actual usage
6. **Enable JSON logs for production**: Easier to parse and aggregate
7. **Use retention policy for large deployments**: Prevent disk exhaustion

## Troubleshooting

### Environment variables not taking effect

**Issue**: Changed `.env` but values unchanged

**Solution**: Ensure `.env` is in working directory where you run commands:
```bash
# Check working directory
pwd

# Verify .env exists
ls -la .env

# Test env var directly
MAX_CONCURRENT_JOBS=5 uv run python -m videoannotator.config_env
```

### OOM (Out of Memory) errors

**Issue**: GPU runs out of memory during processing

**Solution**: Reduce `MAX_CONCURRENT_JOBS`:
```bash
MAX_CONCURRENT_JOBS=1
```

### Jobs not being picked up

**Issue**: Submitted jobs stay in PENDING state

**Solution**: Ensure worker is running and polling:
```bash
# Check worker logs
tail -f logs/videoannotator.log

# Try reducing poll interval
WORKER_POLL_INTERVAL=2 videoannotator worker
```

## See Also

- [Deployment Guide](./deployment/README.md)
- [Configuration Files](../configs/README.md)
- [Worker Architecture](./development/architecture.md)
