# Authentication Guide

VideoAnnotator v1.3.0+ uses token-based authentication to secure the API. This guide covers obtaining, using, and managing API keys.

## Quick Start

### Getting Your API Key

On first startup, VideoAnnotator automatically generates an API key and prints it to the console:

```
================================================================================
[API KEY] VIDEOANNOTATOR API KEY GENERATED
================================================================================

Your API key: va_api_ya9C5x_OZkweZf8JSxen60WY6KD98_VTa5_uHfN5wRM

Save this key securely - it will NOT be shown again!

Usage:
  curl -H 'Authorization: Bearer va_api_ya9C5x...' http://localhost:8000/api/v1/jobs

To disable authentication (development only):
  export AUTH_REQUIRED=false

To generate additional keys:
  python -m scripts.manage_tokens create
================================================================================
```

**IMPORTANT**: Save this key immediately - it's only shown once!

### Using Your API Key

Set it as an environment variable for easy reuse:

```bash
export API_KEY="va_api_ya9C5x_OZkweZf8JSxen60WY6KD98_VTa5_uHfN5wRM"
```

Then use it in API requests:

```bash
# List jobs
curl -H "Authorization: Bearer $API_KEY" \
  http://localhost:18011/api/v1/jobs

# Submit new job
curl -X POST -H "Authorization: Bearer $API_KEY" \
  -F "video=@video.mp4" \
  -F "selected_pipelines=scene,person,face" \
  http://localhost:18011/api/v1/jobs/

# Get job status
curl -H "Authorization: Bearer $API_KEY" \
  http://localhost:18011/api/v1/jobs/{job_id}
```

### Python Client

```python
import requests

API_KEY = "va_api_ya9C5x_OZkweZf8JSxen60WY6KD98_VTa5_uHfN5wRM"
BASE_URL = "http://localhost:18011/api/v1"

headers = {"Authorization": f"Bearer {API_KEY}"}

# List jobs
response = requests.get(f"{BASE_URL}/jobs", headers=headers)
jobs = response.json()

# Submit job
with open("video.mp4", "rb") as f:
    files = {"video": f}
    data = {"selected_pipelines": "scene,person,face"}
    response = requests.post(
        f"{BASE_URL}/jobs/",
        headers=headers,
        files=files,
        data=data
    )
job_id = response.json()["job_id"]
```

## Advanced Usage

### Retrieving Lost Keys

If you lose your API key, you can retrieve it from the token manager:

```bash
# List all active API keys
uv run python -m scripts.manage_tokens list

# Output shows key IDs and metadata (NOT the full key)
# To get the actual key, you'll need to regenerate
```

**Note**: For security, the full key is never stored - only a hash. If lost, you must generate a new key.

### Generating Additional Keys

Create multiple keys for different users or applications:

```bash
# Interactive key generation
uv run python -m scripts.manage_tokens create

# You'll be prompted for:
# - User ID
# - Username
# - Email
# - Scopes (read, write, admin)
# - Expiration (optional)
```

### Revoking Keys

Revoke compromised or unused keys:

```bash
# List keys to find the key ID
uv run python -m scripts.manage_tokens list

# Revoke specific key
uv run python -m scripts.manage_tokens revoke va_api_xyz...
```

### Key Scopes

API keys support granular permissions:

- **read**: View jobs, results, and system info
- **write**: Submit jobs, update configurations
- **admin**: Manage other users' keys, system settings

```bash
# Generate read-only key
uv run python -m scripts.manage_tokens create --scopes read

# Generate full-access key (default)
uv run python -m scripts.manage_tokens create --scopes read,write,admin
```

## Development & Testing

### Disabling Authentication

For local development, you can disable authentication:

```bash
export AUTH_REQUIRED=false
uv run python api_server.py
```

**⚠️ WARNING**: Only use this for local development. Never deploy with authentication disabled!

### Testing with Auth

Your test suite should use a test API key:

```python
# tests/conftest.py
@pytest.fixture
def api_client():
    """API client with test authentication."""
    from auth.token_manager import get_token_manager

    manager = get_token_manager()
    api_key, _ = manager.generate_api_key(
        user_id="test",
        username="test",
        email="test@example.com",
        scopes=["read", "write", "admin"]
    )

    client = TestClient(app)
    client.headers["Authorization"] = f"Bearer {api_key}"
    return client
```

## Production Deployment

### Security Best Practices

1. **Use HTTPS**: Always deploy behind HTTPS (TLS/SSL)
   ```nginx
   # nginx reverse proxy
   location /api/ {
       proxy_pass http://localhost:18011/;
       proxy_ssl_server_name on;
   }
   ```

2. **Rotate Keys Regularly**: Generate new keys every 90 days
   ```bash
   # Create new key
   uv run python -m scripts.manage_tokens create

   # Update your application config
   # Then revoke old key
   uv run python -m scripts.manage_tokens revoke va_api_old...
   ```

3. **Use Different Keys per Environment**:
   - Development: Local test key
   - Staging: Dedicated staging key
   - Production: Production-only key with limited expiration

4. **Store Keys Securely**:
   - Use environment variables (not hardcoded)
   - Use secrets managers (AWS Secrets Manager, Hashicorp Vault, etc.)
   - Never commit keys to version control

5. **Monitor Key Usage**:
   ```bash
   # Check recent key activity
   tail -f logs/api_requests.log | grep "user_id"
   ```

### Key Expiration

Set expiration for production keys:

```bash
# Generate key that expires in 90 days
uv run python -m scripts.manage_tokens create --expires-in-days 90
```

Expired keys are automatically rejected with a clear error message.

### Multi-User Setups

For lab/team environments:

```bash
# Generate key for each user
uv run python -m scripts.manage_tokens create \
  --user-id alice \
  --username "Alice Johnson" \
  --email alice@lab.edu \
  --scopes read,write

uv run python -m scripts.manage_tokens create \
  --user-id bob \
  --username "Bob Smith" \
  --email bob@lab.edu \
  --scopes read
```

## Public Endpoints

Some endpoints don't require authentication:

- `/health` - Server health check
- `/docs` - API documentation (Swagger UI)
- `/redoc` - Alternative API documentation
- `/openapi.json` - OpenAPI specification

These are safe to expose for monitoring and discovery.

## Troubleshooting

### 401 Unauthorized

**Problem**: API returns `{"error": {"code": "AUTHENTICATION_REQUIRED", ...}}`

**Solutions**:
1. Check API key is set:
   ```bash
   echo $API_KEY  # Should show va_api_...
   ```

2. Check Authorization header format:
   ```bash
   # Correct
   -H "Authorization: Bearer va_api_..."

   # Wrong (old format, no longer supported)
   -H "X-API-Key: va_api_..."
   ```

3. Verify key hasn't expired:
   ```bash
   uv run python -m scripts.manage_tokens list
   ```

### Key Not Working After Server Restart

Keys are persisted in `tokens/tokens.json` (encrypted). If this file is deleted, all keys are lost.

**Solution**: Never delete `tokens/tokens.json`. Back it up with your database.

### Authentication Disabled But Still Getting 401

Check the AUTH_REQUIRED environment variable:

```bash
# Verify it's set to false
echo $AUTH_REQUIRED

# Start server with explicit flag
AUTH_REQUIRED=false uv run python api_server.py
```

### Can't Find Auto-Generated Key

**Problem**: Server started but didn't show the API key.

**Possible Causes**:
1. Key already existed (check `tokens/tokens.json`)
2. Output was hidden by background process
3. AUTO_GENERATE_API_KEY=false was set

**Solution**:
```bash
# Generate new key manually
uv run python -m scripts.manage_tokens create
```

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTH_REQUIRED` | `true` | Enable/disable authentication |
| `AUTO_GENERATE_API_KEY` | `true` | Auto-generate key on first startup |
| `SECURITY_ENABLED` | (alias for AUTH_REQUIRED) | Backward compatibility |

### Token Storage

- **Location**: `tokens/tokens.json`
- **Format**: Encrypted JSON
- **Encryption Key**: `tokens/encryption.key` (auto-generated)
- **Permissions**: 600 (Unix systems)

### Key Format

- **Prefix**: `va_api_`
- **Length**: 47 characters total
- **Encoding**: URL-safe base64
- **Entropy**: 256 bits

Example: `va_api_ya9C5x_OZkweZf8JSxen60WY6KD98_VTa5_uHfN5wRM`

## See Also

- [CORS Configuration](cors.md) - Configure allowed origins
- [Production Checklist](production_checklist.md) - Security hardening guide
- [API Reference](../usage/api_reference.md) - Complete API documentation
