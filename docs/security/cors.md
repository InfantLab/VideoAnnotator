# CORS Configuration Guide

Cross-Origin Resource Sharing (CORS) controls which web origins can access your VideoAnnotator API. Proper CORS configuration is critical for security.

## Quick Start

### Default Configuration

VideoAnnotator is configured for the official client by default:

```bash
# Default: Allows official video-annotation-viewer (19011) + server (18011)
# No configuration needed for standard usage
uv run videoannotator
```

Console output will show:
```
[SECURITY] CORS: Allowing official client (port 19011) and server (port 18011)
```

The default origins are:
- **Port 19011**: video-annotation-viewer (official web client)
- **Port 18011**: VideoAnnotator server (same-origin requests)

**For custom clients or testing**, use development mode:
```bash
uv run videoannotator --dev
# Allows ALL origins (*), disables authentication
```

### Testing CORS

```bash
# Test official client port (19011) - should succeed
curl -H "Origin: http://localhost:19011" \
     -H "Access-Control-Request-Method: POST" \
     -X OPTIONS http://localhost:18011/api/v1/jobs

# Response includes:
# access-control-allow-origin: http://localhost:19011
# access-control-allow-methods: POST, GET, OPTIONS, ...

# Test server port (18011) - should succeed
curl -H "Origin: http://localhost:18011" \
     -H "Access-Control-Request-Method: POST" \
     -X OPTIONS http://localhost:18011/api/v1/jobs

# Disallowed origin - should be rejected
curl -H "Origin: http://unauthorized-site.com" \
     -H "Access-Control-Request-Method: POST" \
     -X OPTIONS http://localhost:18011/api/v1/jobs

# Response will NOT include access-control-allow-origin header
# Browser will block the request
```

## Configuration

### Single Custom Origin

Set the `CORS_ORIGINS` environment variable:

```bash
# Allow your web app's origin
export CORS_ORIGINS="https://app.yourdomain.com"
uv run videoannotator
```

### Multiple Origins

Provide a comma-separated list:

```bash
# Allow multiple web apps
export CORS_ORIGINS="https://app.yourdomain.com,https://admin.yourdomain.com,https://staging.yourdomain.com"
uv run videoannotator
```

Whitespace is automatically stripped, so this also works:

```bash
export CORS_ORIGINS="https://app.yourdomain.com, https://admin.yourdomain.com, https://staging.yourdomain.com"
```

### Docker Deployment

```dockerfile
# docker-compose.yml
services:
  videoannotator:
    image: videoannotator:latest
    environment:
      - CORS_ORIGINS=https://app.example.com,https://admin.example.com
    ports:
      - "18011:18011"
```

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: videoannotator
spec:
  template:
    spec:
      containers:
      - name: videoannotator
        image: videoannotator:latest
        env:
        - name: CORS_ORIGINS
          value: "https://app.example.com,https://admin.example.com"
```

## Security Best Practices

### ⚠️ Never Use Wildcards in Production

**WRONG** (Insecure):
```bash
# DO NOT DO THIS
export CORS_ORIGINS="*"
```

This allows ANY website to access your API, enabling:
- Cross-site scripting (XSS) attacks
- Data theft
- Unauthorized API access

### Use Specific Origins

**CORRECT** (Secure):
```bash
export CORS_ORIGINS="https://app.yourdomain.com"
```

### Use HTTPS Origins

Always use `https://` in production:

**WRONG**:
```bash
export CORS_ORIGINS="http://app.example.com"  # Insecure
```

**CORRECT**:
```bash
export CORS_ORIGINS="https://app.example.com"  # Secure
```

### Separate Development and Production

```bash
# Development (.env.development)
CORS_ORIGINS=http://localhost:3000,http://localhost:3001

# Production (.env.production)
CORS_ORIGINS=https://app.yourdomain.com,https://admin.yourdomain.com
```

## Common Scenarios

### React / Vue / Angular App

**Local Development**:
```bash
# Default port for Create React App, Vue CLI, Angular CLI
CORS_ORIGINS=http://localhost:3000
```

**Production**:
```bash
# Your deployed web app
CORS_ORIGINS=https://app.yourdomain.com
```

### Multiple Environments

```bash
# Development, staging, and production
CORS_ORIGINS=http://localhost:3000,https://staging-app.yourdomain.com,https://app.yourdomain.com
```

### Mobile App + Web App

```bash
# Web app only (mobile apps don't need CORS)
CORS_ORIGINS=https://app.yourdomain.com

# Note: Native mobile apps make direct HTTP requests
# and are not subject to CORS restrictions
```

### Internal Lab Network

```bash
# Lab's internal domains
CORS_ORIGINS=http://research-ui.lab.university.edu,http://admin.lab.university.edu
```

### Microservices Architecture

```bash
# API gateway and internal services
CORS_ORIGINS=https://gateway.yourdomain.com,https://admin.yourdomain.com
```

## Troubleshooting

### CORS Error in Browser Console

**Error Message**:
```
Access to XMLHttpRequest at 'http://localhost:18011/api/v1/jobs'
from origin 'http://localhost:3001' has been blocked by CORS policy:
No 'Access-Control-Allow-Origin' header is present on the requested resource.
```

**Solution**:
Add your origin to `CORS_ORIGINS`:

```bash
export CORS_ORIGINS="http://localhost:3001"
uv run python api_server.py
```

### CORS Works Locally But Not in Production

**Problem**: Local development works, but production fails.

**Common Causes**:
1. **Origin mismatch**: Check exact URL (http vs https, www vs non-www)
   ```bash
   # If your site is at https://www.example.com, use:
   CORS_ORIGINS=https://www.example.com

   # NOT:
   CORS_ORIGINS=https://example.com  # Missing www
   ```

2. **Environment variable not set**: Verify on production server
   ```bash
   # SSH into production server
   echo $CORS_ORIGINS
   ```

3. **Reverse proxy stripping headers**: Check nginx/apache config
   ```nginx
   # nginx - preserve CORS headers
   proxy_pass_request_headers on;
   ```

### Preflight Requests Failing

**Problem**: OPTIONS requests return 403/401.

**Cause**: API requires authentication, but preflight requests don't send credentials.

**Solution**: VideoAnnotator automatically allows OPTIONS requests. Ensure your client sends:

```javascript
fetch('http://localhost:18011/api/v1/jobs', {
  method: 'GET',
  headers: {
    'Authorization': 'Bearer va_api_...',
  },
  credentials: 'include'  // Important for CORS with auth
})
```

### CORS Allows Origin But Request Still Fails

**Problem**: Browser shows CORS header, but request blocked.

**Possible Causes**:
1. **Authentication required**: Add Authorization header
   ```javascript
   headers: {
     'Authorization': 'Bearer your_api_key'
   }
   ```

2. **Credentials not included**: Add credentials flag
   ```javascript
   fetch(url, { credentials: 'include' })
   ```

3. **Content-Type issues**: Ensure proper Content-Type
   ```javascript
   headers: {
     'Content-Type': 'application/json'
   }
   ```

## Advanced Configuration

### Dynamic CORS Origins

For complex scenarios, you can modify `src/api/main.py`:

```python
import os

# Load from database, config file, etc.
def get_allowed_origins():
    # Custom logic to determine origins
    if os.environ.get("ENVIRONMENT") == "production":
        return ["https://app.prod.com"]
    elif os.environ.get("ENVIRONMENT") == "staging":
        return ["https://app.staging.com", "https://app.dev.com"]
    else:
        return ["http://localhost:3000"]

cors_origins = get_allowed_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### CORS with API Gateway

If using an API gateway (Kong, AWS API Gateway, etc.), handle CORS there:

**Kong**:
```yaml
plugins:
- name: cors
  config:
    origins:
    - https://app.example.com
    credentials: true
    max_age: 3600
```

**AWS API Gateway**:
```yaml
# serverless.yml
functions:
  api:
    handler: handler.main
    events:
      - http:
          path: /{proxy+}
          method: ANY
          cors:
            origin: https://app.example.com
            headers:
              - Authorization
            allowCredentials: true
```

Then disable VideoAnnotator's built-in CORS (API gateway handles it).

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated list of allowed origins |

### Allowed Values

| Value | Security | Use Case |
|-------|----------|----------|
| `https://app.example.com` | ✅ Secure | Production (recommended) |
| `http://localhost:3000` | ⚠️ Dev only | Local development |
| `https://example.com,https://admin.example.com` | ✅ Secure | Multiple production origins |
| `*` | ❌ **INSECURE** | **Never use** |

### CORS Headers

VideoAnnotator automatically sets:

- `Access-Control-Allow-Origin`: Matching origin or null
- `Access-Control-Allow-Credentials`: `true`
- `Access-Control-Allow-Methods`: `GET, POST, PUT, DELETE, OPTIONS, PATCH`
- `Access-Control-Allow-Headers`: All requested headers
- `Access-Control-Max-Age`: Browser-dependent (typically 86400)

## Testing

### Manual Testing

```bash
# Test from browser console (press F12)
fetch('http://localhost:18011/api/v1/health', {
  headers: { 'Authorization': 'Bearer va_api_...' }
})
  .then(r => r.json())
  .then(console.log)
  .catch(console.error)
```

### Automated Testing

```python
# tests/integration/test_cors.py
def test_cors_allowed_origin(client):
    response = client.options(
        "/api/v1/jobs",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST"
        }
    )
    assert response.status_code in (200, 204)
    assert "access-control-allow-origin" in response.headers

def test_cors_disallowed_origin(client):
    response = client.options(
        "/api/v1/jobs",
        headers={
            "Origin": "http://evil.com",
            "Access-Control-Request-Method": "POST"
        }
    )
    # Should not include allow-origin header for disallowed origin
    if "access-control-allow-origin" in response.headers:
        assert response.headers["access-control-allow-origin"] != "http://evil.com"
```

## See Also

- [Authentication Guide](authentication.md) - API key management
- [Production Checklist](production_checklist.md) - Complete security guide
- [MDN CORS Documentation](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS) - Deep dive into CORS
