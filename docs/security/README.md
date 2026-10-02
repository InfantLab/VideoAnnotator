# Security Documentation

VideoAnnotator includes comprehensive security features for production deployments. This directory contains guides for securing your VideoAnnotator instance.

## 📚 Documentation

### [Authentication Guide](authentication.md)
Complete guide to API key management:
- Getting your auto-generated API key
- Using API keys in requests
- Generating additional keys
- Revoking compromised keys
- Development vs. production configurations
- Troubleshooting authentication issues

**Start here if**: You need to use the API or manage access control.

### [CORS Configuration](cors.md)
Cross-Origin Resource Sharing (CORS) security:
- Default secure configuration
- Allowing specific web origins
- Multiple origin support
- Production best practices
- Common CORS errors and solutions

**Start here if**: You're building a web frontend or getting CORS errors.

### [Production Checklist](production_checklist.md)
Complete security hardening checklist:
- Pre-deployment security tasks
- Authentication & authorization
- Network security (HTTPS, firewalls, rate limiting)
- Data protection (encryption, backups)
- Monitoring & logging
- Server hardening
- Docker & Kubernetes deployment examples
- Incident response procedures

**Start here if**: You're deploying to production.

## 🚀 Quick Start

### Local Development

```bash
# 1. Start server (auth disabled for development)
export AUTH_REQUIRED=false
uv run videoannotator

# 2. Access API without authentication
curl http://localhost:18011/api/v1/jobs
```

### Production Deployment

```bash
# 1. Initialize DB + admin token (run once per environment)
uv run videoannotator setup-db --admin-email you@example.com --admin-username you

# 2. Start server (auth enabled by default)
uv run videoannotator

# 3. Use API key in requests
export API_KEY="va_api_your_key_here"
curl -H "Authorization: Bearer $API_KEY" \
  https://yourdomain.com/api/v1/jobs
```

## 🔐 Security Features

### Authentication (v1.4.1+)
- ✅ **Secure by default**: Authentication required
- ✅ **Auto-generation**: `videoannotator setup-db` creates admin key (server still auto-generates if missing)
- ✅ **Token-based**: Standard Bearer token authentication
- ✅ **Encrypted storage**: Keys stored in encrypted JSON
- ✅ **Granular permissions**: Scope-based access control (read, write, admin)
- ✅ **Key rotation**: Generate and revoke keys easily
- ✅ **Expiration support**: Optional key expiration

### CORS (v1.4.1+)
- ✅ **Restricted by default**: Only `localhost:3000` allowed
- ✅ **No wildcards**: Prevents unauthorized access
- ✅ **Multiple origins**: Comma-separated list support
- ✅ **Environment-based**: Easy production configuration
- ✅ **Preflight support**: Automatic OPTIONS handling

### Data Protection
- ✅ **Encrypted tokens**: Fernet encryption for API keys
- ✅ **Secure permissions**: 600 on sensitive files (Unix)
- ✅ **Path validation**: Prevents directory traversal
- ✅ **Input sanitization**: SQL injection protection

### Logging & Monitoring
- ✅ **Request logging**: All API calls logged
- ✅ **Error tracking**: Structured error logging
- ✅ **Security events**: Failed auth attempts logged
- ✅ **No sensitive data**: API keys masked in logs

## 🛡️ Security Model

### Threat Model

**Protected Against**:
- Unauthorized API access
- Cross-site request forgery (CSRF)
- SQL injection
- Path traversal attacks
- Cross-origin attacks (XSS via CORS)

**Requires Additional Protection** (deploy behind reverse proxy):
- DDoS attacks → Use rate limiting (nginx, CloudFlare)
- Man-in-the-middle → Use HTTPS/TLS
- Brute force → Use fail2ban or rate limiting

### Trust Boundaries

1. **Public Endpoints** (No Authentication):
   - `/health` - Health check
   - `/docs` - API documentation
   - `/redoc` - Alternative docs
   - `/openapi.json` - API specification

2. **Protected Endpoints** (Authentication Required):
   - `/api/v1/jobs/*` - All job operations
   - `/api/v1/pipelines/*` - Pipeline management
   - `/api/v1/system/*` - System operations

3. **Admin Endpoints** (Admin Scope Required):
   - TBD in a future release (user management, system config)

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTH_REQUIRED` | `true` | Enable/disable authentication |
| `AUTO_GENERATE_API_KEY` | `true` | Auto-generate key when server starts and no keys exist |
| `CORS_ORIGINS` | `http://localhost:3000` | Allowed CORS origins (comma-separated) |
| `DEBUG` | `false` | Enable debug mode (disable in production) |

### Files

| File | Purpose | Permissions |
|------|---------|-------------|
| `tokens/tokens.json` | Encrypted API keys | 600 (read/write owner only) |
| `tokens/encryption.key` | Encryption key for tokens | 600 (read/write owner only) |
| `videoannotator.db` | SQLite database | 644 (read/write owner, read group/others) |
| `<log dir>/api_server.log` | Application logs | 644 |
| `<log dir>/api_requests.log` | Request logs | 644 |

## 🔧 Common Tasks

### Generate API Key
```bash
uv run python -m scripts.manage_tokens create
```

### List Active Keys
```bash
uv run python -m scripts.manage_tokens list
```

### Revoke Key
```bash
uv run python -m scripts.manage_tokens revoke va_api_xxxxx
```

### Configure CORS
```bash
# Single origin
export CORS_ORIGINS="https://app.example.com"

# Multiple origins
export CORS_ORIGINS="https://app.example.com,https://admin.example.com"
```

### Disable Authentication (Development Only)
```bash
export AUTH_REQUIRED=false
uv run videoannotator
```

## 🐛 Troubleshooting

### 401 Unauthorized
- Check API key is set: `echo $API_KEY`
- Verify header format: `Authorization: Bearer va_api_...`
- Check key hasn't expired: `uv run python -m scripts.manage_tokens list`

### CORS Errors
- Check browser console for exact error
- Verify origin in `CORS_ORIGINS`: `echo $CORS_ORIGINS`
- Ensure origin matches exactly (http vs https, www vs non-www)

### API Key Lost
- Check `tokens/tokens.json` exists
- Generate new key: `uv run python -m scripts.manage_tokens create`
- For emergency access: `export AUTH_REQUIRED=false`

## 📞 Security Contact

Found a security vulnerability? **DO NOT** open a public issue.

Email: security@yourdomain.com (replace with your contact)

We take security seriously and will respond within 24 hours.

## 📝 License

All security documentation is part of VideoAnnotator and follows the same license as the main project.

## 🔗 See Also

- [Installation Guide](../installation/INSTALLATION.md)
- [Getting Started](../usage/GETTING_STARTED.md)
- [Deployment Guide](../deployment/Docker.md)
- [Contributing Guide](../../CONTRIBUTING.md)
