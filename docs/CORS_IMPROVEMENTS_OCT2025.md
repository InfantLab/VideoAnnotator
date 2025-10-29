# CORS Configuration Update — October 2025

**Quick Summary for Client Team**: CORS is now frictionless for local development while remaining secure by default.

## 🎯 What Changed

### Before (Painful)
- Only 2 origins allowed by default: `http://localhost:3000` and `http://localhost:18011`
- If your React/Vite/Vue/Angular app ran on a different port → CORS error
- Manual environment variable setup required
- Documentation was intimidating and technical

### After (Frictionless)
- **12 common development ports work out of the box**
- **New `--dev` flag for testing** (allows all origins)
- **Simple troubleshooting** with copy-paste solutions
- Production remains secure (can still lock down to specific origins)

## ✅ What This Means for You

### Zero Configuration Development

Just start the server and your web app works:

```bash
# Start server
uv run videoannotator server

# Your web client automatically works if running on:
# - React:   http://localhost:3000 or 3001
# - Vite:    http://localhost:5173 or 5174
# - Vue CLI: http://localhost:8080 or 8081
# - Angular: http://localhost:4200
# - 127.0.0.1 variants of above ports
```

**No environment variables needed!**

### Testing with Remote Clients or Unusual Ports

Use the new `--dev` flag:

```bash
# Allows ALL origins (*), disables auth - perfect for testing
uv run videoannotator server --dev
```

Console shows:
```
[START] Starting server in DEVELOPMENT mode
[WARNING] CORS origins: * (ALL origins allowed)
[WARNING] Authentication: DISABLED
```

### Production Deployment (Lock It Down)

For production, set specific allowed origins:

```bash
# Only allow your production domain
export CORS_ORIGINS="https://myapp.example.com"
uv run videoannotator server

# Or multiple domains
export CORS_ORIGINS="https://app.example.com,https://admin.example.com"
uv run videoannotator server
```

## 🚀 Quick Start Examples

### React Development (Create React App / Vite)
```bash
# Terminal 1: Start VideoAnnotator server
uv run videoannotator server

# Terminal 2: Start your React app
npm start    # CRA default: localhost:3000 ✅
# or
npm run dev  # Vite default: localhost:5173 ✅

# CORS just works - no configuration needed!
```

### Vue Development
```bash
# Terminal 1: Start VideoAnnotator server
uv run videoannotator server

# Terminal 2: Start your Vue app
npm run serve  # Default: localhost:8080 ✅

# CORS just works!
```

### Angular Development
```bash
# Terminal 1: Start VideoAnnotator server
uv run videoannotator server

# Terminal 2: Start your Angular app
ng serve  # Default: localhost:4200 ✅

# CORS just works!
```

### Testing from Remote Machine or Codespace
```bash
# Use dev mode to allow any origin
uv run videoannotator server --dev

# Your remote client can now connect
```

## 🔍 Verification

Check if your origin is allowed:

```bash
# Test CORS preflight request
curl -H "Origin: http://localhost:5173" \
     -H "Access-Control-Request-Method: POST" \
     -X OPTIONS http://localhost:18011/api/v1/jobs

# Success: Response includes access-control-allow-origin header
# Blocked: No access-control-allow-origin header in response
```

Console logs show CORS configuration on startup:
```
[SECURITY] CORS: 12 origins allowed (includes common dev ports: 3000, 5173, 8080, 4200, ...)
[SECURITY] CORS debug - Full origins list: http://localhost:3000,http://localhost:3001,...
```

## 📚 Updated Documentation

- **[Getting Started Guide](usage/GETTING_STARTED.md)** - Updated with dev mode examples
- **[Troubleshooting Guide](installation/troubleshooting.md)** - Simplified CORS section
- **[CORS Configuration Guide](security/cors.md)** - Complete reference

## 🐛 Troubleshooting (Rare Cases)

### Still Getting CORS Errors?

1. **Check your client port**:
   ```bash
   # Is your app running on one of these?
   # 3000, 3001, 5173, 5174, 8080, 8081, 4200, 18011
   ```

2. **Try dev mode** (allows all origins):
   ```bash
   uv run videoannotator server --dev
   ```

3. **Set custom origin**:
   ```bash
   export CORS_ORIGINS="http://localhost:YOUR_PORT"
   uv run videoannotator server
   ```

4. **Check browser console**:
   - Look for the exact origin being blocked
   - Verify the server is running and accessible

### Common Pitfalls

- ❌ **Using `https://` locally**: Browsers use `http://localhost`, not `https://`
- ❌ **Wrong port**: Check which port your client actually runs on
- ❌ **Cached credentials**: Clear browser cache/cookies if changing auth settings
- ❌ **Proxying requests**: If using a proxy, configure it to forward CORS headers

## 💡 Best Practices

### Development
- Use default configuration (just works for common frameworks)
- Use `--dev` flag for testing remote clients or unusual configurations
- Check server logs to see active CORS configuration

### Staging
- Set explicit `CORS_ORIGINS` for your staging domain
- Keep authentication enabled
- Test with realistic client origins

### Production
- **Always** set specific `CORS_ORIGINS` (never use `*`)
- Enable authentication (`AUTH_REQUIRED=true`, which is the default)
- Use HTTPS origins: `https://app.example.com`
- Monitor logs for blocked CORS requests

## 🔗 Related Changes

This update is part of making VideoAnnotator more accessible to researchers and developers. Related improvements:

- **Authentication**: Now auto-generates API keys on first start
- **CLI**: Simplified commands (`uv run videoannotator server`)
- **Logging**: Shows CORS configuration and origin count on startup
- **Error Messages**: More helpful hints when CORS blocks requests

## Questions?

- Check [CORS Guide](security/cors.md) for detailed configuration
- See [Troubleshooting Guide](installation/troubleshooting.md) for common issues
- Review [Getting Started](usage/GETTING_STARTED.md) for quick setup

---

**Technical Details** (for server team):
- Default origins defined in `src/videoannotator/config_env.py`
- CORS middleware configured in `src/videoannotator/api/main.py`
- CLI server command in `src/videoannotator/cli.py`
- Logging in `src/videoannotator/api/startup.py`
