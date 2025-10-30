# CORS Configuration Update — October 2025

**Quick Summary for Client Team**: CORS now correctly whitelists the official video-annotation-viewer client (port 19011) by default.

## 🎯 What Changed

### Before (Broken)
- Default CORS only allowed port 3000
- The official client runs on port 19011 → **CORS error**
- Users had to manually configure `CORS_ORIGINS` environment variable

### After (Fixed)
- **Default allows port 19011** (video-annotation-viewer) and 18011 (server)
- **Official client works out of the box**
- **New `--dev` flag** for testing with custom/remote clients (allows all origins)
- Production remains secure (just 2 specific ports by default)

## ✅ What This Means for You

### Official Client (video-annotation-viewer)

**Just works** - no configuration needed:

```bash
# Terminal 1: Start VideoAnnotator server (simplest command)
uv run videoannotator

# Terminal 2: Start video-annotation-viewer
npm start  # Runs on localhost:19011 by default ✅

# CORS just works - no configuration needed!
```

**Note**: `uv run videoannotator` automatically starts the server - no need for `server` subcommand.

The server logs will show:
```
[SECURITY] CORS: Allowing official client (port 19011) and server (port 18011)
```

### Testing with Custom Ports or Remote Clients

If you're developing a **custom client** or testing from a **remote machine/codespace**, use the `--dev` flag:

```bash
# Allows ALL origins (*), disables auth - perfect for testing
uv run videoannotator --dev
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
uv run videoannotator

# Or multiple domains
export CORS_ORIGINS="https://app.example.com,https://admin.example.com"
uv run videoannotator
```

## 🚀 Quick Start Examples

### Standard Usage (Official Client)
```bash
# Terminal 1: Start VideoAnnotator server
uv run videoannotator

# Terminal 2: Start video-annotation-viewer
cd /path/to/video-annotation-viewer
npm start  # Default: localhost:19011 ✅

# CORS is automatically configured!
```

**Tip**: `uv run videoannotator` is shorthand for `uv run videoannotator server` - the server is the default command.

### Custom Client Development
```bash
# If developing a custom client on a different port:
export CORS_ORIGINS="http://localhost:YOUR_PORT"
uv run videoannotator

# Or use dev mode to allow any origin:
uv run videoannotator --dev
```

## 🔍 Verification

Check if CORS is working:

```bash
# Test the official client port (19011)
curl -H "Origin: http://localhost:19011" \
     -H "Access-Control-Request-Method: POST" \
     -X OPTIONS http://localhost:18011/api/v1/jobs

# Success: Response includes access-control-allow-origin header
# Blocked: No access-control-allow-origin header in response
```

Console logs show CORS configuration on startup:
```
[SECURITY] CORS: Allowing official client (port 19011) and server (port 18011)
```

## 📚 Updated Documentation

- **[Getting Started Guide](usage/GETTING_STARTED.md)** - Updated with dev mode examples
- **[Troubleshooting Guide](installation/troubleshooting.md)** - Simplified CORS section
- **[CORS Configuration Guide](security/cors.md)** - Complete reference

## 🐛 Troubleshooting (Rare Cases)

### Still Getting CORS Errors?

1. **Verify video-annotation-viewer port**:
   ```bash
   # Check if running on default port 19011
   # Look for "Local: http://localhost:19011" in npm start output
   ```

2. **Try dev mode** (allows all origins - for testing only):
   ```bash
   uv run videoannotator --dev
   ```

3. **Set custom origin** (if using non-standard port):
   ```bash
   export CORS_ORIGINS="http://localhost:YOUR_PORT"
   uv run videoannotator
   ```

4. **Check browser console**:
   - Look for the exact origin being blocked
   - Verify the server is running on port 18011
   - Verify the client is running on port 19011

### Common Pitfalls

- ❌ **Using `https://` locally**: Browsers use `http://localhost`, not `https://`
- ❌ **Wrong port**: Check which port your client actually runs on
- ❌ **Cached credentials**: Clear browser cache/cookies if changing auth settings
- ❌ **Proxying requests**: If using a proxy, configure it to forward CORS headers

## 💡 Best Practices

### Development (Official Client)
- Use default configuration (automatically allows port 19011)
- No environment variables needed
- Check server logs to confirm CORS config

### Development (Custom Client)
- Set `CORS_ORIGINS` to your custom port
- Or use `--dev` flag for maximum flexibility (disables auth)
- Test CORS preflight with curl

### Staging
- Set explicit `CORS_ORIGINS` for your staging domain
- Keep authentication enabled
- Test with realistic client origins

### Production
- **Default is secure**: Only ports 18011 and 19011 allowed
- For deployed clients, set `CORS_ORIGINS` to your domain: `https://viewer.example.com`
- Enable authentication (`AUTH_REQUIRED=true`, which is the default)
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
