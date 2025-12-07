# Localhost Connection Troubleshooting

If you see "Unable to connect to server" errors despite the server running, this guide will help diagnose the issue.

## 🚨 Critical: DNS & Trailing Slashes (Dec 2025 Update)

We encountered a complex connection failure chain that required multiple fixes. If you are seeing "COMPLETE FAILURE" or timeouts:

### 1. DNS Hijacking of `localhost`
**Symptom:** Browser requests to `localhost` timeout, but `127.0.0.1` works instantly.
**Cause:** Some ISPs (e.g., Community Fibre) or routers hijack DNS resolution for `localhost`, resolving it to a public IP (e.g., `103.86.96.100`) instead of the local loopback.
**Fix:** The application now automatically forces `127.0.0.1` in `src/api/client.ts`.
**Manual Check:** Run `nslookup localhost` in terminal. If it returns a public IP, you MUST use `127.0.0.1`.

### 2. Strict Trailing Slashes (FastAPI) - RESOLVED
**Symptom:** `404 Not Found` on specific endpoints (like `/jobs`), even though the server is running.
**Cause:** The VideoAnnotator server (FastAPI) was strict about trailing slashes.
- `GET /api/v1/jobs` -> **404 Not Found**
- `GET /api/v1/jobs/` -> **200 OK**
**Fix:** The server has been updated to support both formats. The client now uses standard paths (no trailing slash).

### 3. Token Validation
**Symptom:** "Authentication Required" even with a token.
**Cause:** The client was rejecting `dev-token` as an invalid pattern.
**Fix:** `dev-token` is now explicitly allowed in `src/api/client.ts`.

---

## Automated Diagnostics (Recommended)

**The easiest way to diagnose connection issues:**

1. Go to Settings page in the app
2. Click **"Run Diagnostics"** button
3. Review the automated test results and recommendations
4. The full report is automatically copied to your clipboard

The diagnostics will test:
- DNS resolution (localhost vs 127.0.0.1)
- Basic connectivity
- CORS preflight
- Authentication (if token provided)
- Server version detection

And provide specific recommendations for any failures.

## Manual Diagnosis

### Test 1: PowerShell Connection Test
```powershell
Invoke-WebRequest -Uri "http://localhost:18011/api/v1/system/health" -UseBasicParsing
```

**Expected**: Status 200, response in 2-5 seconds

**If this works**: Server is fine, problem is browser-specific

### Test 2: Browser Fetch Test  
Open browser console on any page and run:
```javascript
fetch('http://localhost:18011/api/v1/system/health')
  .then(r => console.log('Success:', r.status))
  .catch(e => console.error('Failed:', e.message))
```

**If this times out or fails**: Browser cannot reach server

## Common Issue: Disabled localhost in hosts file

**Symptom**: PowerShell works (2-5s), browser times out (30s+)

**Cause**: Windows hosts file has localhost entries commented out, causing DNS resolution to fail or try IPv6 first

**Check**:
```powershell
Get-Content C:\Windows\System32\drivers\etc\hosts | Select-String "localhost"
```

**Fix** (requires Administrator):
1. Open PowerShell as Administrator
2. Run: `notepad C:\Windows\System32\drivers\etc\hosts`
3. Find these lines:
   ```    
   #       127.0.0.1       localhost
   #       ::1             localhost
   ```
4. Uncomment the IPv4 line (remove the `#`):
   ```
   127.0.0.1       localhost
   #       ::1             localhost
   ```
5. Save and close
6. Restart browser

**Why this happens**: Docker Desktop, Windows Updates, or security software may modify the hosts file

## Workaround: Use 127.0.0.1 directly

If you can't edit the hosts file, you can use `http://127.0.0.1:18011` instead of `http://localhost:18011` in your settings.

1. Go to Settings page
2. Change API Server URL from `http://localhost:18011` to `http://127.0.0.1:18011`
3. Save and test connection

## Other Possible Causes

### IPv6 vs IPv4
- **Check server binding**: `netstat -an | Select-String "18011"`
- If server only shows `127.0.0.1:18011` (not `[::]:18011`), it's IPv4-only
- Browser may try IPv6 first, causing delays

### Firewall/Antivirus
- Some security software blocks browser → localhost connections
- Try temporarily disabling firewall/antivirus

### Browser Extensions
- Privacy/security extensions (uBlock, Privacy Badger, etc.) may block localhost
- Try disabling all extensions or use incognito mode

### CORS Issues
- Check browser console for CORS errors
- VideoAnnotator v1.3.0+ auto-configures CORS for common ports
- Verify server logs show OPTIONS requests arriving

## Verification Commands

**Check localhost resolution**:
```powershell
[System.Net.Dns]::GetHostAddresses("localhost")
```
Should return: `127.0.0.1` and/or `::1`

**Check server is listening**:
```powershell
netstat -an | Select-String "18011"
```
Should show: `TCP    127.0.0.1:18011    0.0.0.0:0    LISTENING`

**Test with curl** (if installed):
```powershell
curl http://localhost:18011/api/v1/system/health
```

## Still Not Working?

Check the [GitHub Issues](https://github.com/InfantLab/video-annotation-viewer/issues) or create a new issue with:
- Browser name and version
- PowerShell test results
- Browser console errors
- Contents of hosts file check
- Server startup logs
