# 🎯 SERVER API ALIGNMENT FIXES - 2025-09-26

## Root Cause Analysis

The client was making requests to **endpoints that don't exist** on the server, while ignoring endpoints that **do exist**. 

### Server Truth Discovery

Using the server's OpenAPI specification at `http://localhost:18011/openapi.json`:

**✅ Server Actually Provides (20 endpoints total):**
- `/api/v1/debug/server-info` ← **Use this for server info**
- `/api/v1/pipelines/` ← **Use this for pipeline catalog**  
- `/api/v1/system/health` ← **Use this for health checks**
- `/api/v1/system/config`, `/api/v1/system/metrics`, `/api/v1/system/database`
- Full job management: `/api/v1/jobs/`, `/api/v1/jobs/{id}`, etc.
- Debug endpoints: `/api/v1/debug/*`

**❌ Client Was Requesting (404 Not Found):**
- `/api/v1/system/info` → **Does not exist**
- `/api/v1/system/server-info` → **Does not exist**
- `/api/v1/pipelines/catalog` → **Does not exist**

## Fixes Applied

### 1. API Client Endpoint Alignment (`src/api/client.ts`)

**Before:**
```typescript
// Client tried non-existent endpoints first
'/api/v1/pipelines/catalog'     // 404 - doesn't exist
'/api/v1/system/server-info'    // 404 - doesn't exist  
'/api/v1/system/info'           // 404 - doesn't exist
```

**After:**
```typescript
// Client uses actual server endpoints from OpenAPI spec
'/api/v1/debug/server-info'     // ✅ Works - exists in server
'/api/v1/pipelines/'            // ✅ Works - exists in server
'/api/v1/system/health'         // ✅ Works - exists in server
```

**Changes Made:**
- Removed attempt to call `/api/v1/pipelines/catalog` (doesn't exist)
- Reordered endpoint detection to try working endpoints first
- Added promise caching to prevent duplicate concurrent requests
- Updated comments to reference server's actual OpenAPI spec

### 2. Token Authentication Fix (`.env`)

**Before:** `VITE_API_TOKEN=video-annotator-dev-token-please-change`  
**After:** `VITE_API_TOKEN=dev-token` (matches server expectation)

### 3. Port Configuration (`vite.config.ts`)

**Before:** Port 8080  
**After:** Port 19011 (complements server's 18011)

### 4. DOM Structure Fix (`src/components/TokenSetup.tsx`)

**Before:** `<p><Badge>...</Badge></p>` (invalid nesting)  
**After:** `<div><Badge>...</Badge></div>` (valid structure)

## Results

### ✅ **All Issues Resolved:**
1. **"pipelines.map is not a function"** → Fixed by using actual `/api/v1/pipelines/` endpoint
2. **401 Unauthorized errors** → Fixed by matching server's expected token
3. **Excessive 404 console spam** → Fixed by using existing endpoints only
4. **React DOM nesting warnings** → Fixed invalid HTML structure
5. **Duplicate API requests** → Fixed with promise caching

### 📊 **API Test Results:**
```
✅ Health Check: 200
✅ System Health: 200  
✅ Debug Server Info: 200
✅ Pipelines List: 200 (found 6 pipelines)

✅ Missing endpoints return expected 404s:
❌ System Info (missing): 404 (expected)
❌ Pipeline Catalog (missing): 404 (expected)  
```

### 🧪 **QA Checklist Status:**
- ✅ Pipeline catalog displays 6 server-fetched pipelines
- ✅ Console errors reduced to minimum necessary
- ✅ Token authentication working
- ✅ Port configuration complementary
- ✅ No more DOM warnings

## Key Takeaway

**The server's OpenAPI specification at `/openapi.json` is the source of truth.** The client should align with what the server actually provides rather than making assumptions about v1.2.x endpoint availability.

## Next Steps

1. **Server Version Detection**: Fix parsing to show actual server version instead of "Unknown"
2. **Feature Flag UI**: Color-code enabled/disabled features appropriately
3. **OpenAPI Integration**: Consider using server's OpenAPI spec for dynamic client configuration

---
**Fixed by:** GitHub Copilot  
**Date:** 2025-09-26  
**Server:** VideoAnnotator v1.2.2 (API v1.2.0)  
**Client:** Video Annotation Viewer v0.4.0