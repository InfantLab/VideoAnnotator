# VideoAnnotator Server Issue Report

**Report Date:** 2025-09-26  
**Reporter:** Video Annotation Viewer Client Team  
**Server Version:** VideoAnnotator v1.2.2 (API v1.2.0)  
**Server URL:** http://localhost:18011  
**Status:** ✅ **RESOLVED** - Server team provided fixes (2025-09-26)

## Issue Summary

~~VideoAnnotator server reports version `1.2.2` but is missing several endpoints that are expected to be available in v1.2.x releases according to the documentation and client expectations.~~

**UPDATE:** Server team has provided fixes for the major issues. The jobs endpoint now works correctly and token validation is functioning.

## Server Version Information

```json
{
  "status": "healthy",
  "api_version": "1.2.0",
  "videoannotator_version": "1.2.2",
  "message": "VideoAnnotator API is running"
}
```

**Key Observation:** There's a version discrepancy between `api_version` (1.2.0) and `videoannotator_version` (1.2.2).

## Missing Endpoints

The following endpoints return `404 Not Found` but are expected in v1.2.x:

### System Information Endpoints
- ❌ `GET /api/v1/system/info` → 404
- ❌ `GET /api/v1/system/server-info` → 404  
- ❌ `GET /api/v1/system/version` → 404
- ❌ `GET /api/v1/system/capabilities` → 404

### Pipeline Catalog Endpoints  
- ❌ `GET /api/v1/pipelines/catalog` → 404
- ❌ `GET /api/v1/pipelines/schema` → 404

### Debug Endpoints
- ❌ `GET /api/v1/debug/routes` → 404
- ❌ `GET /api/v1/debug/endpoints` → 404

## Working Endpoints

The following endpoints work correctly:

### Health & Status
- ✅ `GET /health` → 200 (includes version info)
- ✅ `GET /api/v1/system/health` → 200

### Pipeline Management
- ✅ `GET /api/v1/pipelines` → 200 (returns 6 pipelines)

### Debug & Authentication  
- ✅ `GET /api/v1/debug/token-info` → 200
- ✅ `GET /api/v1/debug/server-info` → 200 (comprehensive server info)

### Jobs
- ✅ `GET /api/v1/jobs` → 200 (Fixed in server update 2025-09-26)

## Impact on Client Applications

1. **Feature Detection Confusion**: Clients cannot reliably detect v1.2.x capabilities
2. **Fallback Required**: Clients must implement legacy fallbacks for all v1.2.x features  
3. **Pipeline Catalog**: No access to enhanced pipeline catalog with metadata
4. **Server Introspection**: Limited ability to query server capabilities programmatically

## Questions for Server Team

1. **Version Discrepancy**: Why does `api_version` report 1.2.0 while `videoannotator_version` reports 1.2.2?

2. **Endpoint Availability**: Are the missing v1.2.x endpoints:
   - Not yet implemented in this release?
   - Behind feature flags that need to be enabled?
   - Available at different paths than documented?
   - Planned for a future API version?

3. **API Versioning**: Should clients expect:
   - API version to match VideoAnnotator version?
   - Gradual rollout of v1.2.x endpoints?
   - Explicit feature negotiation?

4. ~~**Jobs Endpoint**: The `GET /api/v1/jobs` endpoint returns 500 - is this a known issue?~~ ✅ **FIXED**

## Recommendations

### For Server Team:
1. **Clarify versioning strategy** - How should API version relate to VideoAnnotator version?
2. **Document endpoint availability** - Which endpoints are available in which API versions?
3. **Fix jobs endpoint** - Address the 500 error on `GET /api/v1/jobs`
4. **Consider feature flags** - If endpoints are behind flags, document how to enable them

### For Client Applications:
1. **Use API version for feature detection** - Check `api_version` rather than `videoannotator_version`
2. **Implement robust fallbacks** - Always have legacy endpoint fallbacks
3. **Utilize working debug endpoints** - Use `/api/v1/debug/server-info` for server introspection

## Test Results File

Complete test results with detailed HTTP responses available in: `server_investigation_results.json`

## Client Workaround

The Video Annotation Viewer client has been updated to:
- ✅ Properly handle API v1.2.0 limitations
- ✅ Use working debug endpoints for server information  
- ✅ Fall back gracefully to legacy endpoints
- ✅ Display appropriate messaging about missing v1.2.x features

---

**Contact:** Video Annotation Viewer development team  
**Client Repository:** https://github.com/InfantLab/video-annotation-viewer