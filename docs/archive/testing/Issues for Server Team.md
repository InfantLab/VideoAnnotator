# Issues for VideoAnnotator Server Team

**Document Purpose:** Track server-side issues discovered during v0.4.0 QA testing that need to be addressed in the VideoAnnotator backend.

**Date Created:** 2025-10-09
**Client Version:** v0.4.0
**Server Version Tested:** v1.2.2
**Status:** Active tracking

---

## 🔴 HIGH PRIORITY - Job Execution Failures

### 1. Jobs Failing to Complete Successfully

**Endpoint:** `POST /api/v1/jobs` (job submission) and job processing
**Issue:** During QA testing, 8 jobs failed to complete successfully.

**🔍 ROOT CAUSE IDENTIFIED:**

```
Error: Unknown pipeline: audio_processing
```

All 8 failed jobs have the same error - the server doesn't recognize the pipeline named `audio_processing`.

**Impact:**

- Jobs submitted through the v0.4.0 job creation wizard are failing
- Unable to verify end-to-end workflow from job submission to results viewing
- User experience degraded - cannot test complete feature set

**Diagnostic Results:**

**Failed Jobs:** 8 total

- Job IDs: See `failed_jobs_diagnostics.json` for complete list
- Date Range: 2025-09-28 to 2025-10-09
- **Common Error**: `Unknown pipeline: audio_processing`

**Questions for Server Team:**

1. **Pipeline Naming Mismatch**:

   - Is there an `audio_processing` pipeline in v1.2.2?
   - Should it be named differently (e.g., `audio_transcription`, `speech_recognition`)?
   - Is this pipeline missing from the server configuration?

2. **Available Pipelines**:

   - What are the correct pipeline names for v1.2.2?
   - Can you provide output of `GET /api/v1/pipelines`?
   - Is there a naming convention change between versions?

3. **Client-Server Mismatch**:
   - Is there a discrepancy between what pipelines the client expects and what the server provides?
   - Should the pipeline catalog endpoint provide canonical names?
   - Any documentation on pipeline naming?

**To Reproduce:**

1. Start VideoAnnotator server v1.2.2 on localhost:18011
2. Use Video Annotation Viewer v0.4.0
3. Navigate to `/create/new`
4. Upload test video (e.g., `demo/videos/3.mp4`)
5. Select pipeline that includes "audio_processing"
6. Configure parameters and submit
7. Job fails immediately with "Unknown pipeline: audio_processing"

**Expected Behavior:**

- Jobs should complete successfully with valid pipelines
- OR client should not offer pipelines that don't exist
- OR server should return clear error during pipeline selection

**Actual Behavior:**

- All jobs fail with "Unknown pipeline: audio_processing"
- Job submission succeeds (returns job ID)
- Job immediately transitions to failed status
- No results generated

**Hypothesis:**

- Client has outdated or incorrect pipeline names
- Server pipeline configuration missing audio processing
- Pipeline naming changed between versions
- Need pipeline catalog endpoint to sync names

**Client Actions Taken:**

- Verified token authentication works
- Verified API connectivity
- Job submission succeeds (receives job ID)
- Job list shows jobs but with failed status
- Created diagnostic script: `scripts/diagnose_failed_jobs.py`

**How to Gather Diagnostic Information:**

**Step 1: Run Client Diagnostic Script**

```bash
cd video-annotation-viewer
python scripts/diagnose_failed_jobs.py
```

This generates `failed_jobs_diagnostics.json` with:

- Failed job IDs and details
- Error messages (if available)
- Pipeline configurations
- Timestamps

**Step 2: Collect Server Information**

Please provide:

1. **Server Logs**:

   ```bash
   # Check server logs during the failure period
   # Look for ERROR, EXCEPTION, or FAILED keywords
   grep -i "error\|exception\|failed" /path/to/server/logs/*.log
   ```

2. **Server Configuration**:

   ```bash
   # Server version and environment
   python -m videoannotator.server --version

   # Check system resources
   df -h  # Disk space
   free -h  # Memory
   nvidia-smi  # GPU (if applicable)
   ```

3. **Pipeline Status**:

   ```bash
   # Check if all pipelines are properly configured
   curl -H "Authorization: Bearer dev-token" \
     http://localhost:18011/api/v1/pipelines
   ```

4. **Job-Specific Logs**:
   - Logs for specific failed job IDs
   - Any stack traces or error messages
   - Resource usage during job execution

**Step 3: Check Common Issues**

- [ ] All required dependencies installed?
- [ ] Model files downloaded and accessible?
- [ ] Sufficient disk space (check temp directories)?
- [ ] CUDA/GPU available if required?
- [ ] Network access for model downloads?
- [ ] File permissions correct?

**Server Action Needed:**

- [ ] Review server logs for error details
- [ ] Run diagnostic checks listed above
- [ ] Verify pipeline configurations
- [ ] Check resource availability
- [ ] Provide `failed_jobs_diagnostics.json` and server logs to development team
- [ ] Document common failure modes and troubleshooting steps

---

## 🔴 HIGH PRIORITY - Authentication & Security

### 2. Debug Endpoint Returns 401 with Valid Token

**Endpoint:** `GET /api/v1/debug/token-info`
**Issue:** Returns 401 Unauthorized even when using a valid token (`dev-token`) that works for other endpoints.

**Impact:**

- Spams browser console with 401 errors during normal operations
- Client cannot retrieve detailed token information (user, permissions, expiry)
- Forces client to fall back to alternative validation methods

**Steps to Reproduce:**

1. Start server with default dev configuration
2. Make request with valid token:
   ```bash
   curl -H "Authorization: Bearer dev-token" http://localhost:18011/api/v1/debug/token-info
   ```
3. Observe 401 response

**Expected Behavior:**

- Should return 200 with token details when valid token is provided
- OR endpoint should be removed/disabled if intentionally restricted
- OR documentation should clarify this endpoint requires special permissions

**Workaround (Client):**

- Client now silently handles 401s on this endpoint
- Falls back to `/api/v1/jobs` for basic token validation
- Missing detailed token metadata (user, permissions, expiry)

**Server Action Needed:**

- [ ] Fix authentication for debug endpoint with standard tokens
- [ ] OR document that this endpoint is restricted/admin-only
- [ ] OR remove endpoint if not intended for production use

---

## 🟡 MEDIUM PRIORITY - API Consistency

### 3. Pipeline Catalog Endpoint Missing (404)

**Endpoint:** `GET /api/v1/pipelines/catalog`
**Issue:** Returns 404 Not Found on v1.2.2 server

**Impact:**

- Client cannot fetch dynamic pipeline catalog
- Must fall back to hardcoded pipeline definitions
- Loses v1.2.x introspection capabilities

**Expected in v1.2.x:**
According to roadmap and documentation, v1.2.x should support pipeline catalog introspection.

**Current Behavior:**

```
GET http://localhost:18011/api/v1/pipelines/catalog 404 (Not Found)
```

**Server Action Needed:**

- [ ] Verify if endpoint should exist in v1.2.2
- [ ] Implement catalog endpoint if planned but missing
- [ ] Update server version requirements in documentation if v1.2.3+ required

**Question for Server Team:**
Is pipeline catalog endpoint implemented? Expected in which version?

---

## 🟢 LOW PRIORITY - Documentation & Clarity

### 4. Inconsistent Default Configuration

**Issue:** Server documentation and examples don't clearly specify expected default values for:

- Default API token for development (`dev-token` vs `video-annotator-dev-token-please-change`)
- Expected CORS origins (`localhost:19011` vs `localhost:8080`)
- Default ports (18011 is standard, but examples vary)

**Impact:**

- Developers experience authentication errors on first setup
- Client and server have different default assumptions
- Requires manual configuration to get started

**Server Action Needed:**

- [ ] Standardize default token in documentation
- [ ] Provide clear "getting started" configuration guide
- [ ] Include example `.env` file for development setup

---

## 📋 QUESTIONS FOR SERVER TEAM

### Q1: Debug Endpoint Intent

Is `/api/v1/debug/token-info` intended for:

- [ ] All authenticated users
- [ ] Admin/special permissions only
- [ ] Development/debugging only (should be disabled in production)

### Q2: v1.2.x Feature Availability

Which features are actually implemented in v1.2.2?

- [ ] Pipeline catalog (`/api/v1/pipelines/catalog`)
- [ ] Parameter schemas per pipeline
- [ ] Pipeline health/status endpoints
- [ ] Server capability introspection

### Q3: Authentication Strategy

What's the recommended authentication approach for:

- Local development (current: `dev-token`)
- Production deployments
- Multiple client applications

---

## 🔍 TESTING NOTES

### Server Setup Used for Testing

```bash
# Server command
python -m videoannotator.server --port 18011

# Version info
VideoAnnotator v1.2.2
```

### Client Configuration

```bash
# .env settings
VITE_API_BASE_URL=http://localhost:18011
VITE_API_TOKEN=dev-token
```

### Successful Endpoints (for reference)

✅ `GET /health` → 200
✅ `GET /api/v1/system/health` → 200
✅ `GET /api/v1/jobs` → 200 (with valid token)
✅ `GET /api/v1/pipelines` → 200
✅ `POST /api/v1/jobs` → 201 (job creation works)

### Problematic Endpoints

❌ `GET /api/v1/debug/token-info` → 401 (should work with valid token)
❌ `GET /api/v1/pipelines/catalog` → 404 (expected in v1.2.x)

---

## 📞 CONTACT & COLLABORATION

**Client Repository:** https://github.com/InfantLab/video-annotation-viewer
**Server Repository:** https://github.com/InfantLab/VideoAnnotator

**For Questions:**

- Create issue in respective repository
- Tag with `server-client-integration` label
- Reference this document

---

## ✅ RESOLVED ISSUES

_(Issues will be moved here when fixed)_

---

**Document Version:** 1.0
**Last Updated:** 2025-10-09
**Next Review:** After server team response
