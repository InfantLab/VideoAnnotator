# VideoAnnotator v1.3.0 Client Team Update

**Release**: v1.3.0-dev (Feature Branch: `001-videoannotator-v1-3`)
**Date**: October 27, 2025
**Status**: Ready for Testing

## Executive Summary

VideoAnnotator v1.3.0 delivers **production reliability and critical fixes** for research lab deployments. This release focuses on data persistence, resource management, configuration validation, and security hardening—addressing the top blockers preventing production adoption.

**Key Improvements:**
- ✅ Videos and job data persist across server restarts (no more data loss)
- ✅ Cancel runaway GPU jobs via API (prevent resource exhaustion)
- ✅ Validate configurations before job submission (catch errors early)
- ✅ Secure-by-default authentication (protect lab resources)
- ✅ Consistent error responses across all API endpoints
- ✅ Enhanced diagnostics and monitoring capabilities

**For most users**: The changes are backward-compatible. Update your installation and continue using the existing API patterns documented in our guides.

---

## What's New: User-Facing Features

### 1. 🎯 Job Cancellation API (Priority 1)

**New Capability**: Cancel running or queued jobs to free up GPU resources.

**New Endpoint**: `POST /api/v1/jobs/{job_id}/cancel`

**Example Usage**:
```bash
# Cancel a running job
curl -X POST "http://localhost:18011/api/v1/jobs/{job_id}/cancel" \
  -H "Authorization: Bearer $API_KEY"

# Response
{
  "id": "abc-123",
  "status": "cancelled",
  "error_message": "Job cancelled by user request",
  "completed_at": "2025-10-27T12:34:56Z"
}
```

**Use Cases**:
- Stop stuck jobs consuming GPU memory
- Cancel incorrect configurations discovered after submission
- Manage resource allocation in multi-user labs
- Emergency shutdown of problematic processing

**Documentation**: See [API Documentation](http://localhost:18011/docs#/Jobs/cancel_job_api_v1_jobs__job_id__cancel_post)

---

### 2. ✅ Configuration Validation API (Priority 1)

**New Capability**: Validate pipeline configurations **before** submitting jobs.

**New Endpoints**:
- `POST /api/v1/config/validate` - Validate full job configuration
- `POST /api/v1/pipelines/{name}/validate` - Validate single pipeline config

**Example Usage**:
```bash
# Validate configuration before submission
curl -X POST "http://localhost:18011/api/v1/config/validate" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "person_tracking": {
        "confidence_threshold": 1.5
      }
    },
    "selected_pipelines": ["person_tracking"]
  }'

# Response shows validation errors
{
  "valid": false,
  "errors": [
    {
      "field": "person_tracking.confidence_threshold",
      "message": "Value must be between 0.0 and 1.0",
      "code": "VALUE_OUT_OF_RANGE",
      "hint": "Try using 0.5 as a sensible default"
    }
  ],
  "warnings": [],
  "pipelines_validated": ["person_tracking"]
}
```

**Benefits**:
- Catch configuration errors immediately (no wasted processing time)
- Get clear, actionable error messages with hints
- Test complex configurations before submission
- Integrate validation into your CI/CD pipelines

**Documentation**: See [Configuration Validation Guide](../usage/configuration_validation.md)

---

### 3. 🔒 Secure-by-Default Configuration (Priority 2)

**Change**: Authentication is now **enabled by default** on fresh installations.

**What This Means**:
- First server startup auto-generates an API key (printed to console)
- All API requests (except health checks) require authentication
- CORS is restricted to `http://localhost:3000` by default

**For New Installations**:
1. Start server: `uv run videoannotator server`
2. Copy the generated API key from console output
3. Use it in requests: `-H "Authorization: Bearer va_api_..."`

**For Existing Installations**:
- ✅ No change if you're already using API keys
- ✅ Existing API keys continue working

**For Local Development** (disable auth):
```bash
export VIDEOANNOTATOR_REQUIRE_AUTH=false
uv run videoannotator server
# Server logs warning: "[WARNING] Authentication DISABLED..."
```

**Documentation**: See [Authentication Guide](../security/authentication.md)

---

### 4. 📊 Enhanced System Diagnostics (Priority 2)

**New CLI Commands**:

```bash
# Run comprehensive system diagnostics
uv run videoannotator diagnose

# Check specific components
uv run videoannotator diagnose system    # System info
uv run videoannotator diagnose gpu       # GPU availability
uv run videoannotator diagnose storage   # Storage health
uv run videoannotator diagnose database  # Database status

# Get JSON output for monitoring/scripting
uv run videoannotator diagnose --json
```

**What It Checks**:
- Python version and platform
- GPU availability and CUDA version
- Storage paths and disk space
- Database connectivity and schema version
- FFmpeg installation

**Use Cases**:
- Troubleshoot installation issues
- Monitor system health in production
- Integrate with monitoring tools (Prometheus, Grafana)
- Debug resource problems

**Documentation**: See [Diagnostics Guide](../usage/diagnostics.md)

---

### 5. 🧹 Storage Cleanup (Optional Feature)

**New Capability**: Automatically clean up old job storage based on retention policy.

**New CLI Command**:
```bash
# Preview cleanup (dry-run)
uv run videoannotator storage-cleanup

# Actually delete old jobs
uv run videoannotator storage-cleanup --force

# Custom retention period (default: 30 days)
uv run videoannotator storage-cleanup --retention-days 90 --force
```

**Configuration**:
```bash
# Set retention policy via environment variable
export STORAGE_RETENTION_DAYS=60
```

**Safety Features**:
- Never deletes RUNNING or PENDING jobs
- Dry-run mode by default (preview before deleting)
- Detailed logging of all cleanup operations
- Skip jobs with missing metadata

**Documentation**: See [Storage Management Guide](../usage/storage_management.md)

---

## API Changes: Backward Compatibility

### ✅ Fully Backward Compatible

The following changes enhance existing functionality **without breaking existing clients**:

1. **Error Response Format** - Enhanced but still compatible
   - All errors now use consistent `ErrorEnvelope` format
   - Existing error parsers will continue working
   - New clients get better field-level error details

2. **Health Endpoint** - Additional fields added
   - `GET /health` - Still returns `status` and `message`
   - **New**: Added GPU status, worker info, diagnostic details
   - Existing health checks unaffected

3. **Job Status Response** - New optional fields
   - Existing fields unchanged: `id`, `status`, `created_at`, etc.
   - **New**: `storage_path` field (for internal tracking)
   - **New**: `retry_count` field (for monitoring)

### ⚠️ Authentication Now Required by Default

**Impact**: If you're setting up a **new** server instance, API requests now require authentication.

**Migration**:
- **Existing installations**: No change needed (keys already in use)
- **New installations**: Use the auto-generated API key or set `VIDEOANNOTATOR_REQUIRE_AUTH=false`
- **CI/CD**: Update scripts to use authentication or set env variable

**Documentation**: [Authentication Guide](../security/authentication.md)

---

## Infrastructure Improvements (Transparent to Clients)

These changes improve reliability but don't require client code updates:

### ✅ Data Persistence
- Videos and results persist across server restarts
- Job metadata stored in SQLite database (default: `./videoannotator.db`)
- Configurable storage location via `STORAGE_ROOT` environment variable

### ✅ Worker Enhancements
- Automatic retry with exponential backoff (up to 3 retries)
- Graceful cancellation of running jobs
- Better error messages and logging
- Memory leak fixes

### ✅ Error Handling
- Consistent error format across all endpoints
- Structured error codes for programmatic handling
- Helpful hints in error messages
- Proper HTTP status codes

### ✅ Database Migrations
- Automatic schema updates on server startup
- No manual migration steps required
- Backward-compatible schema changes

---

## Updated Documentation

### For API Users
- ✅ **[Getting Started Guide](../usage/GETTING_STARTED.md)** - Updated with v1.3.0 features
- ✅ **[API Documentation](http://localhost:18011/docs)** - Interactive Swagger UI with examples
- ✅ **[Authentication Guide](../security/authentication.md)** - Complete auth setup
- ✅ **[Configuration Validation](../usage/configuration_validation.md)** - NEW

### For Troubleshooting
- ✅ **[Troubleshooting Guide](../installation/troubleshooting.md)** - NEW: Comprehensive issue resolution
- ✅ **[Diagnostics Guide](../usage/diagnostics.md)** - NEW: System health checks
- ✅ **[Installation Guide](../installation/INSTALLATION.md)** - Updated with verification steps

### For Developers
- ✅ **[Upgrade Guide](../UPGRADING_TO_v1.3.0.md)** - Migration from v1.2.x (if extending the codebase)
- ✅ **[Testing Overview](../testing/testing_overview.md)** - Coverage reports and testing strategy

---

## Testing & Validation

### Recommended Testing Steps

1. **Install v1.3.0**:
   ```bash
   git checkout 001-videoannotator-v1-3
   uv sync
   ```

2. **Run System Diagnostics**:
   ```bash
   uv run videoannotator diagnose
   ```

3. **Start Server**:
   ```bash
   uv run videoannotator server
   # Copy the API key from console output
   export API_KEY="va_api_..."
   ```

4. **Test Job Submission**:
   ```bash
   # Submit a test job
   curl -X POST "http://localhost:18011/api/v1/jobs/" \
     -H "Authorization: Bearer $API_KEY" \
     -F "video=@test.mp4" \
     -F "selected_pipelines=scene_detection"
   ```

5. **Test Configuration Validation**:
   ```bash
   # Validate a configuration
   curl -X POST "http://localhost:18011/api/v1/config/validate" \
     -H "Authorization: Bearer $API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"config": {"scene_detection": {"threshold": 30.0}}, "selected_pipelines": ["scene_detection"]}'
   ```

6. **Test Job Cancellation**:
   ```bash
   # Cancel a job
   curl -X POST "http://localhost:18011/api/v1/jobs/{job_id}/cancel" \
     -H "Authorization: Bearer $API_KEY"
   ```

### Test Coverage

- **234 tests passing** across 56 completed tasks
- **Modules tested**: API (90%+), Storage (85%+), Validation (100%), Workers, Database
- **Integration tests**: Job lifecycle, cancellation, validation, storage persistence

---

## Breaking Changes

### ⚠️ None for API Clients

This release is **backward compatible** for API users. No breaking changes to existing endpoints or response formats.

### ⚠️ Internal Import Paths (Developers Only)

If you're **extending VideoAnnotator** or running custom scripts that import internal modules:

**Before (v1.2.x)**:
```python
from api.main import app
from storage.sqlite_backend import SQLiteStorageBackend
```

**After (v1.3.0)**:
```python
from videoannotator.api.main import app
from videoannotator.storage.sqlite_backend import SQLiteStorageBackend
```

**Migration**: See [Upgrade Guide](../UPGRADING_TO_v1.3.0.md) for complete details.

**Impact**: Does not affect API clients—only developers extending the codebase.

---

## Environment Variables

### New Configuration Options

```bash
# Authentication (default: true)
export VIDEOANNOTATOR_REQUIRE_AUTH=true

# Storage retention for cleanup (default: 30 days)
export STORAGE_RETENTION_DAYS=30

# Custom storage location (default: ./storage)
export STORAGE_ROOT=/path/to/storage

# Worker concurrency (default: 2)
export MAX_CONCURRENT_JOBS=4

# Worker poll interval (default: 5 seconds)
export WORKER_POLL_INTERVAL=10
```

### Existing Options (Unchanged)
```bash
export DATABASE_PATH=./custom.db
export LOG_LEVEL=DEBUG
export CORS_ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001
```

---

## Known Issues & Limitations

1. **Storage Cleanup** - Optional feature, disabled by default
   - Must manually run `storage-cleanup` command
   - Consider adding to cron job for automated cleanup

2. **Configuration Validation** - Does not validate model availability
   - Checks config structure and value ranges
   - Does not verify if model weights are downloaded

3. **Job Cancellation** - May take up to 5 seconds
   - Graceful shutdown waits for current pipeline stage to complete
   - Use `diagnose` command if jobs appear stuck

---

## Next Steps

### For Production Deployment

1. **Review Security Settings**:
   - Confirm authentication is enabled
   - Configure CORS allowed origins
   - Set up HTTPS/TLS if exposing publicly

2. **Configure Storage**:
   - Set `STORAGE_ROOT` to persistent storage location
   - Configure retention policy with `STORAGE_RETENTION_DAYS`
   - Set up automated cleanup (cron job)

3. **Monitor System Health**:
   - Integrate `/api/v1/system/health` with monitoring tools
   - Use `diagnose` command for troubleshooting
   - Review logs regularly (`logs/` directory)

4. **Test Job Workflows**:
   - Submit test jobs with various pipeline configurations
   - Test cancellation under load
   - Verify validation catches your common config errors

### For Development/Integration

1. **Update API Client Code**:
   - Add support for new `/cancel` endpoint (optional)
   - Integrate `/config/validate` before job submission (recommended)
   - Update authentication headers (if new installation)

2. **Review API Documentation**:
   - Interactive docs: http://localhost:18011/docs
   - Review new endpoint examples and error responses

3. **Test Error Handling**:
   - Verify your error parser handles new `ErrorEnvelope` format
   - Test handling of validation errors with field-level details

---

## Getting Help

- **Documentation**: Start with [Getting Started Guide](../usage/GETTING_STARTED.md)
- **Troubleshooting**: See [Troubleshooting Guide](../installation/troubleshooting.md)
- **API Reference**: http://localhost:18011/docs (when server is running)
- **Issues**: Report bugs on GitHub Issues
- **Questions**: GitHub Discussions

---

## Summary: Action Items by User Type

### 🔬 Researchers (API Users)
- ✅ **No action required** - Update to v1.3.0 and existing workflows continue working
- 📋 **Recommended**: Try new job cancellation and config validation features
- 🔒 **New installations**: Save the auto-generated API key on first startup

### 👨‍💻 Integration Developers
- ✅ **Review**: New endpoints for cancellation and validation
- ✅ **Test**: Error handling with new consistent format
- ✅ **Update**: Authentication headers (if new installation)

### 🏢 Lab Administrators
- ✅ **Configure**: Authentication, CORS, storage location
- ✅ **Monitor**: Use diagnostic commands and health endpoint
- ✅ **Automate**: Set up storage cleanup cron job (optional)

### 🔧 Code Contributors
- ⚠️ **Required**: Update import paths (`videoannotator.*` namespace)
- ✅ **Review**: [Upgrade Guide](../UPGRADING_TO_v1.3.0.md) for migration details

---

**Questions or Issues?** Contact the development team or open a GitHub issue.

**Release Date**: TBD (currently in testing on feature branch `001-videoannotator-v1-3`)
