# 🤝 Client-Server Testing Collaboration Guide

## Overview

# 🤝 Client-Server Testing Collaboration Guide

## Overview

This guide provides client-side developers with tools and protocols for effective collaboration with the VideoAnnotator server team during API integration and testing.

**Target Server**: VideoAnnotator v1.2.x / v1.3.x API Server  
**Client Application**: Video Annotation Viewer (React + TypeScript)  
**Latest Supported Version**: v1.3.0

---

## 🆕 VideoAnnotator v1.3.0 Features

### **New Endpoints**

#### **Job Cancellation**
```http
POST /api/v1/jobs/{job_id}/cancel
Content-Type: application/json
Authorization: Bearer {token}

{
  "reason": "User requested cancellation"  # Optional
}
```

**Response**:
```json
{
  "job_id": "string",
  "status": "cancelled" | "cancelling",
  "message": "Job cancelled successfully",
  "cancelled_at": "2025-10-29T10:00:00Z"
}
```

#### **Configuration Validation**
```http
POST /api/v1/validate/config
Content-Type: application/json
Authorization: Bearer {token}

{
  "config": {
    "pipelines": ["openface3", "whisper"],
    "openface3": { "model": "mobilenet" },
    "whisper": { "model": "base" }
  }
}
```

**Response**:
```json
{
  "valid": true,
  "errors": [],
  "warnings": [
    {
      "field": "whisper.model",
      "message": "Model 'base' may not provide optimal accuracy",
      "hint": "Consider using 'medium' or 'large' for better results"
    }
  ]
}
```

#### **Pipeline-Specific Validation**
```http
POST /api/v1/validate/pipeline/{pipeline_id}
Content-Type: application/json
Authorization: Bearer {token}

{
  "config": {
    "model": "mobilenet",
    "confidence_threshold": 0.5
  }
}
```

#### **Enhanced Health Endpoint**
```http
GET /api/v1/system/health
Authorization: Bearer {token}
```

**Response (v1.3.0)**:
```json
{
  "status": "healthy" | "degraded" | "unhealthy",
  "version": "1.3.0",
  "api_version": "1.3.0",
  "uptime_seconds": 3600,
  "gpu_status": {
    "available": true,
    "device_name": "NVIDIA GeForce RTX 3080",
    "cuda_version": "11.8",
    "memory_total": 10737418240,
    "memory_used": 5368709120,
    "memory_free": 5368709120
  },
  "worker_info": {
    "active_jobs": 2,
    "queued_jobs": 3,
    "max_concurrent_jobs": 4,
    "worker_status": "idle" | "busy" | "overloaded"
  },
  "diagnostics": {
    "database": {
      "status": "healthy" | "degraded" | "unhealthy",
      "message": "All systems operational"
    },
    "storage": {
      "status": "healthy",
      "disk_usage_percent": 45
    },
    "ffmpeg": {
      "available": true,
      "version": "5.1.2"
    }
  }
}
```

**Fallback (v1.2.x)**:
```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "api_version": "1.2.1",
  "videoannotator_version": "1.2.1",
  "message": "API is running"
}
```

### **Client Implementation**

#### **Feature Detection**
```typescript
import { apiClient } from '@/api/client';

// Detect v1.3.0 features
const health = await apiClient.getEnhancedHealth();
const hasJobCancellation = !!health.version && health.version >= '1.3.0';
const hasValidation = !!health.version && health.version >= '1.3.0';
const hasAdvancedDiagnostics = !!health.gpu_status || !!health.worker_info;
```

#### **Job Cancellation Hook**
```typescript
import { useJobCancellation } from '@/hooks/useJobCancellation';

function JobDetailPage() {
  const { cancelJob, isCancelling, canCancel } = useJobCancellation(jobId);
  
  if (!canCancel) return null; // Feature not available
  
  return (
    <button onClick={() => cancelJob('User requested')} disabled={isCancelling}>
      {isCancelling ? 'Cancelling...' : 'Cancel Job'}
    </button>
  );
}
```

#### **Configuration Validation Hook**
```typescript
import { useConfigValidation } from '@/hooks/useConfigValidation';

function JobCreationWizard() {
  const { validateConfig, validation, isValidating } = useConfigValidation();
  
  const handleConfigChange = (newConfig) => {
    setConfig(newConfig);
    validateConfig(newConfig); // Debounced validation
  };
  
  return (
    <>
      {validation.errors.map(error => (
        <ErrorMessage key={error.field}>
          {error.message}
          {error.hint && <Hint>{error.hint}</Hint>}
        </ErrorMessage>
      ))}
    </>
  );
}
```

#### **Server Diagnostics Component**
```typescript
import { ServerDiagnostics } from '@/components/ServerDiagnostics';

function SettingsPage() {
  return (
    <>
      {/* Other settings */}
      <ServerDiagnostics defaultOpen={false} />
    </>
  );
}
```

---

## 🎯 Quick Start for Client Developers

### **1. Test Server Connectivity**
```bash
# Test any VideoAnnotator server instance
python scripts/test_api_quick.py http://localhost:18011 dev-token
python scripts/test_api_quick.py https://your-server.com your-api-token
```

### **2. Browser Debug Console**
```javascript
// Copy scripts/browser_debug_console.js and paste into browser console
VideoAnnotatorDebug.runAllTests()
VideoAnnotatorDebug.checkHealth()  
VideoAnnotatorDebug.monitorJob('job-123')
```

### **3. Monitor API Issues**
- **Server Status**: `/api/v1/debug/server-info`
- **Authentication**: `/api/v1/debug/token-info`
- **Request History**: `/api/v1/debug/request-log`
- **Pipeline Status**: `/api/v1/debug/pipelines`

---

## 🔧 Available Debug Tools

### **Python API Tester** (`scripts/test_api_quick.py`)
Comprehensive automated testing for VideoAnnotator API endpoints:

- **Health Checks**: Basic and detailed server health
- **Authentication**: Token validation and permissions
- **Pipeline Testing**: Available pipelines and status
- **Job Management**: Submission, status tracking, retrieval
- **SSE Testing**: Server-Sent Events connection testing
- **Missing Endpoints**: Detection of unimplemented features

**Usage Examples**:
```bash
# Default local testing
python scripts/test_api_quick.py

# Custom server and token
python scripts/test_api_quick.py https://staging-api.example.com staging-token

# Test specific environment
python scripts/test_api_quick.py http://docker-container:18011 dev-token
```

**Output**: Detailed test results saved to `test_results_api.json`

### **Browser Debug Console** (`scripts/browser_debug_console.js`)
Interactive debugging tools for browser-based development:

**Key Functions**:
- `checkHealth()` - Test API connectivity
- `checkAuth(token)` - Validate authentication  
- `getServerInfo()` - Comprehensive server information
- `checkPipelines()` - Pipeline availability and status
- `submitTestJob()` - Submit mock job for testing
- `monitorJob(jobId)` - Real-time job progress monitoring
- `testSSE(jobId)` - Server-Sent Events testing
- `runAllTests()` - Complete test suite

**Browser Integration**:
```javascript
// Enable request logging
VideoAnnotatorDebug.enableRequestLogging()

// Monitor specific job
VideoAnnotatorDebug.monitorJob('your-job-id-here')

// Test SSE connection
VideoAnnotatorDebug.testSSE()
```

---

## 📋 API Issue Reporting Template

Use this template when reporting server integration issues:

```markdown
## 🐛 API Integration Issue

**Reporter**: [Your Name / Team]
**Date**: [YYYY-MM-DD]
**Priority**: [Critical/High/Medium/Low]
**Component**: [Authentication/Jobs/Pipelines/SSE/Other]

### Issue Description
[Clear description of the integration problem]

### Environment
- Server URL: [e.g., http://localhost:18011]
- Client Version: [from package.json]
- Browser: [Chrome/Firefox/Safari + version]
- API Version: [from /health endpoint]
- Server Version: [from /api/v1/debug/server-info]

### Reproduction Steps
1. [Step 1]
2. [Step 2]
3. [Step 3]

### Expected vs Actual Behavior
**Expected**: [What should happen]
**Actual**: [What actually happens]

### Debug Information
```javascript
// Browser console output
VideoAnnotatorDebug.runAllTests()
```

**Python test results**:
```bash
python scripts/test_api_quick.py [your-server] [your-token]
```

### Client-Side Logs
[Any console errors, network tab info, or relevant client logs]

### Additional Context
[Screenshots, network timing, etc.]
```

---

## 🚀 Integration Testing Workflow

### **Pre-Development Setup**
1. **Install Dependencies**: Ensure Python 3.8+ available
2. **Test Server Connection**: Run `python scripts/test_api_quick.py`
3. **Verify Authentication**: Check your API token works
4. **Browser Setup**: Load debug console script

### **During Development**
1. **API Changes**: Test endpoints before integration
2. **Error Handling**: Use debug endpoints to understand server state
3. **Real-time Monitoring**: Use SSE testing for event-driven features
4. **Job Testing**: Submit test jobs to verify processing pipeline

### **Pre-Deployment**
1. **Full Test Suite**: Run comprehensive API tests
2. **Authentication Verification**: Test with production tokens
3. **Performance Testing**: Monitor response times and error rates
4. **SSE Stability**: Verify real-time connection reliability

---

## 🔍 Common Integration Patterns

### **API Client Setup**
```typescript
// Example API client configuration
const apiClient = new VideoAnnotatorAPI({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:18011',
  token: localStorage.getItem('api_token') || 'dev-token',
  timeout: 30000
});
```

### **Error Handling**
```typescript
// Standardized error handling
try {
  const response = await apiClient.submitJob(videoFile, pipelines);
} catch (error) {
  if (error.status === 401) {
    // Authentication issue
    console.log('Check /api/v1/debug/token-info');
  } else if (error.status === 500) {
    // Server issue  
    console.log('Check /api/v1/debug/server-info');
  }
}
```

### **SSE Integration**
```typescript
// Server-Sent Events setup
const eventSource = new EventSource(
  `${apiUrl}/api/v1/events/stream?token=${token}&job_id=${jobId}`
);

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // Handle job progress updates
};
```

---

## ⚡ Emergency Response

### **Critical Issues**
1. **Server Unavailable**: Check health endpoints first
2. **Authentication Failures**: Verify token with debug endpoints
3. **Job Processing Errors**: Use job debug endpoint for details
4. **SSE Connection Lost**: Test SSE endpoint availability

### **Client-Side Fallbacks**
- **Offline Mode**: Use cached/demo data
- **Polling Fallback**: Replace SSE with HTTP polling
- **Error Boundaries**: Graceful degradation for API failures
- **Retry Logic**: Exponential backoff for transient errors

---

## 📊 Success Metrics

### **Integration Quality**
- **API Test Success Rate**: >95% of automated tests passing
- **Response Time**: <500ms for status endpoints
- **Authentication Success**: 100% with valid tokens
- **Job Submission**: >90% successful processing

### **Development Velocity** 
- **Issue Resolution**: <24 hours for API integration blockers
- **Debug Time**: <30 minutes to identify server-side issues
- **Testing Coverage**: All major API flows tested before deployment

---

## 🔧 Setup Instructions

### **Python Environment**
```bash
# Ensure Python 3.8+ is available
python --version

# No additional dependencies needed
# Scripts use only standard library
```

### **Browser Debug Setup**
```bash
# 1. Copy debug console script
cat scripts/browser_debug_console.js

# 2. Open browser developer tools (F12)
# 3. Paste entire script into Console tab
# 4. Use VideoAnnotatorDebug.* functions

# Example first steps:
VideoAnnotatorDebug.help()
VideoAnnotatorDebug.runAllTests()
```

### **CI Integration Example**
```yaml
name: API Integration Tests
on: [push, pull_request]
jobs:
  api-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Test API Integration
        run: python scripts/test_api_quick.py ${{ secrets.API_URL }} ${{ secrets.API_TOKEN }}
```

---

## 📞 Getting Help

1. **Debug Tools**: Use provided scripts first
2. **Server Team**: Share debug output from tools  
3. **Issue Reports**: Use the template above
4. **Emergency**: Include full test results in critical issues

---

**Document Version**: v2.0  
**Last Updated**: October 2025  
**VideoAnnotator Compatibility**: v1.2.0+ (Full backward compatibility, v1.3.0+ for enhanced features)  
**Status**: Active development collaboration tool