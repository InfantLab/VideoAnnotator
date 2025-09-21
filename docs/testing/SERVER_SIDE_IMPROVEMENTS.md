# VideoAnnotator Server-Side Improvements Needed

## 🔴 **CRITICAL SERVER ISSUES**

### **1. SSE Endpoint Missing**
**Issue**: `/api/v1/events/stream` returns 404 Not Found
**Impact**: Real-time job monitoring completely broken
**Client Error**: 
```
:8000/api/v1/events/stream?token=dev-token:1 Failed to load resource: the server responded with a status of 404 (Not Found)
SSE Error: Error: SSE connection failed after maximum retry attempts
```
**Required**: Implement Server-Sent Events endpoint for job progress updates

### **2. Health Endpoint Inconsistency** 
**Issue**: `/api/v1/system/health` initially returns 404, then works
**Impact**: Unreliable health checking, confusing API status
**Required**: Fix health endpoint reliability and ensure consistent responses

ser
## 🟠 **HIGH PRIORITY SERVER ENHANCEMENTS**
 
### **3. Pipeline Information API**
**Issue**: Basic pipeline selection with no detailed configuration options
**Current State**: Client hardcodes pipeline descriptions and parameters
**Required**: 
- `/api/v1/pipelines` should return detailed pipeline information
- Each pipeline needs: name, description, parameters, sub-options, defaults
- Face pipeline should list all analysis options (age, gender, emotion, etc.)
- Audio pipeline should separate PyAnnote, Whisper, and LAION components

### **4. Enhanced Job Configuration Support**
**Issue**: No server support for database location and output directory preferences
**Required**:
- Accept `db_location` and `output_directory` parameters in job submission
- Allow users to specify custom storage locations
- Validate and sanitize file paths for security

### **5. Authentication Feedback**
**Issue**: No clear indication when API token authentication fails
**Required**:
- Return proper HTTP status codes for authentication failures
- Provide clear error messages for token issues
- Consider token validation endpoint

## 🟡 **MEDIUM PRIORITY SERVER FEATURES**

### **6. Configuration Templates/Presets**
**Issue**: Users need better configuration management
**Required**:
- Server-side configuration presets storage
- User preference persistence
- Default configuration retrieval per pipeline

### **7. Batch Job Management**
**Issue**: No native server-side batch processing
**Enhancement**: 
- Consider server-side batch job handling
- Single submission with multiple videos
- Batch progress tracking

### **8. Job Artifacts Management**
**Issue**: No clear artifact retrieval system mentioned in current API
**Required**:
- `/api/v1/jobs/{id}/artifacts` endpoint for file listing
- Direct download links for completed annotation files
- Proper CORS headers for artifact access

## 🔵 **API SPECIFICATION IMPROVEMENTS**

### **9. OpenAPI Documentation Gaps**
**Issues Identified**:
- SSE endpoint not documented in OpenAPI schema
- Authentication requirements unclear
- Response formats for some endpoints incomplete

### **10. Error Response Standardization**
**Required**:
- Consistent error response format across all endpoints
- Proper HTTP status codes
- User-friendly error messages with actionable guidance

---

## 📋 **IMPLEMENTATION PRIORITY ORDER**

### **Sprint 1 (Critical - Fix Now)**
1. **SSE Endpoint Implementation**: `/api/v1/events/stream` with proper job event streaming
2. **Health Endpoint Fix**: Reliable `/api/v1/system/health` responses
3. **Authentication Error Handling**: Clear feedback for token failures

### **Sprint 2 (High Priority)**
4. **Enhanced Pipeline API**: Detailed pipeline information with all options
5. **Job Configuration**: Database and output directory parameters
6. **Job Artifacts**: File retrieval system for completed jobs

### **Sprint 3 (Medium Priority)**
7. **Configuration Presets**: Server-side preference storage
8. **Batch Processing**: Native server-side batch job handling
9. **API Documentation**: Complete OpenAPI specification updates

---

## 🔧 **TECHNICAL SPECIFICATIONS**

### **SSE Event Format**
Expected event types that client is ready to handle:
```javascript
// Job status updates
{ type: 'job.update', data: { job_id, status, progress, timestamp } }

// Log messages
{ type: 'job.log', data: { job_id, level, message, timestamp } }

// Job completion
{ type: 'job.complete', data: { job_id, artifacts, completion_time } }

// Job errors
{ type: 'job.error', data: { job_id, error_message, error_code } }
```

### **Pipeline API Enhancement**
Expected structure for `/api/v1/pipelines`:
```json
{
  "pipelines": [
    {
      "name": "face_analysis",
      "display_name": "Face Analysis",
      "description": "Comprehensive facial analysis pipeline",
      "enabled": true,
      "components": [
        {
          "name": "openface3",
          "display_name": "OpenFace 3.0",
          "description": "Facial landmark detection and expression analysis",
          "parameters": {
            "confidence_threshold": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0}
          }
        },
        {
          "name": "age_estimation",
          "display_name": "Age Estimation",
          "description": "Deep learning age prediction",
          "enabled": false
        },
        {
          "name": "emotion_recognition", 
          "display_name": "Emotion Recognition",
          "description": "Facial emotion classification",
          "enabled": false
        }
      ]
    }
  ]
}
```

---

## ✅ **CLIENT-SIDE COMPATIBILITY**

The current client implementation is **already prepared** for these server-side improvements:
- SSE connection and reconnection logic implemented
- Error handling for missing endpoints
- Pipeline configuration UI ready for dynamic data
- Batch job submission logic implemented
- Artifact retrieval hooks in place

Once server-side changes are implemented, client should work without major modifications.

---

**Document Version**: v0.3.0  
**Last Updated**: 2025-08-24  
**Status**: Awaiting server-side implementation