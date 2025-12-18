# 🔧 VideoAnnotator Client-Server Debug Tools & Collaboration Guide

## Overview

This document provides debugging tools, testing utilities, and collaboration guidelines for both client-side and server-side development teams working on VideoAnnotator.

---

## 🚀 Server-Side Debug Tools for Client Developers

### **1. Debug API Endpoints**

#### **Server Information Endpoint**

```
GET /api/v1/debug/server-info
```

**Purpose**: Get detailed server configuration and status
**Response**:

```json
{
  "server": {
    "version": "1.2.2",
    "environment": "development",
    "start_time": "2025-01-23T10:30:00Z",
    "uptime_seconds": 3600,
    "debug_mode": true
  },
  "database": {
    "backend": "sqlite",
    "path": "/path/to/videoannotator.db",
    "connection_status": "healthy",
    "total_jobs": 15,
    "active_connections": 3
  },
  "pipelines": {
    "initialized": ["person_tracking", "face_analysis", "scene_detection"],
    "available": [
      "person_tracking",
      "face_analysis",
      "audio_processing",
      "scene_detection"
    ],
    "loading_errors": []
  },
  "system": {
    "gpu_available": true,
    "gpu_memory_used": "2.1GB",
    "cpu_usage": 25.5,
    "memory_usage": 45.2
  }
}
```

#### **API Token Debug Endpoint**

```
GET /api/v1/debug/token-info
Authorization: Bearer {your-token}
```

**Purpose**: Validate and inspect API token details
**Response**:

```json
{
  "token": {
    "valid": true,
    "user_id": "test-user-123",
    "permissions": ["job:submit", "job:read", "job:delete"],
    "expires_at": "2025-02-23T10:30:00Z",
    "rate_limit": {
      "requests_per_minute": 100,
      "remaining": 87,
      "reset_at": "2025-01-23T10:31:00Z"
    }
  }
}
```

#### **Pipeline Debug Endpoint**

```
GET /api/v1/debug/pipelines
```

**Purpose**: Get detailed pipeline configuration and status
**Response**:

```json
{
  "pipelines": [
    {
      "name": "face_analysis",
      "display_name": "Face Analysis",
      "status": "ready",
      "components": [
        {
          "name": "openface3",
          "enabled": true,
          "model_loaded": true,
          "model_path": "/models/openface/openface_3.0.pt",
          "parameters": {
            "confidence_threshold": 0.5,
            "enable_age_estimation": false,
            "enable_emotion_recognition": true,
            "supported_emotions": [
              "happy",
              "sad",
              "angry",
              "surprised",
              "neutral"
            ]
          }
        },
        {
          "name": "laion_face",
          "enabled": true,
          "model_loaded": true,
          "supported_analyses": ["age", "gender", "emotion", "attractiveness"]
        }
      ]
    }
  ]
}
```

#### **Job Debug Endpoint**

```
GET /api/v1/debug/jobs/{job_id}
Authorization: Bearer {your-token}
```

**Purpose**: Get detailed job debugging information
**Response**:

```json
{
  "job": {
    "id": "job_123",
    "status": "processing",
    "created_at": "2025-01-23T10:00:00Z",
    "debug_info": {
      "queue_wait_time_ms": 1500,
      "processing_start_time": "2025-01-23T10:00:15Z",
      "current_pipeline": "face_analysis",
      "progress_percentage": 45.2,
      "estimated_completion": "2025-01-23T10:05:00Z",
      "resource_usage": {
        "gpu_memory_mb": 1200,
        "cpu_usage": 85.3,
        "processing_fps": 2.1
      }
    },
    "pipeline_logs": [
      {
        "timestamp": "10:00:15",
        "level": "INFO",
        "message": "Starting face analysis pipeline"
      },
      {
        "timestamp": "10:01:30",
        "level": "DEBUG",
        "message": "Processing frame 125/300"
      },
      {
        "timestamp": "10:02:45",
        "level": "WARN",
        "message": "Low confidence face detection on frame 180"
      }
    ],
    "errors": [],
    "files": {
      "input_video": "/uploads/video_job_123.mp4",
      "output_directory": "/results/job_123/",
      "temp_files": ["/tmp/job_123_frames/", "/tmp/job_123_audio.wav"]
    }
  }
}
```

### **2. Development Mode Features**

#### **Mock SSE Endpoint** (Until real SSE is implemented)

```javascript
// JavaScript client code for testing
const eventSource = new EventSource(
  "/api/v1/debug/mock-events?token=dev-token&job_id=123",
);

eventSource.onmessage = function (event) {
  const data = JSON.parse(event.data);
  console.log("Mock Event:", data);

  // Expected event format from SERVER_SIDE_IMPROVEMENTS.md:
  // { type: 'job.update', data: { job_id, status, progress, timestamp } }
  // { type: 'job.log', data: { job_id, level, message, timestamp } }
  // { type: 'job.complete', data: { job_id, artifacts, completion_time } }
  // { type: 'job.error', data: { job_id, error_message, error_code } }
};
```

#### **Request Logging**

```
GET /api/v1/debug/request-log
Authorization: Bearer {your-token}
```

**Purpose**: View recent API requests for debugging
**Response**:

```json
{
  "requests": [
    {
      "timestamp": "2025-01-23T10:30:00Z",
      "method": "POST",
      "path": "/api/v1/jobs",
      "status_code": 201,
      "response_time_ms": 1250,
      "user_id": "test-user-123",
      "request_id": "req_abc123"
    },
    {
      "timestamp": "2025-01-23T10:29:30Z",
      "method": "GET",
      "path": "/api/v1/jobs/job_123",
      "status_code": 200,
      "response_time_ms": 45,
      "user_id": "test-user-123",
      "request_id": "req_def456"
    }
  ]
}
```

---

## 🧪 Client-Side Testing Tools

### **1. API Testing Utilities**

#### **JavaScript API Test Library**

```javascript
// api-test-utils.js - For client-side developers
class VideoAnnotatorAPITester {
  constructor(baseUrl, apiToken) {
    this.baseUrl = baseUrl;
    this.apiToken = apiToken;
    this.headers = {
      Authorization: `Bearer ${apiToken}`,
      "Content-Type": "application/json",
    };
  }

  // Test server connectivity and health
  async testServerHealth() {
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      const healthData = await response.json();

      console.log("✅ Basic Health Check:", response.status === 200);

      // Test detailed health endpoint
      const detailedResponse = await fetch(
        `${this.baseUrl}/api/v1/system/health`,
      );
      const detailedHealth = await detailedResponse.json();

      console.log("✅ Detailed Health Check:", detailedResponse.status === 200);
      console.log(
        "Database Status:",
        detailedHealth.services?.database?.status,
      );

      return { basic: healthData, detailed: detailedHealth };
    } catch (error) {
      console.error("❌ Health Check Failed:", error);
      return null;
    }
  }

  // Test authentication
  async testAuthentication() {
    try {
      const response = await fetch(`${this.baseUrl}/api/v1/debug/token-info`, {
        headers: this.headers,
      });
      const tokenInfo = await response.json();

      console.log("✅ Authentication Test:", response.status === 200);
      console.log("Token Valid:", tokenInfo.token?.valid);
      console.log("Permissions:", tokenInfo.token?.permissions);

      return tokenInfo;
    } catch (error) {
      console.error("❌ Authentication Test Failed:", error);
      return null;
    }
  }

  // Test pipeline information
  async testPipelineAPI() {
    try {
      // Test basic pipeline list
      const basicResponse = await fetch(`${this.baseUrl}/api/v1/pipelines`);
      const basicPipelines = await basicResponse.json();

      // Test detailed pipeline debug info
      const debugResponse = await fetch(
        `${this.baseUrl}/api/v1/debug/pipelines`,
      );
      const debugPipelines = await debugResponse.json();

      console.log("✅ Pipeline API Test:", basicResponse.status === 200);
      console.log(
        "Available Pipelines:",
        basicPipelines.pipelines?.length || 0,
      );
      console.log("Pipeline Details Available:", debugResponse.status === 200);

      return { basic: basicPipelines, debug: debugPipelines };
    } catch (error) {
      console.error("❌ Pipeline API Test Failed:", error);
      return null;
    }
  }

  // Test job submission workflow
  async testJobWorkflow() {
    try {
      // Create mock video file for testing
      const mockVideoBlob = new Blob(["fake video content"], {
        type: "video/mp4",
      });
      const formData = new FormData();
      formData.append("video", mockVideoBlob, "test-video.mp4");
      formData.append("selected_pipelines", "person_tracking,scene_detection");

      // Submit job
      const submitResponse = await fetch(`${this.baseUrl}/api/v1/jobs/`, {
        method: "POST",
        headers: { Authorization: `Bearer ${this.apiToken}` },
        body: formData,
      });

      if (submitResponse.status !== 201) {
        console.error("❌ Job Submission Failed:", await submitResponse.text());
        return null;
      }

      const jobData = await submitResponse.json();
      console.log("✅ Job Submitted:", jobData.id);

      // Check job status
      const statusResponse = await fetch(
        `${this.baseUrl}/api/v1/jobs/${jobData.id}`,
        {
          headers: this.headers,
        },
      );
      const statusData = await statusResponse.json();

      console.log("✅ Job Status Check:", statusResponse.status === 200);
      console.log("Job Status:", statusData.status);

      return { job: jobData, status: statusData };
    } catch (error) {
      console.error("❌ Job Workflow Test Failed:", error);
      return null;
    }
  }

  // Test SSE connection (once implemented)
  async testSSEConnection(jobId) {
    return new Promise((resolve) => {
      const eventSource = new EventSource(
        `${this.baseUrl}/api/v1/events/stream?token=${this.apiToken.replace(
          "Bearer ",
          "",
        )}&job_id=${jobId}`,
      );

      let eventCount = 0;
      const events = [];

      eventSource.onopen = () => {
        console.log("✅ SSE Connection Opened");
      };

      eventSource.onmessage = (event) => {
        eventCount++;
        const data = JSON.parse(event.data);
        events.push(data);
        console.log(`📡 SSE Event ${eventCount}:`, data);

        if (eventCount >= 3) {
          // Test first few events
          eventSource.close();
          resolve({ success: true, events });
        }
      };

      eventSource.onerror = (error) => {
        console.error("❌ SSE Connection Error:", error);
        eventSource.close();
        resolve({ success: false, error });
      };

      // Timeout after 10 seconds
      setTimeout(() => {
        eventSource.close();
        resolve({ success: eventCount > 0, events, timeout: true });
      }, 10000);
    });
  }

  // Run all tests
  async runAllTests() {
    console.log("🧪 Running VideoAnnotator API Tests...\n");

    const results = {
      serverHealth: await this.testServerHealth(),
      authentication: await this.testAuthentication(),
      pipelines: await this.testPipelineAPI(),
      jobWorkflow: await this.testJobWorkflow(),
    };

    console.log("\n📊 Test Results Summary:");
    console.log("- Server Health:", results.serverHealth ? "✅" : "❌");
    console.log("- Authentication:", results.authentication ? "✅" : "❌");
    console.log("- Pipeline API:", results.pipelines ? "✅" : "❌");
    console.log("- Job Workflow:", results.jobWorkflow ? "✅" : "❌");

    return results;
  }
}

// Usage example:
// const tester = new VideoAnnotatorAPITester('http://localhost:18011', 'dev-token');
// tester.runAllTests();
```

### **2. Command Line Testing Tools**

#### **Quick API Test Script**

Create file: `scripts/test_api_quick.py`

```python
#!/usr/bin/env python3
"""
Quick API testing script for client developers
Run: python scripts/test_api_quick.py
"""
import requests
import json
import time
from pathlib import Path

class APIQuickTester:
  def __init__(self, base_url="http://localhost:18011", token="dev-token"):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {token}"}
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def test_health_endpoints(self):
        """Test all health-related endpoints"""
        print("🏥 Testing Health Endpoints...")

        # Basic health
        try:
            resp = self.session.get(f"{self.base_url}/health", timeout=5)
            print(f"  Basic Health: {resp.status_code} ({'✅' if resp.status_code == 200 else '❌'})")
        except Exception as e:
            print(f"  Basic Health: ❌ ({e})")

        # Detailed health
        try:
            resp = self.session.get(f"{self.base_url}/api/v1/system/health", timeout=5)
            print(f"  Detailed Health: {resp.status_code} ({'✅' if resp.status_code == 200 else '❌'})")
            if resp.status_code == 200:
                data = resp.json()
                print(f"    Status: {data.get('status', 'unknown')}")
                print(f"    Database: {data.get('services', {}).get('database', {}).get('status', 'unknown')}")
        except Exception as e:
            print(f"  Detailed Health: ❌ ({e})")

    def test_authentication(self):
        """Test authentication and token validation"""
        print("\n🔐 Testing Authentication...")

        try:
            resp = self.session.get(f"{self.base_url}/api/v1/debug/token-info", timeout=5)
            print(f"  Token Validation: {resp.status_code} ({'✅' if resp.status_code == 200 else '❌'})")
            if resp.status_code == 200:
                data = resp.json()
                print(f"    Token Valid: {data.get('token', {}).get('valid', False)}")
                print(f"    Permissions: {data.get('token', {}).get('permissions', [])}")
        except Exception as e:
            print(f"  Token Validation: ❌ ({e})")

    def test_pipelines(self):
        """Test pipeline endpoints"""
        print("\n🔧 Testing Pipeline Endpoints...")

        # Basic pipeline list
        try:
            resp = self.session.get(f"{self.base_url}/api/v1/pipelines", timeout=5)
            print(f"  Pipeline List: {resp.status_code} ({'✅' if resp.status_code == 200 else '❌'})")
            if resp.status_code == 200:
                data = resp.json()
                pipelines = data.get('pipelines', [])
                print(f"    Available: {len(pipelines)} pipelines")
                for p in pipelines[:3]:  # Show first 3
                    print(f"    - {p.get('name', 'unknown')}: {p.get('description', 'no description')}")
        except Exception as e:
            print(f"  Pipeline List: ❌ ({e})")

        # Debug pipeline info
        try:
            resp = self.session.get(f"{self.base_url}/api/v1/debug/pipelines", timeout=5)
            print(f"  Pipeline Debug: {resp.status_code} ({'✅' if resp.status_code == 200 else '❌'})")
        except Exception as e:
            print(f"  Pipeline Debug: ❌ ({e})")

    def test_job_endpoints(self):
        """Test job management endpoints"""
        print("\n📋 Testing Job Endpoints...")

        # Job listing
        try:
            resp = self.session.get(f"{self.base_url}/api/v1/jobs", timeout=5)
            print(f"  Job List: {resp.status_code} ({'✅' if resp.status_code == 200 else '❌'})")
            if resp.status_code == 200:
                data = resp.json()
                print(f"    Total Jobs: {data.get('total', 0)}")
        except Exception as e:
            print(f"  Job List: ❌ ({e})")

        # Test job submission (mock video)
        try:
            files = {'video': ('test.mp4', b'fake video content', 'video/mp4')}
            data = {'selected_pipelines': 'person_tracking'}

            resp = self.session.post(
                f"{self.base_url}/api/v1/jobs/",
                files=files,
                data=data,
                timeout=10
            )
            print(f"  Job Submission: {resp.status_code} ({'✅' if resp.status_code == 201 else '❌'})")

            if resp.status_code == 201:
                job_data = resp.json()
                job_id = job_data.get('id')
                print(f"    Job ID: {job_id}")

                # Test job status retrieval
                status_resp = self.session.get(f"{self.base_url}/api/v1/jobs/{job_id}", timeout=5)
                print(f"  Job Status: {status_resp.status_code} ({'✅' if status_resp.status_code == 200 else '❌'})")

                return job_id
        except Exception as e:
            print(f"  Job Submission: ❌ ({e})")

        return None

    def test_missing_endpoints(self):
        """Test for missing endpoints that should return 404"""
        print("\n❓ Testing Missing Endpoints (Should be 404)...")

        missing_endpoints = [
            "/api/v1/events/stream",
            "/api/v1/jobs/123/results",
            "/api/v1/jobs/123/artifacts",
            "/api/v1/videos"
        ]

        for endpoint in missing_endpoints:
            try:
                resp = self.session.get(f"{self.base_url}{endpoint}", timeout=5)
                status = "❌ (Implemented)" if resp.status_code != 404 else "✅ (Missing as expected)"
                print(f"  {endpoint}: {resp.status_code} {status}")
            except Exception as e:
                print(f"  {endpoint}: ❌ ({e})")

    def run_all_tests(self):
        """Run comprehensive API tests"""
        print("🧪 VideoAnnotator API Quick Test Suite")
        print("=" * 50)

        self.test_health_endpoints()
        self.test_authentication()
        self.test_pipelines()
        job_id = self.test_job_endpoints()
        self.test_missing_endpoints()

        print("\n" + "=" * 50)
        print("🏁 Test Suite Complete!")
        print("\n💡 Tips for Client Developers:")
        print("- Use /api/v1/debug/* endpoints for detailed information")
        print("- Check logs at /api/v1/debug/request-log")
        print("- Monitor server resources at /api/v1/debug/server-info")
        if job_id:
            print(f"- Debug job details at /api/v1/debug/jobs/{job_id}")

if __name__ == "__main__":
    import sys

  base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:18011"
    token = sys.argv[2] if len(sys.argv) > 2 else "dev-token"

    tester = APIQuickTester(base_url, token)
    tester.run_all_tests()
```

### **3. Browser Development Tools**

#### **API Debug Console**

Add this to the client app for easy debugging:

```javascript
// Add to client application for debugging
window.VideoAnnotatorDebug = {
  // Quick API health check
  async checkHealth() {
    const response = await fetch("/api/v1/system/health");
    const data = await response.json();
    console.table(data.services);
    return data;
  },

  // Test authentication
  async checkAuth() {
    const token = localStorage.getItem("api_token") || "dev-token";
    const response = await fetch("/api/v1/debug/token-info", {
      headers: { Authorization: `Bearer ${token}` },
    });
    const data = await response.json();
    console.log("Token Info:", data);
    return data;
  },

  // Get server info
  async getServerInfo() {
    const response = await fetch("/api/v1/debug/server-info");
    const data = await response.json();
    console.table(data.server);
    console.table(data.system);
    return data;
  },

  // Monitor job progress
  async monitorJob(jobId) {
    console.log(`Monitoring job: ${jobId}`);
    const checkStatus = async () => {
      const response = await fetch(`/api/v1/jobs/${jobId}`);
      const job = await response.json();
      console.log(`Job ${jobId} status: ${job.status}`);

      if (job.status === "completed" || job.status === "failed") {
        console.log("Job finished:", job);
        return job;
      }

      setTimeout(checkStatus, 2000); // Check every 2 seconds
    };

    checkStatus();
  },

  // Enable request logging
  logAllRequests: true,
};

// Intercept fetch requests for debugging when enabled
if (window.VideoAnnotatorDebug.logAllRequests) {
  const originalFetch = window.fetch;
  window.fetch = function (...args) {
    console.log("API Request:", args[0], args[1]);
    return originalFetch.apply(this, arguments).then((response) => {
      console.log("API Response:", response.status, response.url);
      return response;
    });
  };
}
```

---

## 📚 Shared Documentation & Guidelines

### **Error Code Reference**

Create standardized error codes both teams can reference:

```javascript
// Shared error codes for client-server communication
const API_ERRORS = {
  // Authentication Errors (401)
  AUTH_TOKEN_MISSING: {
    code: "E1001",
    message: "Authorization token is required",
  },
  AUTH_TOKEN_INVALID: {
    code: "E1002",
    message: "Authorization token is invalid or expired",
  },
  AUTH_INSUFFICIENT_PERMISSIONS: {
    code: "E1003",
    message: "Insufficient permissions for this operation",
  },

  // Job Errors (400/422)
  JOB_VIDEO_MISSING: {
    code: "E2001",
    message: "Video file is required for job submission",
  },
  JOB_INVALID_PIPELINE: {
    code: "E2002",
    message: "Invalid pipeline specified",
  },
  JOB_CONFIG_INVALID: {
    code: "E2003",
    message: "Job configuration is invalid",
  },

  // Server Errors (500)
  SERVER_PIPELINE_INIT_FAILED: {
    code: "E3001",
    message: "Failed to initialize processing pipeline",
  },
  SERVER_DATABASE_ERROR: {
    code: "E3002",
    message: "Database operation failed",
  },
  SERVER_STORAGE_ERROR: {
    code: "E3003",
    message: "File storage operation failed",
  },

  // Missing Features (501)
  FEATURE_NOT_IMPLEMENTED: {
    code: "E4001",
    message: "Feature not yet implemented",
  },
  ENDPOINT_NOT_AVAILABLE: {
    code: "E4002",
    message: "Endpoint not available in current version",
  },
};
```

### **Testing Collaboration Protocol**

#### **Issue Reporting Template**

```markdown
## 🐛 API Issue Report

**Reporter**: [Client/Server Team]
**Date**: [YYYY-MM-DD]
**Priority**: [Critical/High/Medium/Low]

### Issue Description

[Brief description of the problem]

### Reproduction Steps

1. [Step 1]
2. [Step 2]
3. [Step 3]

### Expected Behavior

[What should happen]

### Actual Behavior

[What actually happens]

### Debug Information

- **Server Info**: [Output from /api/v1/debug/server-info]
- **Token Info**: [Output from /api/v1/debug/token-info]
- **Request Log**: [Relevant entries from /api/v1/debug/request-log]
- **Browser Console**: [Any client-side errors]

### Environment

- Server Version: [e.g., v1.4.1]
- Client Version: [e.g., v1.0.3]
- Browser: [if applicable]
- API Base URL: [e.g., http://localhost:18011]

### Additional Context

[Any other relevant information]
```

---

## 🚀 Implementation Priority

### **Sprint 1 (Immediate - Week 1)**

1. **Add debug endpoints** to existing server
2. **Create API test script** for client developers
3. **Implement mock SSE endpoint** for client testing

### **Sprint 2 (Week 2-3)**

4. **Add browser debug tools** to client application
5. **Create shared error code reference**
6. **Establish testing collaboration protocol**

### **Sprint 3 (Week 4+)**

7. **Advanced debugging features** based on team feedback
8. **Automated testing integration**
9. **Performance monitoring tools**

These tools will dramatically improve collaboration between client and server teams, reduce debugging time, and help identify issues faster! 🔧✨

---

**Document Version**: v1.0
**Last Updated**: January 2025
**Status**: Ready for implementation
