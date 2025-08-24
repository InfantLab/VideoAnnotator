# 🔧 VideoAnnotator Server Integration Tools

This directory contains debugging and testing tools for client-server collaboration with VideoAnnotator API servers.

## Quick Start

### 1. Test API Server Connection
```bash
# Test local server
python test_api_quick.py

# Test specific server
python test_api_quick.py http://your-server.com your-api-token
```

### 2. Browser Debugging Console
```javascript
// Copy browser_debug_console.js contents
// Paste into browser developer console (F12)
VideoAnnotatorDebug.runAllTests()
```

## Tools Overview

### `test_api_quick.py` - Automated API Testing
- **Purpose**: Comprehensive testing of VideoAnnotator API endpoints
- **No Dependencies**: Uses only Python standard library  
- **Output**: Detailed test results saved to `test_results_api.json`

**Test Coverage**:
- ✅ Health endpoints (basic + detailed)
- ✅ Authentication and token validation
- ✅ Pipeline availability and status
- ✅ Job submission and management  
- ✅ Server-Sent Events (SSE) connection
- ✅ Debug endpoint availability
- ✅ Missing endpoint detection

**Usage Examples**:
```bash
# Default: localhost:8000 with dev-token
python test_api_quick.py

# Custom server and token  
python test_api_quick.py https://api.example.com my-api-token

# Docker container testing
python test_api_quick.py http://videoannotator:8001 test-token
```

### `browser_debug_console.js` - Interactive Browser Tools
- **Purpose**: Real-time debugging and monitoring in browser
- **Integration**: Paste into browser console for immediate access
- **Features**: API testing, job monitoring, SSE testing, request logging

**Available Functions**:
```javascript
// Health and connectivity
VideoAnnotatorDebug.checkHealth()
VideoAnnotatorDebug.getServerInfo()

// Authentication 
VideoAnnotatorDebug.checkAuth('your-token')

// Pipeline testing
VideoAnnotatorDebug.checkPipelines()

// Job management
VideoAnnotatorDebug.submitTestJob()
VideoAnnotatorDebug.monitorJob('job-id')

// Real-time events
VideoAnnotatorDebug.testSSE()

// Comprehensive testing
VideoAnnotatorDebug.runAllTests()

// Request monitoring
VideoAnnotatorDebug.enableRequestLogging()
VideoAnnotatorDebug.disableRequestLogging()

// Help and documentation
VideoAnnotatorDebug.help()
```

## Common Use Cases

### 🔍 **Troubleshooting API Issues**
```bash
# 1. Run comprehensive test
python test_api_quick.py http://your-server.com

# 2. Check specific endpoints in browser
VideoAnnotatorDebug.checkHealth()
VideoAnnotatorDebug.checkAuth()
```

### 📊 **Monitoring Job Progress**
```javascript
// Submit test job and monitor
const job = await VideoAnnotatorDebug.submitTestJob();
VideoAnnotatorDebug.monitorJob(job.id);
```

### 🔐 **Authentication Debugging**
```javascript
// Test token validity
VideoAnnotatorDebug.checkAuth('your-token');

// Check what permissions you have
// (Results shown in browser console)
```

### 🚀 **Performance Testing**
```javascript
// Enable request logging to monitor API calls
VideoAnnotatorDebug.enableRequestLogging();

// Your API calls will now be logged
// Use your application normally

// Disable when done
VideoAnnotatorDebug.disableRequestLogging();
```

### 📡 **SSE Connection Testing**
```javascript
// Test Server-Sent Events
VideoAnnotatorDebug.testSSE();

// Monitor specific job events
VideoAnnotatorDebug.testSSE('your-job-id');
```

## Integration with CI/CD

### GitHub Actions Example
```yaml
name: API Integration Tests
on: [push, pull_request]
jobs:
  api-integration:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Test VideoAnnotator API
        run: python scripts/test_api_quick.py ${{ secrets.API_URL }} ${{ secrets.API_TOKEN }}
```

### Pre-deployment Testing
```bash
# Add to your deployment pipeline
python scripts/test_api_quick.py $STAGING_API_URL $STAGING_TOKEN
# Exit code 0 = all tests passed
# Exit code 1 = some tests failed
```

## Output and Results

### Python Test Results
- **Console Output**: Color-coded test results with status
- **JSON File**: Detailed results saved to `test_results_api.json`
- **Exit Codes**: 0 for success, 1 for failures

### Browser Console Output
- **Structured Logs**: Clear test results in browser console
- **Tables**: Formatted data display for server info
- **Groups**: Organized output for complex operations
- **Real-time**: Live updates for monitoring operations

## Troubleshooting

### Common Issues

**Python ModuleNotFoundError**: 
- Scripts use only standard library - no pip install needed

**Connection Refused**: 
- Check server URL and ensure VideoAnnotator API is running
- Test with: `curl http://your-server/health`

**Authentication Errors**: 
- Verify token with server team
- Use debug endpoints to check token validity

**CORS Issues in Browser**: 
- Browser tools work with same-origin requests
- For cross-origin, use Python script instead

## Support

For issues with these tools:
1. Run the Python test script first for automated diagnostics
2. Check the collaboration guide: `docs/CLIENT_SERVER_COLLABORATION_GUIDE.md`
3. Include test results when reporting server integration issues

---

**Compatibility**: VideoAnnotator API v1.2.0+  
**Python Requirements**: Python 3.8+  
**Browser Support**: Modern browsers with ES6+ support