# 🤝 VideoAnnotator Client-Server Testing Collaboration Guide

## Overview

This guide establishes workflows and protocols for effective collaboration between client-side and server-side development teams during VideoAnnotator v1.2.0 development.

---

## 🎯 Quick Start for Teams

### **For Client Developers**

1. **Test Server Connectivity**:
   ```bash
   # Quick API test
   uv run python scripts/test_api_quick.py http://localhost:18011 your-token
   ```

2. **Browser Debug Console**:
   ```javascript
   // Paste into browser console
   VideoAnnotatorDebug.runAllTests()
   VideoAnnotatorDebug.checkHealth()
   ```

3. **Monitor API Issues**:
   - Check `/api/v1/debug/server-info` for server status
   - Use `/api/v1/debug/token-info` to verify authentication
   - Monitor `/api/v1/debug/request-log` for API call history

### **For Server Developers**

1. **Enable Debug Endpoints**:
   ```python
   # Add to API router (already done in v1.2.0)
   api_router.include_router(debug_router, prefix="/debug", tags=["debug"])
   ```

2. **Monitor Client Issues**:
   ```bash
   # Check debug endpoints are working
   curl -H "Authorization: Bearer dev-token" http://localhost:18011/api/v1/debug/server-info
   ```

3. **Test SSE Implementation**:
   ```bash
   # Test mock SSE endpoint
   curl -H "Authorization: Bearer dev-token" http://localhost:18011/api/v1/debug/mock-events
   ```

---

## 📋 Issue Reporting Workflow

### **1. Issue Discovery**
- **Client Team**: Use automated testing tools to identify API issues
- **Server Team**: Monitor debug endpoints and logs for errors
- **Both Teams**: Use standardized error codes for communication

### **2. Issue Documentation**
Use this template for all API issues:

```markdown
## 🐛 API Issue Report

**Reporter**: [Client Team / Server Team]
**Date**: [YYYY-MM-DD]
**Priority**: [Critical/High/Medium/Low]
**Issue Type**: [Bug/Enhancement/Missing Feature]

### Issue Description
[Clear description of the problem]

### Environment
- Server URL: [e.g., http://localhost:18011]
- API Version: [from /health endpoint]
- Server Version: [from /api/v1/debug/server-info]
- Client Version: [if applicable]
- Browser: [if applicable]

### Reproduction Steps
1. [Step 1]
2. [Step 2] 
3. [Step 3]

### Expected Behavior
[What should happen]

### Actual Behavior
[What actually happens]

### Debug Information
**Server Info**: [Output from `/api/v1/debug/server-info`]
**Token Info**: [Output from `/api/v1/debug/token-info`]
**Request Log**: [Recent entries from `/api/v1/debug/request-log`]
**Browser Console**: [Any client-side errors]

### Test Results
[Output from test_api_quick.py or browser debug console]

### Additional Context
[Screenshots, logs, etc.]
```

### **3. Issue Triage**
- **Critical**: API server crashes, authentication bypass, data corruption
- **High**: Missing endpoints, authentication failures, job processing failures
- **Medium**: Performance issues, inconsistent responses, documentation gaps
- **Low**: UI/UX improvements, additional features, minor bugs

---

## 🔧 Debugging Tools Reference

### **Server Debug Endpoints**

| Endpoint | Purpose | Auth Required |
|----------|---------|---------------|
| `/api/v1/debug/server-info` | Server status and configuration | No |
| `/api/v1/debug/token-info` | Token validation and permissions | Yes |
| `/api/v1/debug/pipelines` | Pipeline configuration details | No |
| `/api/v1/debug/jobs/{id}` | Detailed job debugging info | Yes |
| `/api/v1/debug/request-log` | Recent API request history | Yes |
| `/api/v1/debug/mock-events` | Mock SSE events for testing | No |

### **Client Testing Tools**

| Tool | Purpose | Usage |
|------|---------|-------|
| `test_api_quick.py` | Automated API testing | `python scripts/test_api_quick.py` |
| `browser_debug_console.js` | Browser debugging tools | Paste into browser console |
| `VideoAnnotatorAPITester` | JavaScript testing library | Include in client app |

### **Shared Error Codes**

| Code | Type | Description | HTTP Status |
|------|------|-------------|-------------|
| E1001 | AUTH_TOKEN_MISSING | Authorization token required | 401 |
| E1002 | AUTH_TOKEN_INVALID | Token invalid or expired | 401 |
| E2001 | JOB_VIDEO_MISSING | Video file required | 422 |
| E2002 | JOB_INVALID_PIPELINE | Invalid pipeline specified | 422 |
| E3001 | SERVER_PIPELINE_INIT_FAILED | Pipeline initialization failed | 500 |
| E4001 | FEATURE_NOT_IMPLEMENTED | Feature not implemented | 501 |

---

## 🚀 Development Workflow

### **Daily Standup Integration**
1. **Server Team Reports**:
   - New endpoints implemented
   - Breaking changes made
   - Debug information available
   
2. **Client Team Reports**:
   - API integration issues found
   - Performance concerns
   - Missing features needed

3. **Shared Blockers**:
   - Critical API issues
   - Authentication problems
   - Missing endpoints

### **Sprint Planning**
1. **Review Critical Issues**: Address blocking API problems first
2. **Prioritize Missing Endpoints**: Based on client development needs
3. **Plan Integration Testing**: Schedule joint testing sessions
4. **Update Debug Tools**: Add new endpoints to testing tools

### **Integration Testing Sessions**
- **Weekly 1-hour sessions** with both teams
- **Test latest API changes** together
- **Resolve integration issues** in real-time
- **Update documentation** based on findings

---

## 📊 Testing Automation

### **Continuous Integration**
```yaml
# Example CI integration
name: API Integration Tests
on: [push, pull_request]
jobs:
  api-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Start API server
        run: uv run python api_server.py &
      - name: Wait for server
        run: sleep 10
      - name: Run API tests
   run: uv run python scripts/test_api_quick.py http://localhost:18011 test-token
```

### **Automated Issue Detection**
- **Daily API health checks** in staging environment
- **Performance regression testing** for API response times
- **Authentication testing** with various token scenarios
- **Missing endpoint monitoring** for 404 responses

---

## 📚 Documentation Synchronization

### **API Documentation Updates**
1. **Server Team**: Update OpenAPI spec when adding endpoints
2. **Client Team**: Update integration docs when using new features
3. **Both Teams**: Keep error code reference updated

### **Change Communication**
- **Slack/Teams notifications** for API changes
- **Shared changelog** for breaking changes
- **Documentation reviews** before major releases

---

## ⚡ Emergency Response

### **Critical Issue Protocol**
1. **Immediate Notification**: Alert both teams via emergency channel
2. **Quick Assessment**: Use debug endpoints to assess scope
3. **Temporary Workarounds**: Implement client-side fallbacks if possible
4. **Root Cause Analysis**: Use debug logs and server info
5. **Coordinated Fix**: Server fix + client update if needed

### **Rollback Procedures**
- **Server Rollback**: Revert to previous stable API version
- **Client Fallback**: Switch to mock data or cached responses
- **Communication**: Clear timeline for resolution

---

## 🎉 Success Metrics

### **Collaboration Effectiveness**
- **Issue Resolution Time**: Average time from report to fix
- **Integration Success Rate**: Percentage of successful API integrations
- **Test Coverage**: Number of API endpoints covered by automated tests
- **Communication Quality**: Number of issues resolved without meetings

### **Technical Quality**
- **API Uptime**: 99.9% availability during development
- **Response Times**: <200ms for status endpoints
- **Error Rate**: <1% of API calls result in 5xx errors
- **Documentation Accuracy**: 95% of endpoints properly documented

---

## 🔧 Tool Setup Instructions

### **Server Team Setup**
```bash
# 1. Ensure debug endpoints are enabled
# (Already included in v1.2.0)

# 2. Test debug endpoints
curl http://localhost:18011/api/v1/debug/server-info

# 3. Monitor request logs
curl -H "Authorization: Bearer dev-token" \
   http://localhost:18011/api/v1/debug/request-log
```

### **Client Team Setup**
```bash
# 1. Run quick API test
uv run python scripts/test_api_quick.py

# 2. Copy browser debug console to clipboard
cat scripts/browser_debug_console.js | pbcopy  # macOS
cat scripts/browser_debug_console.js | xclip   # Linux

# 3. Paste into browser console and run tests
# VideoAnnotatorDebug.runAllTests()
```

---

## 📞 Support Contacts

- **Server Team Lead**: [Contact info]
- **Client Team Lead**: [Contact info] 
- **DevOps/Infrastructure**: [Contact info]
- **Emergency Escalation**: [Contact info]

---

**Document Version**: v1.0  
**Last Updated**: January 2025  
**Next Review**: Weekly during active development  
**Status**: Active - update as processes evolve