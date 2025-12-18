# 🔐 VideoAnnotator Client Token Setup Guide

## Overview

This guide provides secure, user-friendly methods for setting up authentication tokens between VideoAnnotator clients and servers. The system supports multiple authentication flows designed for different use cases.

**Target Audience**: Client developers, system administrators, end users
**Server Version**: VideoAnnotator API v1.4.1+

---

## 🎯 Authentication Methods

### **1. API Keys (Recommended for Production)**

- **Use Case**: Long-term programmatic access, production deployments
- **Lifetime**: Configurable (30-365 days, or no expiration)
- **Security**: High - encrypted storage, secure generation
- **Management**: Full lifecycle management via CLI tools

### **2. Session Tokens (Web Applications)**

- **Use Case**: Web-based clients, temporary access
- **Lifetime**: Short-term (1-24 hours)
- **Security**: JWT-based with automatic expiration
- **Management**: Automatic cleanup, refresh token support

### **3. Development Tokens (Development Only)**

- **Use Case**: Local development, testing, debugging
- **Lifetime**: No expiration
- **Security**: Basic - hardcoded values
- **Management**: Built-in for convenience

---

## 🚀 Quick Start

### **For End Users (Client Applications)**

#### **Option 1: Environment Variables (Recommended)**

```bash
# Set your API token (most secure)
export VIDEOANNOTATOR_API_TOKEN="va_api_your_token_here"
export VIDEOANNOTATOR_API_URL="http://your-server:18011"

# Start your client application
npm run dev
# or
python your_client.py
```

#### **Option 2: Configuration File**

Create `~/.videoannotator/config.json`:

```json
{
  "api_url": "http://your-server:18011",
  "api_token": "va_api_your_token_here",
  "user_info": {
    "username": "your_username",
    "email": "your@email.com"
  }
}
```

#### **Option 3: Interactive Login**

Many clients support interactive token entry:

```bash
# Client prompts for token on first run
videoannotator-client
> Enter your API token: va_api_your_token_here
> Token saved securely
```

### **For Developers (Testing)**

```javascript
// Use development tokens for testing
const apiToken = "dev-token"; // Built-in development token
const apiUrl = "http://localhost:18011";

// Make authenticated requests
fetch(`${apiUrl}/api/v1/jobs/`, {
  headers: {
    Authorization: `Bearer ${apiToken}`,
    "Content-Type": "application/json",
  },
});
```

---

## macOS Notes

- Apple Silicon (M1/M2) may require OpenMP runtime for some audio models. If you see OpenMP or segfault errors, install:

```bash
brew install libomp
```

- Ensure Node.js is installed if using the web viewer:

```bash
brew install node
```

---

## 🔧 Server-Side Token Management

### **Creating API Keys (Server Administrator)**

#### **Interactive CLI Method (Recommended)**

```bash
# Create new API key interactively (stored in database)
uv run videoannotator generate-token
> Enter user email: user@example.com
> Enter username (default: user): user
> Enter key name/description: My API Key
> Expires in days (default: 365, 0 for no expiry): 365

# Result
[SUCCESS] API Key Created Successfully!
Token:      va_...
User:       user (user@example.com)
...
```

#### **Command-Line Method**

```bash
# Create API key with specific parameters
uv run videoannotator generate-token \
  --user user@example.com \
  --username user \
  --key-name "Production client access" \
  --expires-days 365 \
  --output user-token.json

# Create token with no expiration
uv run videoannotator generate-token \
  --user admin@example.com \
  --no-expiry \
  --key-name "Admin access token"
```

### **Token Management Commands**

> **Note**: The `videoannotator` CLI currently supports token generation. For listing or revoking database-backed tokens, you may need to access the database directly or use the legacy `scripts/manage_tokens.py` for file-based tokens (not recommended for new deployments).

```bash
# Initialize database and create admin user
uv run videoannotator setup-db --create-admin
```

---

## 🛡️ Security Best Practices

### **For Server Administrators**

1. **Token Storage Security**:

   ```bash
   # Tokens are automatically encrypted on disk
   ls -la tokens/
   # -rw------- tokens.json      (encrypted)
   # -rw------- encryption.key   (secure key)
   ```

2. **Regular Token Rotation**:

   ```bash
   # Set up automated cleanup
   crontab -e
   # Add: 0 2 * * * cd /path/to/videoannotator && python scripts/manage_tokens.py cleanup

   # Monitor token usage
   python scripts/manage_tokens.py stats
   ```

3. **Scope Management**:
   - **read**: View data, download results
   - **write**: Submit jobs, upload videos
   - **admin**: User management, server configuration
   - **debug**: Access debug endpoints, detailed logs

### **For Client Developers**

1. **Secure Token Storage**:

   ```javascript
   // ❌ DON'T: Store in code or plain text
   const apiToken = "va_api_hardcoded_token";
   localStorage.setItem("token", "va_api_plaintext");

   // ✅ DO: Use secure storage
   // Environment variables
   const apiToken = process.env.VIDEOANNOTATOR_API_TOKEN;

   // Encrypted storage (web)
   import { EncryptedStorage } from "secure-storage";
   const storage = new EncryptedStorage("your-secret-key");
   storage.setItem("api_token", token);

   // OS keychain (desktop)
   import keytar from "keytar";
   keytar.setPassword("videoannotator", "api_token", token);
   ```

2. **Token Validation**:

   ```javascript
   // Validate token before making requests
   async function validateToken(token) {
     try {
       const response = await fetch(`${API_URL}/api/v1/debug/token-info`, {
         headers: { Authorization: `Bearer ${token}` },
       });
       return response.ok;
     } catch {
       return false;
     }
   }
   ```

3. **Error Handling**:
   ```javascript
   // Handle authentication errors gracefully
   fetch(url, { headers: { Authorization: `Bearer ${token}` } }).then(
     (response) => {
       if (response.status === 401) {
         // Token expired or invalid
         promptForNewToken();
         return;
       }
       return response.json();
     },
   );
   ```

### **For End Users**

1. **Token Security**:

   - Never share your API token
   - Don't paste tokens in chat/email
   - Use secure password managers
   - Report lost/stolen tokens immediately

2. **Token Lifecycle**:
   - Note your token expiration date
   - Request new tokens before expiration
   - Revoke old tokens when no longer needed

---

## 🔄 Client Integration Examples

### **React/JavaScript Client**

```javascript
// config.js
export const API_CONFIG = {
  baseURL: process.env.REACT_APP_API_URL || "http://localhost:18011",
  token:
    process.env.REACT_APP_API_TOKEN ||
    localStorage.getItem("videoannotator_token"),
};

// api-client.js
class VideoAnnotatorAPI {
  constructor(config) {
    this.baseURL = config.baseURL;
    this.token = config.token;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const headers = {
      "Content-Type": "application/json",
      Authorization: `Bearer ${this.token}`,
      ...options.headers,
    };

    const response = await fetch(url, { ...options, headers });

    if (response.status === 401) {
      throw new Error("Authentication failed - check your API token");
    }

    return response.json();
  }

  // API methods
  async getJobs() {
    return this.request("/api/v1/jobs/");
  }

  async submitJob(videoFile, pipelines) {
    const formData = new FormData();
    formData.append("video", videoFile);
    formData.append("selected_pipelines", pipelines.join(","));

    return this.request("/api/v1/jobs/", {
      method: "POST",
      headers: {}, // Let browser set Content-Type for FormData
      body: formData,
    });
  }
}

// usage
const client = new VideoAnnotatorAPI(API_CONFIG);
```

### **Python Client**

```python
# videoannotator_client.py
import os
import requests
from pathlib import Path
import json

class VideoAnnotatorClient:
    def __init__(self, api_url=None, api_token=None):
    self.api_url = api_url or os.getenv('VIDEOANNOTATOR_API_URL', 'http://localhost:18011')
        self.api_token = api_token or self._load_token()

        if not self.api_token:
            raise ValueError("API token required. Set VIDEOANNOTATOR_API_TOKEN environment variable or ~/.videoannotator/config.json")

    def _load_token(self):
        # Try environment variable first
        token = os.getenv('VIDEOANNOTATOR_API_TOKEN')
        if token:
            return token

        # Try config file
        config_file = Path.home() / '.videoannotator' / 'config.json'
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
                return config.get('api_token')

        return None

    def _request(self, method, endpoint, **kwargs):
        url = f"{self.api_url}{endpoint}"
        headers = kwargs.pop('headers', {})
        headers['Authorization'] = f'Bearer {self.api_token}'

        response = requests.request(method, url, headers=headers, **kwargs)

        if response.status_code == 401:
            raise Exception('Authentication failed - check your API token')

        response.raise_for_status()
        return response.json()

    def get_jobs(self):
        return self._request('GET', '/api/v1/jobs/')

    def submit_job(self, video_path, pipelines):
        with open(video_path, 'rb') as f:
            files = {'video': f}
            data = {'selected_pipelines': ','.join(pipelines)}
            return self._request('POST', '/api/v1/jobs/', files=files, data=data)

# Usage
client = VideoAnnotatorClient()
jobs = client.get_jobs()
```

### **Configuration File Setup**

```bash
# Create user config directory
mkdir -p ~/.videoannotator

# Create configuration file
cat > ~/.videoannotator/config.json << EOF
{
  "api_url": "http://your-server:18011",
  "api_token": "va_api_your_token_here",
  "user_info": {
    "username": "your_username",
    "email": "your@email.com"
  },
  "client_settings": {
    "timeout": 30,
    "retry_count": 3,
    "auto_refresh": true
  }
}
EOF

# Secure the config file
chmod 600 ~/.videoannotator/config.json
```

---

## 🔍 Troubleshooting

### **Common Issues**

#### **"Authentication failed - check your API token"**

```bash
# 1. Ensure you are using a valid token
# If using database tokens (recommended), ensure the token exists in the DB.

# 2. Create new token if needed
uv run videoannotator generate-token --user your@email.com
```

#### **"Connection refused" or "Server not found"**

```bash
# Check server status
curl http://your-server:18011/health

# Check if server is running
ps aux | grep videoannotator

# Check server logs
tail -f logs/api_server.log
```

#### **"Permission denied" or "Insufficient scopes"**

```bash
# Request token with appropriate permissions
# Note: Database tokens currently default to read/write scopes.
uv run videoannotator generate-token \
  --user your@email.com \
  --key-name "Admin Access"
```

### **Development Debugging**

```bash
# Use debug endpoints to test authentication
curl -H "Authorization: Bearer your_token" \
  http://localhost:18011/api/v1/debug/token-info

# Check server logs for authentication issues
tail -f logs/api_requests.log | grep "401\|403"

# Test with development token
curl -H "Authorization: Bearer dev-token" \
  http://localhost:18011/api/v1/debug/server-info
```

---

## 📚 Integration Checklist

### **For Client Developers**

- [ ] **Token Storage**: Implement secure token storage (environment variables, encrypted storage, or OS keychain)
- [ ] **Token Validation**: Validate tokens before making requests
- [ ] **Error Handling**: Handle 401/403 responses gracefully
- [ ] **Token Refresh**: Implement token refresh logic for session tokens
- [ ] **Fallback Methods**: Support multiple token input methods (env vars, config files, interactive)
- [ ] **Security**: Never log or expose tokens in client-side code
- [ ] **User Experience**: Provide clear authentication error messages

### **For Server Administrators**

- [ ] **Token Management**: Set up regular token cleanup and monitoring
- [ ] **Scope Configuration**: Define appropriate scopes for different user roles
- [ ] **Security**: Ensure token storage encryption and secure file permissions
- [ ] **Monitoring**: Monitor token usage and authentication failures
- [ ] **Documentation**: Provide clear token setup instructions for users
- [ ] **Backup**: Include token storage in backup procedures

### **For End Users**

- [ ] **Token Acquisition**: Get API token from administrator or self-service portal
- [ ] **Secure Storage**: Store token securely (not in plain text)
- [ ] **Configuration**: Set up client configuration with token and server URL
- [ ] **Testing**: Test authentication with client application
- [ ] **Monitoring**: Note token expiration and renewal requirements

---

## 🎯 Advanced Features

### **Session Token with Refresh**

```javascript
class TokenManager {
  constructor(apiUrl) {
    this.apiUrl = apiUrl;
    this.accessToken = localStorage.getItem("access_token");
    this.refreshToken = localStorage.getItem("refresh_token");
  }

  async refreshAccessToken() {
    const response = await fetch(`${this.apiUrl}/api/v1/auth/refresh`, {
      method: "POST",
      headers: { Authorization: `Bearer ${this.refreshToken}` },
    });

    if (response.ok) {
      const { access_token } = await response.json();
      this.accessToken = access_token;
      localStorage.setItem("access_token", access_token);
      return access_token;
    }

    throw new Error("Token refresh failed");
  }

  async authenticatedRequest(url, options = {}) {
    let token = this.accessToken;

    // Try request with current token
    let response = await fetch(url, {
      ...options,
      headers: {
        Authorization: `Bearer ${token}`,
        ...options.headers,
      },
    });

    // If 401, try refreshing token
    if (response.status === 401 && this.refreshToken) {
      try {
        token = await this.refreshAccessToken();
        response = await fetch(url, {
          ...options,
          headers: {
            Authorization: `Bearer ${token}`,
            ...options.headers,
          },
        });
      } catch {
        // Refresh failed, redirect to login
        this.redirectToLogin();
      }
    }

    return response;
  }
}
```

### **Multi-Environment Configuration**

```bash
# ~/.videoannotator/environments.json
{
  "development": {
    "api_url": "http://localhost:18011",
    "api_token": "dev-token"
  },
  "staging": {
    "api_url": "https://staging-api.example.com",
    "api_token": "va_api_staging_token_here"
  },
  "production": {
    "api_url": "https://api.example.com",
    "api_token": "va_api_production_token_here"
  }
}

# Usage
export VIDEOANNOTATOR_ENV=production
videoannotator-client  # Automatically uses production config
```

---

**Document Version**: v1.0
**Last Updated**: 24 August 2025
**Compatible With**: VideoAnnotator API v1.4.1+
**Status**: Production ready with comprehensive security features
