# CORS & Authentication Protocol for Frontend Development

This document outlines the standard protocol for connecting frontend clients (web apps, scripts, external tools) to the VideoAnnotator API server, specifically addressing CORS (Cross-Origin Resource Sharing) and Authentication challenges in development environments (Dev Containers, Codespaces).

## 1. The Challenge

When running the VideoAnnotator server in a Dev Container or Codespace, and a frontend client in a browser or another environment, you will encounter two main hurdles:

1.  **Connectivity**: The client cannot reach `localhost` if it's not running on the same machine/network stack as the server.
2.  **CORS**: The browser blocks requests because the "Origin" (where the client is served from) does not match the allowed origins on the server.

## 2. CORS Configuration (The "400 Bad Request" Fix)

By default, the server allows `localhost` and `127.0.0.1` on specific ports. This is insufficient for:
-   **Codespaces**: The origin is `https://<codespace-name>-<port>.app.github.dev`.
-   **Custom Tunnels**: Any other forwarding URL.
-   **Production/Staging**: Different domains.

### Protocol for Developers

1.  **Identify your Origin**: Open your frontend application in the browser and copy the URL (e.g., `http://localhost:3000` or `https://shiny-space-waffle-18011.app.github.dev`).
2.  **Configure the Server**:
    -   Open `.env` in the workspace root.
    -   Add or update the `CORS_ORIGINS` variable.
    -   **For Development (Recommended)**: Allow ALL origins to avoid headaches.
        ```dotenv
        CORS_ORIGINS="*"
        ```
    -   **For Strict Testing**: Add your specific origin.
        ```dotenv
        CORS_ORIGINS="http://localhost:3000,https://your-codespace-url.app.github.dev"
        ```
3.  **Restart the Server**: The server must be restarted for this change to take effect.

**Note**: When `CORS_ORIGINS="*"` is set, the server automatically switches to a permissive mode that allows credentials (cookies/auth headers) with wildcard origins, which is normally restricted by browsers.

## 3. Authentication Protocol

The API requires an API Key for all non-public endpoints.

### Step 1: Obtain the API Key

When the server starts for the first time, it generates an admin key.

-   **Check the Console**: Look for the banner `[API KEY] VIDEOANNOTATOR API KEY GENERATED` in the terminal logs.
-   **Check the File**: If you missed the logs, the key is stored in `tokens/tokens.json`.
    -   *Note*: This file is git-ignored. Do not commit it.

### Step 2: Use the API Key

Pass the key in the `Authorization` header of every HTTP request.

```http
Authorization: Bearer <YOUR_API_KEY>
```

**Example (JavaScript/Fetch):**
```javascript
const response = await fetch('http://localhost:18011/api/v1/jobs', {
  headers: {
    'Authorization': 'Bearer ' + apiKey,
    'Content-Type': 'application/json'
  }
});
```

### Step 3: Disable Auth (Optional for Dev)

If you want to disable authentication completely for rapid prototyping:
1.  Set `AUTH_REQUIRED=false` in `.env`.
2.  Restart the server.

## 4. Connectivity & Address Resolution

"Address Resolution: Both localhost and 127.0.0.1 failed" usually means the client is looking in the wrong place.

### Scenario A: Client in Browser (Host Machine) -> Server in Dev Container
-   **VS Code Port Forwarding**: Ensure port `18011` is forwarded.
    -   Check the "Ports" tab in VS Code.
    -   If "Visibility" is Private, you may need to authenticate in the browser first.
-   **Address**: Use `http://localhost:18011`.

### Scenario B: Client in Browser (Codespaces) -> Server in Codespace
-   **Address**: Use the forwarded URL provided by GitHub (e.g., `https://...-18011.app.github.dev`).
-   **CORS**: You MUST set `CORS_ORIGINS="*"` or add the Codespace URL to `.env`.

### Scenario C: Client in Container -> Server in Container
-   **Address**: Do NOT use `localhost`. Use the service name defined in `docker-compose.yml`.
    -   Example: `http://videoannotator-dev:18011`
-   **CORS**: Usually not an issue for server-to-server communication, but `CORS_ORIGINS` still applies if the client sends an `Origin` header.

## 5. Troubleshooting Checklist

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| `OPTIONS ... 400 Bad Request` | CORS Origin mismatch or invalid Host header. | Set `CORS_ORIGINS="*"` in `.env`. |
| `Connection Refused` | Port not forwarded or server not running. | Check VS Code Ports tab. Check server logs. |
| `401 Unauthorized` | Missing or invalid API Key. | Check `Authorization: Bearer <key>` header. |
| `403 Forbidden` | Valid key but insufficient permissions (rare in dev). | Check token scopes. |
| `Network Error` (Browser) | Mixed Content (HTTP vs HTTPS). | If using Codespaces (HTTPS), ensure client uses HTTPS URL for API. |
