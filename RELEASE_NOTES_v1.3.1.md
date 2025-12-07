# Release Notes v1.3.1

**Date:** December 7, 2025
**Version:** 1.3.1

## Overview

This patch release focuses on improving the developer experience for frontend clients, specifically addressing CORS (Cross-Origin Resource Sharing) issues in cloud development environments (like GitHub Codespaces) and standardizing API URL routing.

## Key Changes

### 1. CORS & Connectivity Improvements
-   **Wildcard Origin Support**: Updated the server configuration to support `CORS_ORIGINS="*"` which now correctly enables `Access-Control-Allow-Credentials` with wildcard matching (via regex). This is critical for Codespaces and dynamic dev environments.
-   **Documentation**: Added `docs/development/CORS_AND_AUTH_PROTOCOL.md` detailing the standard protocol for connecting clients to the server.
-   **Logging**: Enhanced startup logs to explicitly show the active CORS configuration.

### 2. API Routing Standardization
-   **Trailing Slash Support**: The API now gracefully handles URLs with or without trailing slashes (e.g., `/api/v1/jobs` and `/api/v1/jobs/` are treated equivalently).
-   **Redirects**: Enabled `redirect_slashes=True` in the FastAPI application to automatically redirect clients to the canonical URL structure.
-   **Explicit Routes**: Added explicit route handlers for trailing slash variants on key endpoints (`jobs`, `pipelines`) to prevent 307 Temporary Redirects from causing issues with some strict HTTP clients or proxies.

### 3. Configuration
-   **Default Dev Config**: The `.env` file now defaults to `CORS_ORIGINS="*"` to reduce friction for new developers.

## Upgrade Instructions

No database migrations or breaking changes.
1.  Pull the latest code.
2.  Update your `.env` file if you are experiencing CORS issues (set `CORS_ORIGINS="*"`).
3.  Restart the server: `uv run videoannotator --dev`.
