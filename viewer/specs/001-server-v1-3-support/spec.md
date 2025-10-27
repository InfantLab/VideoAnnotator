# Feature Specification: VideoAnnotator Server v1.3.0 Client Support

**Feature Branch**: `001-server-v1-3-support`  
**Created**: 2025-10-27  
**Status**: Draft  
**Input**: User description: "Update client to support VideoAnnotator server v1.3.0 features including job cancellation, configuration validation, enhanced authentication, and improved error handling"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Job Cancellation (Priority: P1)

Users can cancel running or queued annotation jobs to free up GPU resources when they realize a job was submitted with incorrect configuration or is no longer needed.

**Why this priority**: Critical for production deployments where GPU resources are expensive and limited. Prevents resource waste and allows users to correct mistakes immediately without waiting for jobs to complete.

**Independent Test**: Submit a long-running job via the job creation wizard, navigate to the job detail page, and verify the cancel button appears and successfully terminates the job within 5 seconds, updating the status to "cancelled".

**Acceptance Scenarios**:

1. **Given** a user has submitted a video processing job, **When** they navigate to the job detail page and click the "Cancel Job" button, **Then** the job status updates to "cancelled" within 5 seconds and the UI displays a confirmation message.

2. **Given** a user is viewing the jobs list page, **When** they attempt to cancel an already-completed job, **Then** the cancel button is disabled or hidden and a tooltip explains that completed jobs cannot be cancelled.

3. **Given** a user cancels a running job, **When** they refresh the job detail page, **Then** the job status remains "cancelled" and displays the cancellation timestamp and reason.

4. **Given** a network error occurs during cancellation, **When** the user retries the cancel operation, **Then** the system handles the retry gracefully and provides clear feedback about the operation status.

---

### User Story 2 - Configuration Validation (Priority: P1)

Users can validate their pipeline configuration before submitting jobs, catching configuration errors immediately and receiving clear, actionable error messages with hints for correction.

**Why this priority**: Critical for user experience and resource efficiency. Prevents wasted processing time on invalid configurations and reduces frustration by providing immediate feedback with helpful suggestions.

**Independent Test**: Open the job creation wizard, enter an invalid configuration (e.g., confidence threshold of 1.5), and verify that validation errors appear with specific field names, error messages, and helpful hints before the submit button is enabled.

**Acceptance Scenarios**:

1. **Given** a user is configuring a person tracking pipeline, **When** they enter a confidence threshold value of 1.5 (out of range 0.0-1.0), **Then** the system displays a field-level error message: "Value must be between 0.0 and 1.0" with a hint "Try using 0.5 as a sensible default".

2. **Given** a user has entered invalid configuration values, **When** they attempt to submit the job, **Then** the submit button is disabled and all validation errors are clearly displayed with field-level specificity.

3. **Given** a user has corrected all validation errors, **When** the configuration passes validation, **Then** the submit button becomes enabled and any previous error messages are cleared.

4. **Given** a user is editing a complex multi-pipeline configuration, **When** they trigger validation, **Then** the system validates all selected pipelines and displays errors grouped by pipeline with clear indication of which pipeline each error belongs to.

---

### User Story 3 - Enhanced Authentication Management (Priority: P2)

Users can easily configure API authentication tokens, understand connection status at a glance, and receive clear guidance when authentication fails or tokens are invalid.

**Why this priority**: Important for production deployments with secure-by-default servers. Improves security posture while maintaining good user experience through clear status indicators and helpful error messages.

**Independent Test**: Configure a valid API token in settings, verify the token status indicator shows "connected", then change to an invalid token and verify clear error messages guide the user to correct the issue.

**Acceptance Scenarios**:

1. **Given** a user opens the settings page for the first time, **When** the server requires authentication and no token is configured, **Then** the UI displays a prominent setup guide with instructions for obtaining an API token from the server console.

2. **Given** a user has configured an API token, **When** the token is valid and the server is reachable, **Then** the token status indicator shows "connected" with a green indicator and displays server version information.

3. **Given** a user's API token expires or becomes invalid, **When** they attempt any API operation, **Then** the system displays a clear authentication error with guidance to check settings and update the token.

4. **Given** a user is working with a development server that has authentication disabled, **When** they access the application, **Then** the system detects authentication is optional and displays a warning indicator that the connection is unsecured.

---

### User Story 4 - Improved Error Handling (Priority: P2)

Users receive consistent, clear error messages across all operations with specific field-level details, error codes, and helpful hints for resolution when something goes wrong.

**Why this priority**: Significantly improves user experience by reducing confusion and support burden. Helps users self-diagnose and resolve issues without contacting support.

**Independent Test**: Trigger various error scenarios (network failure, validation error, authentication failure) and verify that each displays a consistent error format with appropriate detail level, error codes, and actionable guidance.

**Acceptance Scenarios**:

1. **Given** a network error occurs during job submission, **When** the error is displayed to the user, **Then** the error message clearly indicates a network issue, provides the HTTP status code, and suggests checking connectivity and server status.

2. **Given** a validation error occurs with multiple field issues, **When** the error is displayed, **Then** each field error is shown separately with the field name, specific error message, error code, and a helpful hint for correction.

3. **Given** an unexpected server error occurs, **When** the error is displayed, **Then** the message includes a request ID or timestamp for support reference and suggests checking server logs or contacting administrators.

4. **Given** a user encounters an error, **When** they view the error details, **Then** technical details are collapsible or hidden by default but accessible for troubleshooting, maintaining a clean primary UI.

---

### User Story 5 - Enhanced Health and Diagnostics (Priority: P3)

Users can view comprehensive server health information including GPU status, worker availability, and system capabilities directly from the settings page, helping them understand server status and troubleshoot connection issues.

**Why this priority**: Nice-to-have feature that improves transparency and helps advanced users troubleshoot issues independently. Not critical for basic operation but valuable for production deployments.

**Independent Test**: Navigate to settings page, expand the server diagnostics section, and verify display of server version, GPU status, worker information, and system capabilities with appropriate formatting and status indicators.

**Acceptance Scenarios**:

1. **Given** a user is on the settings page, **When** they expand the server diagnostics section, **Then** the system displays server version, uptime, GPU availability, active workers, and available pipelines in a well-organized format.

2. **Given** the server has GPU resources available, **When** viewing diagnostics, **Then** GPU information shows CUDA version, available memory, and GPU model with appropriate visual indicators.

3. **Given** the server is under heavy load, **When** viewing diagnostics, **Then** worker information shows the number of active jobs, queue depth, and concurrency limits with warning indicators if capacity is reached.

4. **Given** a server connection issue exists, **When** attempting to load diagnostics, **Then** the system gracefully handles the failure and displays cached or partial information with a clear indicator of stale data.

---

### Edge Cases

- What happens when a user attempts to cancel a job that has already completed or failed before the cancel request reaches the server?
- How does the system handle validation when the server is running an older version that doesn't support the validation endpoints?
- What happens when a user submits a job while their API token expires mid-request?
- How does the UI behave when validation returns warnings (non-blocking issues) versus errors (blocking issues)?
- What happens when network connectivity is lost during job cancellation and the user can't confirm if the operation succeeded?
- How does the system handle authentication scenarios where the server is initially reachable but becomes unreachable after token validation?
- What happens when validation rules change between server versions and cached validation state becomes stale?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Client MUST support the new job cancellation endpoint (`POST /api/v1/jobs/{job_id}/cancel`) with appropriate UI controls on job detail and jobs list pages.

- **FR-002**: Client MUST integrate configuration validation endpoints (`POST /api/v1/config/validate` and `POST /api/v1/pipelines/{name}/validate`) and display field-level validation errors before job submission.

- **FR-003**: Client MUST handle the enhanced error response format (`ErrorEnvelope`) consistently across all API operations, displaying field-level errors, error codes, and hints.

- **FR-004**: Client MUST support authentication-required mode by default, prompting users to configure API tokens and displaying clear authentication status indicators.

- **FR-005**: Client MUST detect and gracefully handle servers running older versions that lack v1.3.0 endpoints (job cancellation, config validation), falling back to previous behavior with appropriate user notifications.

- **FR-006**: Client MUST display enhanced health endpoint data including GPU status, worker information, and diagnostic details in the settings/diagnostics interface.

- **FR-007**: Client MUST update job status response handling to support new optional fields (`storage_path`, `retry_count`) without breaking existing functionality.

- **FR-008**: Client MUST provide clear user guidance when connecting to servers with authentication disabled, displaying security warnings appropriately.

- **FR-009**: Client MUST handle validation errors and warnings distinctly, allowing job submission with warnings but blocking submission with errors.

- **FR-010**: Client MUST cache validation results appropriately to avoid excessive server requests while ensuring fresh validation when configuration changes.

- **FR-011**: Users MUST be able to manually refresh server capabilities and health information from the settings page.

- **FR-012**: Client MUST handle network errors during cancellation operations gracefully, providing retry mechanisms and clear status feedback.

- **FR-013**: Client MUST display server version information in the UI to help users understand compatibility and available features.

- **FR-014**: Client MUST preserve backward compatibility with VideoAnnotator servers v1.2.x, detecting capabilities and adapting UI accordingly.

### Key Entities

- **Job Cancellation Request**: Represents a user's intent to cancel a job, including job ID, timestamp, and optional reason. Results in updated job status of "cancelled".

- **Configuration Validation Result**: Contains validation status (valid/invalid), list of errors with field names and messages, list of warnings, error codes, helpful hints, and pipelines validated.

- **Enhanced Error Response**: Structured error information including error code, message, field name (for field-level errors), hint text, HTTP status, and optional request ID.

- **Server Capabilities**: Information about server version, available endpoints, feature flags, GPU availability, worker status, and supported pipeline versions.

- **Authentication Status**: Current state of API authentication including token validity, server reachability, authentication mode (required/optional), and server version.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can successfully cancel running jobs with status confirmation appearing within 5 seconds of clicking the cancel button, tested across Chrome, Firefox, and Edge browsers.

- **SC-002**: Configuration validation displays field-level errors with helpful hints within 1 second of user input, preventing 100% of invalid job submissions that would have failed during processing.

- **SC-003**: Authentication setup completion rate reaches 95% for new users connecting to secure servers, measured by successful first API call after token configuration.

- **SC-004**: Error messages provide sufficient information for users to self-resolve 80% of common issues (invalid config, network errors, authentication failures) without contacting support.

- **SC-005**: Client successfully detects and adapts to both v1.2.x and v1.3.0 servers, with appropriate feature availability and zero breaking errors when connecting to older servers.

- **SC-006**: Server diagnostics information loads and displays within 2 seconds on the settings page, providing actionable health status to users.

- **SC-007**: Zero user-reported data loss or state inconsistency issues when using job cancellation under various network conditions and server load scenarios.

## Assumptions

- VideoAnnotator server v1.3.0 implements all documented endpoints and response formats as specified in the server update documentation.
- Server responses follow the documented ErrorEnvelope format consistently across all endpoints.
- API token authentication follows Bearer token standard as currently implemented in v1.2.x.
- Server capability detection can be performed via version information in health endpoint response.
- Configuration validation rules match the actual pipeline parameter constraints enforced during job execution.
- Network latency for API calls remains under 2 seconds under normal conditions.
- Users have JavaScript enabled and use modern browsers (Chrome 90+, Firefox 88+, Safari 14+, Edge 90+).

## Out of Scope

- Storage cleanup features (server-side only, no client UI required per server documentation).
- Enhanced diagnostics CLI commands (server-side only).
- Database migration handling (server handles automatically).
- Worker retry logic and exponential backoff (internal server concern).
- CORS configuration management (server environment variable, not client feature).
- SSL/TLS certificate setup (deployment/infrastructure concern).
- Custom pipeline development or pipeline catalog modifications.

## Dependencies

- VideoAnnotator server v1.3.0-dev deployment availability for testing.
- Server API documentation accuracy for new endpoints.
- Backward compatibility testing requires access to v1.2.x server instances.
- UI component library (shadcn/ui) supports required form validation patterns.
- Existing API client architecture (`src/api/client.ts`) supports extension for new endpoints.

## Risks

- **Server Version Detection**: May be challenging to reliably detect server version and capabilities without dedicated version endpoint. Mitigation: Use health endpoint response structure and fallback to feature detection.

- **Validation Performance**: Real-time validation on every config change could create excessive server load. Mitigation: Implement debouncing and intelligent caching strategies.

- **Cancellation Race Conditions**: Job might complete naturally just as user triggers cancellation. Mitigation: Handle "already completed" responses gracefully with clear user messaging.

- **Breaking Changes**: Server ErrorEnvelope format changes might break existing error handling. Mitigation: Implement defensive parsing with fallbacks to previous format.

- **Authentication UX**: Requiring tokens by default might increase initial setup friction. Mitigation: Provide excellent first-run guidance and clear setup instructions.
