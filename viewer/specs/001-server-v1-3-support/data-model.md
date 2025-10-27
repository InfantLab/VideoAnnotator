# Data Model: VideoAnnotator Server v1.3.0 Integration

**Feature**: VideoAnnotator Server v1.3.0 Client Support  
**Phase**: 1 (Data Structures & Type Definitions)  
**Date**: 2025-10-27

## Purpose

This document defines the TypeScript types, Zod schemas, and data structures for integrating VideoAnnotator server v1.3.0 features. All types follow strict TypeScript conventions and include runtime validation via Zod schemas.

---

## 1. Server Capabilities

Represents detected server version and available features.

### TypeScript Interface

```typescript
interface ServerCapabilities {
  /** Server version string (e.g., "1.3.0", "1.2.5") */
  version: string;
  
  /** Whether server supports job cancellation endpoint */
  supportsCancellation: boolean;
  
  /** Whether server supports configuration validation endpoints */
  supportsValidation: boolean;
  
  /** Whether server returns ErrorEnvelope format */
  supportsEnhancedErrors: boolean;
  
  /** Whether health endpoint includes GPU/worker diagnostics */
  supportsEnhancedHealth: boolean;
  
  /** Timestamp when capabilities were last detected */
  detectedAt: Date;
}
```

### Zod Schema

```typescript
const ServerCapabilitiesSchema = z.object({
  version: z.string(),
  supportsCancellation: z.boolean(),
  supportsValidation: z.boolean(),
  supportsEnhancedErrors: z.boolean(),
  supportsEnhancedHealth: z.boolean(),
  detectedAt: z.date(),
});
```

### Validation Rules
- `version` must be non-empty string
- All boolean flags default to `false` if detection fails
- `detectedAt` used to determine cache freshness (refresh if > 5 minutes old)

### State Transitions
1. **Initial**: `null` (not detected)
2. **Detecting**: Loading state during health endpoint fetch
3. **Detected**: Populated with server capabilities
4. **Stale**: `detectedAt` > 5 minutes; show refresh prompt
5. **Error**: Detection failed; assume v1.2.x defaults

---

## 2. Error Envelope (v1.3.0)

Enhanced error response format with field-level details and hints.

### TypeScript Interface

```typescript
interface ErrorEnvelope {
  /** Unique error code for programmatic handling */
  error_code?: string;
  
  /** Human-readable error message */
  message: string;
  
  /** Field name for field-level validation errors */
  field?: string;
  
  /** Helpful hint for resolution */
  hint?: string;
  
  /** HTTP status code */
  status?: number;
  
  /** Request ID for support/debugging */
  request_id?: string;
}

/** Parsed error for internal use */
interface ParsedError {
  message: string;
  code?: string;
  field?: string;
  hint?: string;
  isStructured: boolean; // true if v1.3.0 format, false if legacy
}
```

### Zod Schema

```typescript
const ErrorEnvelopeSchema = z.object({
  error_code: z.string().optional(),
  message: z.string(),
  field: z.string().optional(),
  hint: z.string().optional(),
  status: z.number().optional(),
  request_id: z.string().optional(),
});

const LegacyErrorSchema = z.union([
  z.string(),
  z.object({ error: z.string() }),
  z.object({ message: z.string() }),
]);
```

### Validation Rules
- `message` is required (always present in both formats)
- `error_code` follows SCREAMING_SNAKE_CASE convention
- `field` uses dot notation for nested fields (e.g., "scene_detection.threshold")
- `hint` is plain text suggestion (no HTML/markdown)

### Display Logic
- Show `message` prominently
- Show `field` inline near form input if present
- Show `hint` below message in muted color
- Show `error_code` only in dev mode or detailed error view
- Show `request_id` in "Report Issue" flow

---

## 3. Configuration Validation Result

Response from `/api/v1/config/validate` endpoint.

### TypeScript Interface

```typescript
interface ValidationError {
  /** Field name with dot notation */
  field: string;
  
  /** Error message */
  message: string;
  
  /** Error code for programmatic handling */
  code: string;
  
  /** Helpful hint for correction */
  hint?: string;
}

interface ValidationWarning {
  /** Field name with dot notation */
  field: string;
  
  /** Warning message */
  message: string;
  
  /** Warning code */
  code: string;
  
  /** Helpful hint */
  hint?: string;
}

interface ConfigValidationResult {
  /** Whether configuration is valid */
  valid: boolean;
  
  /** List of validation errors (blocking) */
  errors: ValidationError[];
  
  /** List of validation warnings (non-blocking) */
  warnings: ValidationWarning[];
  
  /** Pipelines that were validated */
  pipelines_validated: string[];
  
  /** Timestamp of validation */
  validated_at?: Date;
}
```

### Zod Schema

```typescript
const ValidationErrorSchema = z.object({
  field: z.string(),
  message: z.string(),
  code: z.string(),
  hint: z.string().optional(),
});

const ValidationWarningSchema = z.object({
  field: z.string(),
  message: z.string(),
  code: z.string(),
  hint: z.string().optional(),
});

const ConfigValidationResultSchema = z.object({
  valid: z.boolean(),
  errors: z.array(ValidationErrorSchema),
  warnings: z.array(ValidationWarningSchema),
  pipelines_validated: z.array(z.string()),
  validated_at: z.date().optional(),
});
```

### Validation Rules
- `valid` is `false` if `errors.length > 0`
- `warnings` do not affect `valid` status
- `field` uses dot notation matching config structure
- `code` follows SCREAMING_SNAKE_CASE (e.g., "VALUE_OUT_OF_RANGE", "REQUIRED_FIELD_MISSING")

### State Transitions
1. **Idle**: No validation performed yet
2. **Validating**: API call in progress
3. **Valid**: `valid: true`, no errors
4. **Invalid**: `valid: false`, errors present
5. **Valid with Warnings**: `valid: true`, warnings present

### Display Logic
- **Errors**: Show inline near relevant form fields; disable submit button
- **Warnings**: Show in separate warnings panel; allow submission with confirmation
- **Hints**: Display below error/warning message in lighter text
- Group errors/warnings by pipeline if multi-pipeline config

---

## 4. Job Cancellation Request/Response

### TypeScript Interface

```typescript
interface JobCancellationRequest {
  /** Job ID to cancel */
  job_id: string;
  
  /** Optional reason for cancellation (for logging) */
  reason?: string;
}

interface JobCancellationResponse {
  /** Job ID */
  id: string;
  
  /** Updated status ("cancelled") */
  status: "cancelled";
  
  /** Error message (always "Job cancelled by user request" or similar) */
  error_message: string;
  
  /** Timestamp when job was cancelled */
  completed_at: string; // ISO 8601
  
  /** Optional metadata */
  metadata?: Record<string, unknown>;
}
```

### Zod Schema

```typescript
const JobCancellationResponseSchema = z.object({
  id: z.string(),
  status: z.literal("cancelled"),
  error_message: z.string(),
  completed_at: z.string().datetime(),
  metadata: z.record(z.unknown()).optional(),
});
```

### Validation Rules
- `job_id` must be valid UUID or server-assigned ID format
- `status` must be exactly "cancelled" (lowercase)
- `completed_at` must be valid ISO 8601 timestamp
- `reason` is optional; default to "Cancelled by user" if not provided

### State Transitions
1. **Running/Pending**: Cancellable states
2. **Cancelling**: Optimistic UI state during API call
3. **Cancelled**: Final state after successful cancellation
4. **Error**: Cancellation failed; revert to previous state

### UI Logic
- Show cancel button only for `status in ["pending", "running", "queued"]`
- Disable cancel button for `status in ["completed", "failed", "cancelled"]`
- Show confirmation dialog before cancellation (prevent accidental clicks)
- Display optimistic "Cancelling..." state immediately after confirmation
- Show success toast with "Job cancelled successfully" message
- Show error toast if cancellation fails with specific error message

---

## 5. Enhanced Health Response (v1.3.0)

### TypeScript Interface

```typescript
interface GPUInfo {
  available: boolean;
  cuda_version?: string;
  device_count?: number;
  device_names?: string[];
  memory_total?: number; // bytes
  memory_available?: number; // bytes
}

interface WorkerInfo {
  active_jobs: number;
  max_concurrent: number;
  queue_depth: number;
  worker_status: "healthy" | "degraded" | "offline";
}

interface EnhancedHealthResponse {
  /** Basic health status */
  status: "ok" | "degraded" | "error";
  
  /** Status message */
  message: string;
  
  /** Server version (v1.3.0+) */
  version?: string;
  
  /** Server uptime in seconds */
  uptime?: number;
  
  /** GPU information (v1.3.0+) */
  gpu_status?: GPUInfo;
  
  /** Worker information (v1.3.0+) */
  worker_info?: WorkerInfo;
  
  /** Diagnostic details */
  diagnostics?: {
    database_connected: boolean;
    storage_accessible: boolean;
    ffmpeg_available: boolean;
  };
  
  /** Timestamp */
  timestamp: string; // ISO 8601
}
```

### Zod Schema

```typescript
const GPUInfoSchema = z.object({
  available: z.boolean(),
  cuda_version: z.string().optional(),
  device_count: z.number().optional(),
  device_names: z.array(z.string()).optional(),
  memory_total: z.number().optional(),
  memory_available: z.number().optional(),
});

const WorkerInfoSchema = z.object({
  active_jobs: z.number(),
  max_concurrent: z.number(),
  queue_depth: z.number(),
  worker_status: z.enum(["healthy", "degraded", "offline"]),
});

const EnhancedHealthResponseSchema = z.object({
  status: z.enum(["ok", "degraded", "error"]),
  message: z.string(),
  version: z.string().optional(),
  uptime: z.number().optional(),
  gpu_status: GPUInfoSchema.optional(),
  worker_info: WorkerInfoSchema.optional(),
  diagnostics: z.object({
    database_connected: z.boolean(),
    storage_accessible: z.boolean(),
    ffmpeg_available: z.boolean(),
  }).optional(),
  timestamp: z.string().datetime(),
});
```

### Validation Rules
- `status` enum ensures type safety
- Optional fields indicate v1.3.0 vs v1.2.x
- `gpu_status.available = false` means no GPU detected
- `worker_info.worker_status` reflects overall health

### Display Logic
- **v1.2.x**: Show only `status` and `message`
- **v1.3.0**: Show all fields in expandable diagnostics section
- Color-code `worker_status`: green (healthy), yellow (degraded), red (offline)
- Show GPU memory as percentage: `(available / total) * 100`
- Format uptime as human-readable (e.g., "3 days, 4 hours")
- Refresh every 30 seconds when diagnostics section is expanded

---

## 6. Enhanced Job Response (v1.3.0 Optional Fields)

### TypeScript Interface

```typescript
interface Job {
  id: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  created_at: string;
  updated_at: string;
  completed_at?: string;
  
  // Existing fields (v1.2.x)
  video_filename: string;
  selected_pipelines: string[];
  config?: Record<string, unknown>;
  error_message?: string;
  
  // New optional fields (v1.3.0)
  storage_path?: string; // Internal path on server
  retry_count?: number; // Number of retry attempts
  cancelled_by?: string; // User ID or "user_request"
  cancellation_reason?: string;
}
```

### Zod Schema

```typescript
const JobSchema = z.object({
  id: z.string(),
  status: z.enum(["pending", "running", "completed", "failed", "cancelled"]),
  created_at: z.string().datetime(),
  updated_at: z.string().datetime(),
  completed_at: z.string().datetime().optional(),
  video_filename: z.string(),
  selected_pipelines: z.array(z.string()),
  config: z.record(z.unknown()).optional(),
  error_message: z.string().optional(),
  storage_path: z.string().optional(),
  retry_count: z.number().optional(),
  cancelled_by: z.string().optional(),
  cancellation_reason: z.string().optional(),
});
```

### Validation Rules
- All v1.3.0 fields are optional (backward compatibility)
- `retry_count` starts at 0; max 3 retries (server enforced)
- `storage_path` is informational only (not used by client)
- `cancelled_by` and `cancellation_reason` only present if `status === "cancelled"`

### Display Logic
- Show `retry_count` in job detail view (if > 0)
- Show `cancellation_reason` in job detail view (if cancelled)
- `storage_path` displayed only in admin/debug mode
- Existing job list and detail views work unchanged (optional fields ignored)

---

## 7. Cache Keys & State Management

### React Query Keys

```typescript
// Query key factory for type safety
const queryKeys = {
  jobs: ['jobs'] as const,
  job: (id: string) => ['job', id] as const,
  health: ['server-health'] as const,
  capabilities: ['server-capabilities'] as const,
  validation: (config: unknown) => ['config-validation', hashConfig(config)] as const,
};
```

### Cache Invalidation Rules

| Action | Invalidates |
|--------|-------------|
| Job cancelled | `['job', jobId]`, `['jobs']` |
| Job submitted | `['jobs']` |
| Config changed | `['config-validation', *]` (all validation caches) |
| Server URL changed | All queries (reset QueryClient) |
| Manual refresh | `['server-health']`, `['server-capabilities']` |

### State Persistence

```typescript
// LocalStorage keys for persistence
const storageKeys = {
  API_URL: 'videoannotator_api_url',
  API_TOKEN: 'videoannotator_api_token',
  CAPABILITIES_CACHE: 'videoannotator_capabilities_cache',
  LAST_VALIDATION: 'videoannotator_last_validation', // Session storage
};
```

---

## Summary

### Type Safety Guarantees
- All API responses validated with Zod schemas at runtime
- TypeScript strict mode ensures compile-time type checking
- Backward compatibility: all v1.3.0 fields are optional

### Data Flow
1. **Capability Detection**: Health endpoint → `ServerCapabilities` → React Context
2. **Error Handling**: API error → `parseError()` → `ParsedError` → Error component
3. **Validation**: Form change → debounce → API call → `ConfigValidationResult` → UI feedback
4. **Cancellation**: Button click → confirmation → API call → `JobCancellationResponse` → UI update

### Performance Considerations
- React Query caching reduces redundant API calls
- Configuration hash prevents duplicate validation requests
- Optimistic updates improve perceived performance
- Lazy loading keeps bundle size minimal

---

**Data Model Complete**: All types, schemas, and validation rules defined. Ready for contract generation (Phase 1).
