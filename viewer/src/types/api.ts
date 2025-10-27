// API types for VideoAnnotator v1.3.0
// Types for enhanced server features: cancellation, validation, enhanced health

// =============================================================================
// ERROR HANDLING (v1.3.0)
// =============================================================================

/**
 * Field-level error detail from config validation
 */
export interface FieldError {
  field: string;
  message: string;
  error_code?: string;
  hint?: string;
}

/**
 * Enhanced error envelope (v1.3.0 standard format)
 * Supports both string errors and structured field errors
 */
export interface ErrorEnvelope {
  error: string | FieldError[];
  error_code?: string;
  request_id?: string;
  hint?: string;
}

/**
 * Parsed error for consistent UI display
 */
export interface ParsedError {
  message: string;
  code?: string;
  requestId?: string;
  hint?: string;
  fieldErrors?: FieldError[];
}

// =============================================================================
// SERVER CAPABILITIES (v1.3.0)
// =============================================================================

/**
 * Server capability detection result
 */
export interface ServerCapabilities {
  version: string;
  supportsJobCancellation: boolean;
  supportsConfigValidation: boolean;
  supportsEnhancedHealth: boolean;
  supportsEnhancedErrors: boolean;
  detectedAt: Date;
}

// =============================================================================
// JOB CANCELLATION (v1.3.0)
// =============================================================================

/**
 * Job cancellation request payload
 */
export interface CancelJobRequest {
  reason?: string;
}

/**
 * Job cancellation response
 */
export interface JobCancellationResponse {
  job_id: string;
  status: 'cancelled' | 'cancelling';
  message: string;
  cancelled_at?: string;
}

// =============================================================================
// CONFIG VALIDATION (v1.3.0)
// =============================================================================

/**
 * Configuration validation request
 */
export interface ConfigValidationRequest {
  config: Record<string, unknown>;
  pipeline_name?: string;
}

/**
 * Validation issue severity
 */
export type ValidationSeverity = 'error' | 'warning' | 'info';

/**
 * Single validation issue
 */
export interface ValidationIssue {
  field: string;
  message: string;
  severity: ValidationSeverity;
  error_code?: string;
  hint?: string;
  suggested_value?: unknown;
}

/**
 * Configuration validation result
 */
export interface ConfigValidationResult {
  valid: boolean;
  errors: ValidationIssue[];
  warnings: ValidationIssue[];
  validated_config?: Record<string, unknown>;
}

// =============================================================================
// ENHANCED HEALTH (v1.3.0)
// =============================================================================

/**
 * GPU status information
 */
export interface GpuStatus {
  available: boolean;
  device_name?: string;
  cuda_version?: string;
  memory_total?: number;
  memory_used?: number;
  memory_free?: number;
}

/**
 * Worker/queue status information
 */
export interface WorkerInfo {
  active_jobs: number;
  queued_jobs: number;
  max_concurrent_jobs: number;
  worker_status: 'idle' | 'busy' | 'overloaded';
}

/**
 * System diagnostics information
 */
export interface SystemDiagnostics {
  database: {
    status: 'healthy' | 'degraded' | 'unhealthy';
    message?: string;
  };
  storage: {
    status: 'healthy' | 'degraded' | 'unhealthy';
    disk_usage_percent?: number;
    message?: string;
  };
  ffmpeg: {
    available: boolean;
    version?: string;
  };
}

/**
 * Enhanced health check response (v1.3.0)
 */
export interface EnhancedHealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version: string;
  api_version: string;
  uptime_seconds: number;
  gpu_status?: GpuStatus;
  worker_info?: WorkerInfo;
  diagnostics?: SystemDiagnostics;
  message?: string;
}

/**
 * Legacy health response (v1.2.x compatibility)
 */
export interface LegacyHealthResponse {
  status: string;
  api_version?: string;
  videoannotator_version?: string;
  message?: string;
}

/**
 * Union type for health responses
 */
export type HealthResponse = EnhancedHealthResponse | LegacyHealthResponse;

// =============================================================================
// JOB MANAGEMENT (Extended for v1.3.0)
// =============================================================================

/**
 * Job status including new cancelled state
 */
export type JobStatus = 
  | 'pending'
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled';

/**
 * Cancellable job statuses
 */
export type CancellableStatus = 'pending' | 'queued' | 'running';

/**
 * Check if a job status allows cancellation
 */
export function isCancellable(status: JobStatus): status is CancellableStatus {
  return status === 'pending' || status === 'queued' || status === 'running';
}

/**
 * Extended job interface with cancellation support
 */
export interface Job {
  job_id: string;
  status: JobStatus;
  created_at: string;
  updated_at?: string;
  completed_at?: string;
  cancelled_at?: string;
  pipeline_name: string;
  config: Record<string, unknown>;
  input_file?: string;
  output_file?: string;
  error_message?: string;
  progress?: number;
  logs?: string[];
}

// =============================================================================
// CACHE KEYS (React Query)
// =============================================================================

/**
 * React Query cache key factories
 */
export const QueryKeys = {
  serverCapabilities: ['server', 'capabilities'] as const,
  health: ['server', 'health'] as const,
  jobs: ['jobs'] as const,
  job: (jobId: string) => ['jobs', jobId] as const,
  configValidation: (configHash: string) => ['config', 'validation', configHash] as const,
  pipelines: ['pipelines'] as const,
} as const;

// =============================================================================
// TYPE GUARDS
// =============================================================================

/**
 * Type guard for enhanced health response
 */
export function isEnhancedHealthResponse(
  response: HealthResponse
): response is EnhancedHealthResponse {
  return 'uptime_seconds' in response || 'gpu_status' in response || 'worker_info' in response;
}

/**
 * Type guard for error envelope
 */
export function isErrorEnvelope(obj: unknown): obj is ErrorEnvelope {
  return (
    typeof obj === 'object' &&
    obj !== null &&
    'error' in obj
  );
}

/**
 * Type guard for field errors
 */
export function isFieldErrorArray(error: unknown): error is FieldError[] {
  return (
    Array.isArray(error) &&
    error.length > 0 &&
    typeof error[0] === 'object' &&
    'field' in error[0] &&
    'message' in error[0]
  );
}
