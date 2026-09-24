/**
 * Shared type definitions for VideoAnnotator v1.2.x pipeline discovery
 * and parameter schema introspection.
 */

export type PipelineParameterType =
  | 'boolean'
  | 'string'
  | 'text'
  | 'integer'
  | 'number'
  | 'enum'
  | 'multiselect'
  | 'object';

export interface PipelineParameterOption {
  value: string | number | boolean;
  label?: string;
  description?: string;
}

export interface PipelineParameterSchema {
  name: string;
  type: PipelineParameterType;
  label?: string;
  description?: string;
  required?: boolean;
  default?: string | number | boolean | string[] | number[] | boolean[] | Record<string, unknown>;
  unit?: string;
  group?: string;
  advanced?: boolean;
  min?: number;
  max?: number;
  step?: number;
  pattern?: string;
  enum?: PipelineParameterOption[];
  dependencies?: string[];
}

export interface PipelineCapability {
  feature: string;
  enabled: boolean;
  details?: Record<string, unknown>;
}

export interface PipelineDescriptor {
  id: string;
  name: string;
  description?: string;
  group?: string;
  version?: string;
  model?: string;
  outputFormats?: string[];
  defaultEnabled?: boolean;
  capabilities?: PipelineCapability[];
  parameters?: PipelineParameterSchema[]; // Optional inline schema
  /** False when the pipeline's extras group isn't installed on the server. Absent (pre-v1.5.0 server) is treated as available. */
  available?: boolean;
  /** Human-readable install command (e.g. "pip install videoannotator[face]"), present only when available === false. */
  installHint?: string;
}

export interface PipelineCatalog {
  pipelines: PipelineDescriptor[];
  version?: string;
  generatedAt?: string;
  source?: string;
}

export interface VideoAnnotatorFeatureFlags {
  pipelineCatalog?: boolean;
  pipelineSchemas?: boolean;
  pipelineHealth?: boolean;
  jobSSE?: boolean;
  artifactListing?: boolean;
}

export interface VideoAnnotatorServerInfo {
  version: string;
  build?: string;
  commit?: string;
  features?: VideoAnnotatorFeatureFlags;
  capabilities?: PipelineCapability[];
  catalogVersion?: string;
  lastUpdated?: string;
}

export interface PipelineCatalogResponse {
  catalog: PipelineCatalog;
  server: VideoAnnotatorServerInfo;
  /** True when at least one extras group has finished installing but the server hasn't restarted to activate it yet. Absent (pre-v1.5.0 server) is treated as false. */
  restartRequired: boolean;
}

export interface PipelineSchemaResponse {
  pipeline: PipelineDescriptor;
  parameters: PipelineParameterSchema[];
}

export interface PipelineCatalogCacheEntry {
  catalog: PipelineCatalog;
  server: VideoAnnotatorServerInfo;
  restartRequired: boolean;
  fetchedAt: number;
}

/**
 * Lifecycle status of a pipeline-extras install job.
 * See specs/002-pipeline-extras-install/data-model.md#extrasinstalljob-new
 */
export type ExtrasInstallJobStatus = 'pending' | 'running' | 'completed' | 'failed';

/** Response from POST /api/v1/pipelines/extras/{extra}/install */
export interface ExtrasInstallTriggerResponse {
  jobId: string;
  extraName: string;
  status: ExtrasInstallJobStatus;
}

/** Response from GET /api/v1/pipelines/extras/install-jobs/{job_id} */
export interface ExtrasInstallJob {
  jobId: string;
  extraName: string;
  status: ExtrasInstallJobStatus;
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  commandOutput: string | null;
  restartRequired: boolean;
  /**
   * VideoAnnotator spec 011, set once completed: `live` = usable now, no restart;
   * `restart_required` = the install changed packages the server had loaded.
   * null from servers that predate it (treat as restart_required, per 005).
   */
  activation?: 'live' | 'restart_required' | string | null;
  conflictingDistributions?: { name: string; oldVersion: string; newVersion: string | null }[];
}

/**
 * Response from GET /api/v1/vlm/models. An unreachable Ollama server is a
 * thrown APIError (503), not a value in this type — see
 * specs/009-vlm-prompt-workflow (VideoAnnotator repo) FR-005: reachable vs.
 * unreachable are distinguished at the HTTP-status level, not folded into
 * this shape.
 */
export interface VlmModelsResponse {
  baseUrl: string;
  models: string[];
}

/** Request for POST /api/v1/vlm/preview — test a prompt against one frame
 * (or burst) without creating a job. Exactly one of `image` or
 * (`videoPath` + `timestampSec`) must be provided. */
export interface VlmPreviewRequest {
  image?: Blob;
  videoPath?: string;
  timestampSec?: number;
  prompt: string;
  model: string;
  samplingMode?: 'single_frame' | 'frame_burst';
  frameIntervalSec?: number;
  burstOffsets?: number[];
  think?: boolean;
  baseUrl?: string;
}

/** Response from POST /api/v1/vlm/preview. */
export interface VlmPreviewResponse {
  label: string;
  reasoning: string;
  rawResponse: string;
  totalTime: number;
  loadTime: number;
  promptTokens: number;
  respTokens: number;
  tokensPerSec: number;
}

