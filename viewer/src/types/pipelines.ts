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
}

