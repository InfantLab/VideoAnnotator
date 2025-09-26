/**
 * Shared type definitions for VideoAnnotator v1.2.x pipeline discovery
 * and parameter schema introspection.
 */

export type PipelineParameterType =
  | 'boolean'
  | 'string'
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
}

export interface PipelineSchemaResponse {
  pipeline: PipelineDescriptor;
  parameters: PipelineParameterSchema[];
}

export interface PipelineCatalogCacheEntry {
  catalog: PipelineCatalog;
  server: VideoAnnotatorServerInfo;
  fetchedAt: number;
}

