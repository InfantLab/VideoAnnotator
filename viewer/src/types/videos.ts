/**
 * Videos and projects (VideoAnnotator spec 029): shapes from the server's
 * OpenAPI schema. The API keeps its words (datasets, presets, batches, jobs);
 * the viewer says project, workflow and run.
 */
import type { components } from '@/api/schema';

type Schemas = components['schemas'];

export type VideosResponse = Schemas['VideosResponse'];
export type VideoItem = Schemas['VideoItem'];
export type VideoResult = Schemas['VideoResult'];
export type VideoCounts = Schemas['VideoCounts'];
export type ProjectCount = Schemas['ProjectCount'];
export type FolderCount = Schemas['FolderCount'];
export type VideoScope = Schemas['VideoScope'];
export type RunSummary = Schemas['RunSummary'];
export type ActivityRun = Schemas['ActivityRun'];
export type Activity = Schemas['ActivityResponse'];
export type RunSettings = Schemas['RunSettingsResponse'];
export type RunSettingsSource = Schemas['RunSettingsSource'];
export type VideosRunRequest = Schemas['VideosRunRequest'];
export type VideosRunResponse = Schemas['VideosRunResponse'];
export type ProjectSettings = Schemas['RunSettings'];

/** One scope of the Videos page (none: all videos). */
export interface VideosQuery {
  project?: string;
  folder?: string;
  recursive?: boolean;
  notInProject?: boolean;
  run?: string;
  q?: string;
  offset?: number;
  limit?: number;
}

/** A workflow's export choices (spec 029 R7); absent means the defaults. */
export interface ExportChoices {
  table?: 'csv' | 'parquet';
  elan?: boolean;
  methods?: boolean;
}

/** What a workflow or a project's settings came from. */
export interface SettingsSource {
  kind: 'project' | 'run' | 'edit' | 'workflow';
  id?: string | null;
  name?: string | null;
}
