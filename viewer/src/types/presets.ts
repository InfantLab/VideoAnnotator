/**
 * Saved pipeline presets (VideoAnnotator spec 007): a named pipeline selection
 * plus its config, shared with everyone on the server.
 */
export interface Preset {
  id: string;
  name: string;
  description?: string | null;
  owner_user_id: string;
  selected_pipelines: string[];
  /** Same shape as a job submission's `config`, keyed by pipeline id. */
  config: Record<string, unknown>;
  tags?: Record<string, unknown>;
  created_at: string;
  updated_at?: string | null;
  last_used_at?: string | null;
  /** Pipelines it names that this server can't run right now. */
  unavailable_pipelines?: string[];
}

export interface PresetListResponse {
  presets: Preset[];
  total: number;
}

export interface PresetCreateRequest {
  name: string;
  description?: string;
  selected_pipelines: string[];
  config: Record<string, unknown>;
}
