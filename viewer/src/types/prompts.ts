/** The prompt library (VideoAnnotator spec 020). */
export interface LibraryPrompt {
  /** SHA-256 of the exact text: the same hash a VLM output's provenance records. */
  sha256: string;
  text: string;
  name?: string | null;
  tags: string[];
  starred: boolean;
  hidden: boolean;
  first_used_at: string;
  last_used_at: string;
  first_user_id?: string | null;
  updated_at?: string | null;
  updated_by?: string | null;
  models: string[];
  job_ids: string[];
  use_count: number;
}

export interface PromptListResponse {
  prompts: LibraryPrompt[];
  total: number;
}

export type PromptUpdateRequest = Partial<Pick<LibraryPrompt, 'name' | 'tags' | 'starred' | 'hidden'>>;
