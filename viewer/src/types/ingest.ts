// Server-side folder ingest (VideoAnnotator `api/v1/ingest.py`).
//
// Uploading a corpus one multipart request at a time is the single biggest
// piece of friction in running a batch: forty videos means forty uploads with
// the tab held open. When the server runs on the researcher's own machine —
// the normal case — the files are already on a disk it can see, so a job can
// simply reference one where it is. These types describe that path.
//
// Hand-written rather than derived from `schema.d.ts`, same as `./batches`:
// the generated schema predates these endpoints.

export interface IngestDirectory {
  name: string;
  path: string;
  /** Videos directly inside this folder; not recursive. */
  video_count: number;
}

export interface IngestVideo {
  name: string;
  path: string;
  size_bytes: number | null;
}

export interface IngestBrowseResponse {
  /** Folder listed, or null when listing the roots. */
  path: string | null;
  /** Parent folder, or null at a root — there is nothing above it to browse. */
  parent: string | null;
  /** Folders the server is willing to read from at all. */
  roots: string[];
  directories: IngestDirectory[];
  videos: IngestVideo[];
  video_count: number;
  truncated: boolean;
}

export interface IngestRequest {
  path: string;
  recursive?: boolean;
  selected_pipelines?: string[];
  config?: Record<string, unknown>;
  batch_id?: string;
  batch_name?: string;
  dataset_id?: string;
}

/** A file ingest declined to turn into a job, and why. */
export interface IngestSkipped {
  filename: string;
  reason: string;
}

export interface IngestResponse {
  batch_id: string;
  batch_name: string | null;
  path: string;
  total: number;
  created: string[];
  skipped: IngestSkipped[];
}

/**
 * Whether this server offers folder ingest to this client.
 *
 * The endpoint is admin-only and refuses callers that aren't on the server's
 * own machine, and older servers don't have it at all — so rather than
 * predicting any of that, we ask once and treat every failure the same way:
 * the feature is unavailable, upload is still there, say so plainly.
 */
export interface IngestAvailability {
  available: boolean;
  /** Why not, when we know — shown to explain the missing option. */
  reason?: string;
}
