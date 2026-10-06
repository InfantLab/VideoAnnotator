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
  /** Videos here or a few levels below (spec 022); absent on older servers. */
  has_videos?: boolean;
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
  /** Spec 022: only these videos, as paths relative to `path`. Omit for the whole folder. */
  files?: string[];
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
  /** The run's results folder (spec 022); absent on older servers. */
  results_folder?: FolderRef | null;
}

/** A location, and the same location as the researcher's own machine shows it. */
export interface FolderRef {
  path: string;
  /** Differs from `path` under Docker, where the server sees `/results/...`. */
  display_path: string;
}

/**
 * What this browser may do with videos on the server's machine (spec 022),
 * from `GET /api/v1/ingest/access`. The server decides "same machine" from how
 * the request reached it; the viewer never guesses from its own URL.
 */
export interface IngestAccess {
  same_machine: boolean;
  can_read_in_place: boolean;
  /** Why `can_read_in_place` is false, written for researchers. */
  reason: string | null;
  allowed_folders: FolderRef[];
  results_root: FolderRef;
  can_open_folders: boolean;
  /** Where My folders starts: Home, Videos, Desktop, ... that exist (spec 022). */
  places?: Place[];
  /** The server runs in a container (spec 024); absent on older servers. */
  in_container?: boolean;
  /** Started by `videoannotator-start`, which shares folders and can stop sharing them (spec 024). */
  managed_by_launcher?: boolean;
}

export interface Place extends FolderRef {
  /** "Home", "Videos", "Desktop", ... */
  label: string;
  has_videos: boolean;
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
