// Submission batches (VideoAnnotator spec 008).
//
// A batch is not a server-side resource with a record of its own — it is
// "the set of jobs currently carrying a given batch_id". The client mints the
// id at submission time and sends it with every per-video upload in that
// submission; the server groups by it on read. That means a batch appears as
// soon as its first job exists, and stops existing when its last member job is
// deleted.
//
// These types are hand-written rather than derived from `schema.d.ts`: the
// generated schema is produced from a server OpenAPI spec that predates these
// endpoints. Same precedent as the extras-install types (see
// `ExtrasInstallJob` in `@/types/pipelines`). Regenerating the schema against a
// live server would supersede this file.

/** Per-state job counts within one batch. Always all five keys, zeroes included. */
export interface BatchStatusCounts {
  pending: number;
  running: number;
  completed: number;
  failed: number;
  cancelled: number;
}

/** Aggregate view of one batch, computed by the server on every read. */
export interface BatchSummary {
  batch_id: string;
  /** Human label given at submission. Null for batches submitted without one. */
  batch_name: string | null;
  /** Saved dataset the batch was submitted from, if any (spec 007). */
  dataset_id: string | null;
  /** Earliest member job's creation time — i.e. when the batch was submitted. */
  created_at: string | null;
  total: number;
  by_status: BatchStatusCounts;
  /** Percentage of member jobs in any terminal state (completed/failed/cancelled). */
  completion_percentage: number;
  /**
   * Null until at least one job in the batch has completed — the server has no
   * observed duration to extrapolate from before that. Render "estimating…"
   * rather than a zero.
   */
  estimated_seconds_remaining: number | null;
}

export interface BatchListResponse {
  batches: BatchSummary[];
  total: number;
  page: number;
  per_page: number;
}

/** One job a bulk action declined to touch, and why. */
export interface BatchSkippedJob {
  job_id: string;
  reason: string;
}

export interface BatchCancelResponse {
  batch_id: string;
  cancelled: string[];
  skipped: BatchSkippedJob[];
}

export interface BatchRetryResponse {
  batch_id: string;
  retried: string[];
  skipped: BatchSkippedJob[];
}

/** True once every member job has reached a terminal state. */
export function isBatchFinished(batch: BatchSummary): boolean {
  return batch.total > 0 && batch.by_status.pending + batch.by_status.running === 0;
}

/** True while any member job could still be cancelled. */
export function isBatchCancellable(batch: BatchSummary): boolean {
  return batch.by_status.pending + batch.by_status.running > 0;
}

/** True when at least one member job is in a state batch-retry would act on. */
export function isBatchRetryable(batch: BatchSummary): boolean {
  return batch.by_status.failed + batch.by_status.cancelled > 0;
}

/**
 * Display name for a batch. Falls back to a description of its size rather
 * than exposing a raw uuid, which tells a researcher nothing.
 */
export function batchDisplayName(batch: BatchSummary): string {
  if (batch.batch_name) return batch.batch_name;
  return batch.total === 1 ? '1 video' : `${batch.total} videos`;
}
