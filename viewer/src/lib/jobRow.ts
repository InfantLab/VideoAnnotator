// Per-video row fields shared by the flat jobs table and the batch table.

import type { JobResponse } from '@/api/client';
import { isCompletedWithErrors } from '@/lib/jobOutcome';

export function formatDuration(seconds: number | null) {
  if (!seconds) return 'N/A';
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

export function formatFileSize(bytes: number | null) {
  if (!bytes) return 'N/A';
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const getString = (value: unknown): string | undefined =>
  typeof value === 'string' ? value : undefined;

export const getNumber = (value: unknown): number | null =>
  typeof value === 'number' && Number.isFinite(value) ? value : null;

/**
 * Defensive field access — the server has used several names for these over
 * its versions, and a viewer pointed at an older server should still show a
 * filename rather than "N/A".
 */
export function videoFieldsOf(job: JobResponse) {
  const record = job as JobResponse & Record<string, unknown>;

  let videoName =
    getString(record.video_filename) ??
    getString(record.filename) ??
    getString(record.video_name);

  const videoPath = getString(record.video_path);
  if (!videoName && videoPath) {
    videoName = videoPath.split(/[/\\]/).pop() || videoPath;
  }

  return {
    videoName: videoName || 'N/A',
    videoDuration:
      getNumber(record.video_duration_seconds) ?? getNumber(record.duration_seconds),
    videoSize: getNumber(record.video_size_bytes) ?? getNumber(record.file_size_bytes),
  };
}

export function progressOf(job: JobResponse): number {
  return getNumber((job as JobResponse & Record<string, unknown>).progress_percentage) ?? 0;
}

/** Where a video stands, as the batch page groups and filters them. */
export type VideoOutcome = 'failed' | 'errors' | 'running' | 'queued' | 'done' | 'cancelled';

export const OUTCOME_ORDER: VideoOutcome[] = ['failed', 'errors', 'running', 'queued', 'done', 'cancelled'];

export const OUTCOME_LABELS: Record<VideoOutcome, string> = {
  failed: 'failed',
  errors: 'with errors',
  running: 'running',
  queued: 'queued',
  done: 'done',
  cancelled: 'cancelled',
};

export function outcomeOf(job: { status: string; error_message?: string | null }): VideoOutcome {
  if (job.status === 'failed') return 'failed';
  if (isCompletedWithErrors(job)) return 'errors';
  if (job.status === 'completed') return 'done';
  if (job.status === 'pending') return 'queued';
  if (job.status === 'cancelled') return 'cancelled';
  return 'running';
}

/**
 * The pipelines every video in a batch ran, or null when they differ (a
 * video retried with other settings). A batch is normally one submission, so
 * these are said once for the whole run rather than repeated on every row.
 */
export function sharedPipelines(jobs: JobResponse[]): string[] | null {
  if (jobs.length === 0) return null;
  const key = (job: JobResponse) => [...(job.selected_pipelines ?? [])].sort().join('\n');
  const first = key(jobs[0]);
  return jobs.every((job) => key(job) === first) ? [...(jobs[0].selected_pipelines ?? [])] : null;
}

export function filterJobs(
  jobs: JobResponse[],
  outcome: VideoOutcome | 'all',
  query: string
): JobResponse[] {
  const needle = query.trim().toLowerCase();
  return jobs.filter(
    (job) =>
      (outcome === 'all' || outcomeOf(job) === outcome) &&
      (!needle || videoFieldsOf(job).videoName.toLowerCase().includes(needle))
  );
}
