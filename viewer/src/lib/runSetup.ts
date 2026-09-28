/**
 * First-run setup a run was submitted with, and time-remaining that isn't
 * skewed by it.
 *
 * A run's first video pays for loading the pipelines' libraries and, the first
 * time, downloading model weights (DeepFace alone is ~1.1 GB). None of that
 * shows in job progress, so without saying so a run sits at "pending 0%" for
 * minutes and looks stuck. The server has no per-job stage yet, so the viewer
 * remembers the pipelines' `weights_not_cached` notes from the submit step and
 * explains the wait itself.
 */

import type { PipelineDescriptor, ReadinessItem } from '@/types/pipelines';

export interface SetupNote {
  pipeline: string;
  message: string;
  approxMb: number | null;
  /** The weights' own name, shared by pipelines that use the same model.
   * Absent on notes remembered before it was recorded. */
  name?: string;
}

/** The `weights_not_cached` notes of the selected pipelines. */
export function weightsNotesFor(pipelines: PipelineDescriptor[], selectedIds: string[]): SetupNote[] {
  return pipelines
    .filter((p) => selectedIds.includes(p.id))
    .flatMap((p) =>
      (p.readiness?.notes ?? [])
        .filter((note: ReadinessItem) => note.kind === 'weights_not_cached')
        .map((note) => ({
          pipeline: p.name,
          message: note.message,
          approxMb: note.approxMb ?? null,
          name: note.name || undefined,
        }))
    );
}

/**
 * "about 1.1 GB", or null when no note carries a size. Weights two pipelines
 * share (pyannote for audio_processing and speaker_diarization) download
 * once, so they count once.
 */
export function totalDownloadLabel(notes: SetupNote[]): string | null {
  const seen = new Set<string>();
  const mb = notes.reduce((sum, n) => {
    if (n.name) {
      if (seen.has(n.name)) return sum;
      seen.add(n.name);
    }
    return sum + (n.approxMb ?? 0);
  }, 0);
  if (mb <= 0) return null;
  return mb >= 1000 ? `about ${(mb / 1000).toFixed(1)} GB` : `about ${Math.round(mb)} MB`;
}

const storageKey = (batchId: string) => `vav.runSetupNotes.${batchId}`;

// Browser storage can be missing or throw (private windows, blocked site data).
// Losing these notes only loses an explanation, so failures are ignored.
export function rememberRunSetup(batchId: string, notes: SetupNote[]): void {
  if (notes.length === 0) return;
  try {
    localStorage.setItem(storageKey(batchId), JSON.stringify(notes));
  } catch {
    /* explanation only */
  }
}

export function recallRunSetup(batchId: string | undefined): SetupNote[] {
  if (!batchId) return [];
  try {
    const raw = localStorage.getItem(storageKey(batchId));
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

/**
 * Seconds left, from how fast videos finish *after* the first one.
 *
 * The first video's time includes one-off setup, so it overstates every later
 * one. Once two or more have finished, the rate between the first and last
 * completion measures real throughput (and allows for jobs running side by
 * side). With fewer than two, returns null: estimating from the first video
 * alone would be mostly setup.
 */
export function throughputSecondsRemaining(
  jobs: { status: string; completed_at?: string | null }[],
  remaining: number
): number | null {
  if (remaining <= 0) return null;
  const done = jobs
    .filter((j) => j.status === 'completed' && j.completed_at)
    .map((j) => new Date(j.completed_at as string).getTime())
    .filter((t) => Number.isFinite(t))
    .sort((a, b) => a - b);
  if (done.length < 2) return null;
  const spanSeconds = (done[done.length - 1] - done[0]) / 1000;
  if (spanSeconds <= 0) return null;
  const perVideo = spanSeconds / (done.length - 1);
  return perVideo * remaining;
}
