/**
 * How the job wizard was opened (spec 019, and spec 018's datasets): passed as
 * React Router state by the pages that send people there.
 */

export interface StartSettings {
  selectedPipelines?: string[] | null;
  config?: Record<string, unknown> | null;
}

export type WizardStart =
  /** "Edit and run again": the job's own video, its settings to change. */
  | ({ mode: 'rerun'; jobId: string; label: string } & StartSettings)
  /** The same for every finished job of a batch. */
  | ({ mode: 'rerunBatch'; batchId: string; label: string; videoCount: number } & StartSettings)
  /** "Use these settings": the settings, new videos. */
  | ({ mode: 'settings'; label: string } & StartSettings)
  /** "Start a job" from the Datasets page. */
  | { mode: 'dataset'; datasetId: string; name: string };

export const wizardState = (start: WizardStart) => ({ wizardStart: start });

/** Reads the wizard's start, including the shapes older pages sent. */
export function wizardStartOf(state: unknown): WizardStart | null {
  if (!state || typeof state !== 'object') return null;
  const s = state as Record<string, unknown>;
  if (s.wizardStart && typeof s.wizardStart === 'object') return s.wizardStart as WizardStart;
  const dataset = s.startFromDataset as { id?: string; name?: string } | undefined;
  if (dataset?.id) return { mode: 'dataset', datasetId: dataset.id, name: dataset.name ?? '' };
  if (typeof s.retryJobId === 'string') {
    return {
      mode: 'rerun',
      jobId: s.retryJobId,
      label: (s.retryJobVideoFilename as string | undefined) ?? s.retryJobId,
      selectedPipelines: s.retryJobPipelines as string[] | undefined,
      config: s.retryJobConfig as Record<string, unknown> | undefined,
    };
  }
  return null;
}

/** A job's settings, ready to start the wizard from. */
export function settingsOf(job: { selected_pipelines?: string[] | null; config?: Record<string, unknown> | null }): StartSettings {
  return { selectedPipelines: job.selected_pipelines ?? null, config: job.config ?? null };
}
