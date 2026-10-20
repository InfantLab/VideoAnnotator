/**
 * What New run is opened on (spec 029 R9): some videos, a project, a folder,
 * a run to repeat, or nothing (then it asks). Passed as router state; the old
 * wizard's shapes (`lib/wizardStart.ts`) are still read, so older links and
 * pages land in the right place (R13).
 */
import { wizardStartOf, type StartSettings } from '@/lib/wizardStart';

export type RunTarget =
  | { kind: 'videos'; paths: string[]; label: string; projectId?: string | null; projectName?: string | null }
  | { kind: 'project'; id: string; name: string }
  | { kind: 'folder'; path: string; recursive: boolean }
  /** "Run again…": a run (batch id) or a run of one video (job id), its settings to change. */
  | { kind: 'rerun'; runId: string; label: string; single: boolean; videoCount?: number };

export interface RunStart {
  target: RunTarget | null;
  /** Settings to start from instead of the usual order (Ask AI's "Run on more videos", "Use these settings"). */
  settings?: StartSettings & { label?: string };
  /** A workflow chosen before opening New run (Pipelines → Use in a new run). */
  workflowId?: string;
}

export const runState = (start: RunStart) => ({ runStart: start });

/** Reads New run's start from router state, in any shape a page sent it. */
export function runStartOf(state: unknown): RunStart {
  if (state && typeof state === 'object' && 'runStart' in state) {
    const start = (state as { runStart?: RunStart }).runStart;
    if (start && typeof start === 'object') return { target: start.target ?? null, settings: start.settings, workflowId: start.workflowId };
  }
  const old = wizardStartOf(state);
  if (!old) return { target: null };
  switch (old.mode) {
    case 'rerun':
      return {
        target: { kind: 'rerun', runId: old.jobId, label: old.label, single: true },
        settings: { selectedPipelines: old.selectedPipelines, config: old.config },
      };
    case 'rerunBatch':
      return {
        target: { kind: 'rerun', runId: old.batchId, label: old.label, single: false, videoCount: old.videoCount },
        settings: { selectedPipelines: old.selectedPipelines, config: old.config },
      };
    case 'dataset':
      return { target: { kind: 'project', id: old.datasetId, name: old.name } };
    case 'settings':
      return { target: null, settings: { selectedPipelines: old.selectedPipelines, config: old.config, label: old.label } };
  }
}

/** The project a run on this target belongs to, if any. */
export function projectOf(target: RunTarget | null): string | null {
  if (!target) return null;
  if (target.kind === 'project') return target.id;
  if (target.kind === 'videos') return target.projectId ?? null;
  return null;
}

/** "3 videos", "Peekaboo study", "/data/pk". */
export function targetLabel(target: RunTarget): string {
  switch (target.kind) {
    case 'videos':
      return target.label;
    case 'project':
      return target.name;
    case 'folder':
      return target.path;
    case 'rerun':
      return target.label;
  }
}
