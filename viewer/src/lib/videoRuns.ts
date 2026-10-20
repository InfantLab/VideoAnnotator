/**
 * Which run each lane starts on (spec 028 FR-016, R11): per pipeline, the
 * newest job on this video whose result completed; a `run=` job wins where it
 * has that pipeline's results.
 */
import type { FailedPipeline, RunRef } from '@/lib/lanes';
import type { VideoRun } from '@/types/video';

/** pipeline name → job id */
export type RunSelection = Record<string, string>;

const completed = (run: VideoRun, pipeline: string) =>
  run.pipelines.some((p) => p.name === pipeline && p.status === 'completed' && p.files.some((f) => f.kind === 'result'));

/** `runs` newest first, as `/videos/runs` lists them. */
export function defaultSelection(runs: VideoRun[], preferred?: string | null): RunSelection {
  const selection: RunSelection = {};
  const pipelines = new Set(runs.flatMap((r) => r.pipelines.map((p) => p.name)));
  const first = runs.find((r) => r.job_id === preferred);
  for (const pipeline of pipelines) {
    const run = (first && completed(first, pipeline) ? first : undefined) ?? runs.find((r) => completed(r, pipeline));
    if (run) selection[pipeline] = run.job_id;
  }
  return selection;
}

export const MISSING_RESULTS =
  "Finished, but its result files are no longer in the run's results folder (moved or deleted). Run it again to get them back.";

/**
 * Pipelines with nothing to show on this video, with the newest run's reason:
 * those that failed, and those that finished but whose files have since gone.
 */
export function failedPipelines(runs: VideoRun[], selection: RunSelection): FailedPipeline[] {
  const failed: FailedPipeline[] = [];
  const seen = new Set(Object.keys(selection));
  for (const run of runs) {
    for (const p of run.pipelines) {
      if (seen.has(p.name)) continue;
      if (p.status === 'failed') {
        seen.add(p.name);
        failed.push({ run: run.job_id, pipeline: p.name, error: p.error ?? null });
      } else if (p.status === 'completed' && !p.files.some((f) => f.kind === 'result')) {
        seen.add(p.name);
        failed.push({ run: run.job_id, pipeline: p.name, error: MISSING_RESULTS, missing: true });
      }
    }
  }
  return failed;
}

const DAY = new Intl.DateTimeFormat(undefined, { day: 'numeric', month: 'short' });

/** "Peekaboo study (2026-10-12)", else "run 12 Oct". */
export function runRef(run: VideoRun): RunRef {
  const when = run.completed_at ?? run.created_at;
  const date = when ? new Date(when) : null;
  const label = run.run_name ?? (date && !Number.isNaN(date.getTime()) ? `run ${DAY.format(date)}` : `run ${run.job_id.slice(0, 8)}`);
  return { id: run.job_id, label, date: when ?? null };
}

/** One pipeline's results from one run, as a lane; `compare` adds it below the chosen one. */
export interface RunPick {
  pipeline: string;
  jobId: string;
  compare?: boolean;
}

export function picksOf(selection: RunSelection, compare: RunPick[] = []): RunPick[] {
  const chosen = Object.entries(selection).map(([pipeline, jobId]) => ({ pipeline, jobId }));
  const extra = compare
    .filter((c) => selection[c.pipeline] !== c.jobId)
    .map((c) => ({ pipeline: c.pipeline, jobId: c.jobId, compare: true }));
  return [...chosen, ...extra];
}

/** A run a lane's menu offers for one pipeline. */
export interface RunOption extends RunRef {
  status: string;
  error: string | null;
  /** Completed with a result file: a lane can show it. */
  usable: boolean;
}

/** The runs on this video with `pipeline`, newest first. */
export function runOptions(runs: VideoRun[], pipeline: string): RunOption[] {
  return runs.flatMap((run) => {
    const p = run.pipelines.find((x) => x.name === pipeline);
    if (!p) return [];
    return [{ ...runRef(run), status: p.status, error: p.error ?? null, usable: completed(run, pipeline) }];
  });
}

/** The `run=` address parameter: the one job every chosen pipeline shows, else none. */
export function sharedRun(selection: RunSelection): string | null {
  const jobs = new Set(Object.values(selection));
  return jobs.size === 1 ? [...jobs][0] : null;
}
