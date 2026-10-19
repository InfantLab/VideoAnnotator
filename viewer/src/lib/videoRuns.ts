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

/** Pipelines that never completed on this video, with the newest run's reason. */
export function failedPipelines(runs: VideoRun[], selection: RunSelection): FailedPipeline[] {
  const failed: FailedPipeline[] = [];
  const seen = new Set(Object.keys(selection));
  for (const run of runs) {
    for (const p of run.pipelines) {
      if (seen.has(p.name) || p.status !== 'failed') continue;
      seen.add(p.name);
      failed.push({ run: run.job_id, pipeline: p.name, error: p.error ?? null });
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
