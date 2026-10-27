/** Runs as the viewer shows them (spec 029): a batch, or one video's job. */
import type { ActivityRun, RunSummary } from '@/types/videos';
import { formatDuration, formatWhen } from '@/lib/when';

export const videosWord = (n: number) => `${n} ${n === 1 ? 'video' : 'videos'}`;

/** A run of one video without a batch: its id is the job's. */
export const isSingle = (run: Pick<RunSummary, 'id' | 'job_ids'>) => run.job_ids.length === 1 && run.job_ids[0] === run.id;

export const finishedCount = (run: Pick<ActivityRun, 'done' | 'failed' | 'cancelled'>) => run.done + run.failed + run.cancelled;

type Counts = Pick<RunSummary, 'active' | 'total' | 'done' | 'failed' | 'running' | 'waiting' | 'cancelled'>;

/** "running 31/48", "waiting 0/48", "✓ 12 Oct", "stopped". */
export function runStatus(run: RunSummary): string {
  if (run.active) return `${run.running > 0 ? 'running' : 'waiting'} ${finishedCount(run)}/${run.total}`;
  if (run.cancelled > 0 && run.done === 0) return 'stopped';
  const when = formatWhen(run.finished_at);
  return run.failed > 0 ? `finished ${when}, ${run.failed} failed` : `finished ${when}`;
}

export type RunState = 'running' | 'waiting' | 'done' | 'failed' | 'stopped';

export function runStateOf(run: Counts): RunState {
  if (run.active) return run.running > 0 ? 'running' : 'waiting';
  if (run.failed > 0) return 'failed';
  if (run.cancelled > 0) return 'stopped';
  return 'done';
}

/** The run's outcome in words (viewer design "Run row"): "3 of 8 done · 1 running · 4 waiting", "46 done, 2 failed". */
export function runOutcome(run: Counts): string {
  if (run.active) {
    const parts = [`${run.done} of ${run.total} done`];
    if (run.running) parts.push(`${run.running} running`);
    if (run.waiting) parts.push(`${run.waiting} waiting`);
    if (run.failed) parts.push(`${run.failed} failed`);
    return parts.join(' · ');
  }
  const parts = [`${run.done} done`];
  if (run.failed) parts.push(`${run.failed} failed`);
  if (run.cancelled) parts.push(`${run.cancelled} stopped`);
  return parts.join(', ');
}

/** How long it took, or "so far" while it runs; '' without a start. */
export function runDuration(run: Pick<RunSummary, 'active' | 'started_at' | 'finished_at'>, now: Date = new Date()): string {
  if (!run.started_at) return '';
  const end = run.active ? now : run.finished_at ? new Date(run.finished_at) : null;
  if (!end) return '';
  const took = formatDuration((end.getTime() - new Date(run.started_at).getTime()) / 1000);
  return run.active ? `${took} so far` : took;
}

export const isVlmRun = (run: Pick<RunSummary, 'pipelines'>) => run.pipelines.includes('vlm_annotation');
