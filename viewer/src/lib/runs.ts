/** Runs as the viewer shows them (spec 029): a batch, or one video's job. */
import type { ActivityRun, RunSummary } from '@/types/videos';

export const videosWord = (n: number) => `${n} ${n === 1 ? 'video' : 'videos'}`;

/** A run of one video without a batch: its id is the job's. */
export const isSingle = (run: Pick<RunSummary, 'id' | 'job_ids'>) => run.job_ids.length === 1 && run.job_ids[0] === run.id;

export const finishedCount = (run: Pick<ActivityRun, 'done' | 'failed' | 'cancelled'>) => run.done + run.failed + run.cancelled;

/** "running 31/48", "waiting 0/48", "✓ 12 Oct", "stopped". */
export function runStatus(run: RunSummary): string {
  if (run.active) return `${run.running > 0 ? 'running' : 'waiting'} ${finishedCount(run)}/${run.total}`;
  if (run.cancelled > 0 && run.done === 0) return 'stopped';
  const when = run.finished_at ? new Date(run.finished_at).toLocaleDateString(undefined, { day: 'numeric', month: 'short' }) : '';
  return run.failed > 0 ? `finished ${when}, ${run.failed} failed` : `finished ${when}`;
}

export const isVlmRun = (run: Pick<RunSummary, 'pipelines'>) => run.pipelines.includes('vlm_annotation');
