/**
 * Where a pending job is in the server's queue, so a queued job doesn't look
 * the same as a stuck one. The server sets `queue_position` (1-based, oldest
 * pending first, the order the worker takes them) only while a job is pending.
 */
export function queueLabel(job: { status: string; queue_position?: unknown }): string | null {
  const position = job.queue_position;
  if (job.status !== 'pending' || typeof position !== 'number' || !Number.isInteger(position) || position < 1) {
    return null;
  }
  return position === 1 ? 'Next in queue' : `${ordinal(position)} in queue`;
}

function ordinal(n: number): string {
  const lastTwo = n % 100;
  if (lastTwo >= 11 && lastTwo <= 13) return `${n}th`;
  return `${n}${{ 1: 'st', 2: 'nd', 3: 'rd' }[n % 10] ?? 'th'}`;
}
