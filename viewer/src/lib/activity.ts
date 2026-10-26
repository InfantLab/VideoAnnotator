/**
 * What changed between two polls of the runs going on (spec 029 R17): a run
 * started, finished, failed or was stopped. A run that finished while the
 * viewer was closed is reported once after it opens, from `recent`.
 */
import type { Activity, ActivityRun } from '@/types/videos';

export type TransitionKind = 'started' | 'finished' | 'failed' | 'stopped';

export interface Transition {
  kind: TransitionKind;
  run: ActivityRun;
}

const ANNOUNCED_KEY = 'videoannotator_announced_runs';
const ANNOUNCED_MAX = 50;

export function loadAnnounced(): Set<string> {
  try {
    const stored = JSON.parse(localStorage.getItem(ANNOUNCED_KEY) ?? '[]');
    return new Set(Array.isArray(stored) ? stored.map(String) : []);
  } catch {
    return new Set();
  }
}

export function saveAnnounced(ids: Set<string>): void {
  try {
    localStorage.setItem(ANNOUNCED_KEY, JSON.stringify([...ids].slice(-ANNOUNCED_MAX)));
  } catch {
    // Private windows: announcements just repeat after a reload.
  }
}

function ended(run: ActivityRun): TransitionKind {
  if (run.stopped) return 'stopped';
  if (run.total > 0 && run.failed === run.total) return 'failed';
  return 'finished';
}

/**
 * Runs that finished over an hour before the viewer first looked: not news, so
 * not a message (UAT #40; a fresh browser otherwise announced every recent run).
 * Their runs still list them.
 */
export function finishedBeforeLooking(first: Activity, now: number = Date.now()): string[] {
  return first.recent
    .filter((run) => run.finished_at && now - new Date(run.finished_at).getTime() > 60 * 60 * 1000)
    .map((run) => run.id);
}

/**
 * `previous` is null on the first poll: then only finished runs not yet
 * announced are reported (so a reload doesn't repeat them, and a run that
 * finished while the tab was closed still is).
 */
export function activityTransitions(
  previous: Activity | null,
  next: Activity,
  announced: Set<string>,
): Transition[] {
  const found: Transition[] = [];
  const wasActive = new Set(previous?.active.map((r) => r.id) ?? []);
  if (previous) {
    for (const run of next.active) if (!wasActive.has(run.id)) found.push({ kind: 'started', run });
  }
  for (const run of next.recent) {
    if (announced.has(run.id)) continue;
    if (previous && !wasActive.has(run.id)) {
      // Finished before we saw it running: only worth saying on the first poll.
      continue;
    }
    found.push({ kind: ended(run), run });
  }
  return found;
}

export function progressLine(runs: ActivityRun[]): string {
  const total = runs.reduce((n, r) => n + r.total, 0);
  const finished = runs.reduce((n, r) => n + r.done + r.failed + r.cancelled, 0);
  return `${runs.length} ${runs.length === 1 ? 'run' : 'runs'} · ${finished}/${total}`;
}
