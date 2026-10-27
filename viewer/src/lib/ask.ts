/** Helpers for the Ask AI window (spec 027). */
import type { AskAnswer, AskEntry, AskSampling } from '@/types/ask';
import { answerKey, answerText } from '@/lib/answerColours';
import { formatClock } from './lanes';
import { formatWhen } from './when';

export const MAX_SHOWN = 4;
const SAME_TIME = 1e-3;

/** What a history entry is called: its name, else its question's first line. */
export function entryTitle(entry: Pick<AskEntry, 'name' | 'prompt'>): string {
  if (entry.name) return entry.name;
  const line = entry.prompt.trim().split('\n')[0] || '(empty question)';
  return line.length > 60 ? `${line.slice(0, 57)}…` : line;
}

/** How a try or run is told apart from another of the same question: its name, else "Run 14:10 · gemma4:12b · Fields" (UAT #24). */
export function entryLabel(entry: Pick<AskEntry, 'name' | 'kind' | 'created_at' | 'model' | 'answer_shape'>): string {
  if (entry.name) return entry.name;
  const kind = entry.kind === 'run' ? 'Run' : 'Try';
  return [`${kind} ${formatWhen(entry.created_at)}`, entry.model, entry.answer_shape?.length ? 'Fields' : null].filter(Boolean).join(' · ');
}

/** "every 5 s · 3 frames 0.5 s apart · 0:10 to 0:20" */
export function samplingSummary(s: Partial<AskSampling>): string {
  const parts: string[] = [];
  if (s.moments && s.moments.length) {
    parts.push(`at ${s.moments.length} moment${s.moments.length === 1 ? '' : 's'}`);
  } else {
    parts.push(`every ${s.interval ?? 5} s`);
    if ((s.start ?? 0) > 0 || s.end != null) {
      parts.push(`${formatClock(s.start ?? 0)} to ${s.end != null ? formatClock(s.end) : 'end'}`);
    }
  }
  const frames = s.frames ?? 1;
  if (frames > 1) parts.push(`${frames} frames ${s.spacing ?? s.interval ?? 5} s apart`);
  return parts.join(' · ');
}

/** The sampling as the server keeps it, with every field set. */
export function fullSampling(s: Partial<AskSampling> | null | undefined): AskSampling {
  return {
    interval: s?.interval ?? 5,
    start: s?.start ?? 0,
    end: s?.end ?? null,
    moments: s?.moments ?? null,
    frames: s?.frames ?? 1,
    spacing: s?.spacing ?? null,
  };
}

export interface AlignedRow {
  t: number;
  /** Per column, the answers at this time: none, one, or a repeat. */
  cells: AskAnswer[][];
}

/**
 * One row per sample time across the columns. A time one column has and
 * another hasn't is an empty cell there, never a guess (Principle VI).
 */
export function alignRows(columns: AskAnswer[][]): AlignedRow[] {
  const times: number[] = [];
  for (const answers of columns) {
    for (const a of answers) {
      if (!times.some((t) => Math.abs(t - a.t) < SAME_TIME)) times.push(a.t);
    }
  }
  times.sort((a, b) => a - b);
  return times.map((t) => ({
    t,
    cells: columns.map((answers) => answers.filter((a) => Math.abs(a.t - t) < SAME_TIME)),
  }));
}

const repeatKey = (a: AskAnswer) => (a.status === 'answered' ? `a:${a.label ?? ''}` : a.status);

/** A moment tried more than once that didn't get the same answer each time. */
export function repeatsDiffer(answers: AskAnswer[]): boolean {
  return answers.length > 1 && new Set(answers.map(repeatKey)).size > 1;
}

export interface Tally {
  answers: Array<[string, number]>;
  errors: number;
  skipped: number;
  /** Shaped answers that didn't fit (spec 033), never counted as a value. */
  misfits: number;
}

/** Distinct answers with counts, most common first; errors, skipped and misfits apart. */
export function tally(answers: AskAnswer[]): Tally {
  const counts = new Map<string, number>();
  const shown = new Map<string, string>();
  let errors = 0;
  let skipped = 0;
  let misfits = 0;
  for (const a of answers) {
    if (a.status === 'error') errors++;
    else if (a.status === 'skipped') skipped++;
    else if (a.fit === false) misfits++;
    else {
      // Grouped loosely ("Happy." and "happy" are one answer), shown as first written (UAT #24).
      const key = answerKey(a);
      if (!shown.has(key)) shown.set(key, answerText(a));
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
  }
  const sorted = [...counts.entries()]
    .map(([key, n]) => [shown.get(key) ?? key, n] as [string, number])
    .sort((x, y) => y[1] - x[1] || x[0].localeCompare(y[0]));
  return { answers: sorted, errors, skipped, misfits };
}

export const ALLOWED_ANSWERS_HINT =
  "Say in the question which answers are allowed, e.g. 'Answer with one of: touching, not touching.'";

export const SAMPLING_DOCS_URL = 'https://github.com/InfantLab/VideoAnnotator/blob/master/docs/usage/ask_ai.md#sampling';

/** More distinct answers than this suggests the question doesn't say which are allowed. */
export const MANY_ANSWERS = 10;

/** The sample time closest to `t`, or null when there are none. */
export function nearest(times: number[], t: number): number | null {
  let best: number | null = null;
  for (const time of times) if (best === null || Math.abs(time - t) < Math.abs(best - t)) best = time;
  return best;
}

/** "2 of 3 frames" when a burst reached past either end of the video. */
export function framesSentNote(answer: AskAnswer, requested: number): string | null {
  const sent = answer.frame_numbers?.length ?? 0;
  return requested > 1 && answer.status !== 'skipped' && sent < requested ? `${sent} of ${requested} frames` : null;
}
