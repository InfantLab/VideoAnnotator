/** The Data tab's filtering and playhead lookup (spec 028 US3), kept pure for tests. */
import { isAnswerField, tableKindOf } from '@/lib/laneKinds';
import type { Row } from '@/lib/table/rows';

/**
 * What the summary keeps (viewer design D6, UAT #46): the events a researcher
 * would name. Per-frame measures are the lanes' graphs, and per-class scores
 * (every scene, every emotion) are in "All rows" and the downloads.
 */
const SUMMARY_MEASURES = new Set(['speech', 'speaker_turn', 'scene', 'emotion', 'vlm_label', 'note']);

export const inSummary = (row: Pick<Row, 'measure'>) => SUMMARY_MEASURES.has(row.measure) || isAnswerField(row.measure);

const sameEvent = (row: Row) =>
  [row.pipeline, row.measure, row.track, row.start, row.end, row.label, row.value].join('\u0000');

/** Indexes into `rows` (sorted by start) that pass the lane filter and the search; the summary drops repeats of one event. */
export function filterRows(rows: Row[], kind: string, search: string, summary = false): number[] {
  const needle = search.trim().toLowerCase();
  const out: number[] = [];
  const seen = new Set<string>();
  rows.forEach((row, i) => {
    if (summary) {
      if (!inSummary(row)) return;
      const key = sameEvent(row);
      if (seen.has(key)) return;
      seen.add(key);
    }
    if (kind !== 'all' && tableKindOf(row).kind !== kind) return;
    if (needle) {
      const haystack = `${row.label ?? ''} ${row.value ?? ''} ${row.track}`.toLowerCase();
      if (!haystack.includes(needle)) return;
    }
    out.push(i);
  });
  return out;
}

/** Position in `shown` of the last row starting at or before `t`; -1 before the first. */
export function shownAtTime(rows: Row[], shown: number[], t: number): number {
  let lo = 0;
  let hi = shown.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (rows[shown[mid]].start <= t + 1e-6) lo = mid + 1;
    else hi = mid;
  }
  return lo - 1;
}

/** Every column of a row, for its hover. */
export const rowDetails = (row: Row, runLabel?: string): string =>
  Object.entries({ ...row, run: runLabel ?? row.run })
    .filter(([, v]) => v !== null && v !== '')
    .map(([k, v]) => `${k}: ${v}`)
    .join('\n');
