/** Shared by the lane timeline (spec 027 R15; spec 028 extends it). */
import { categoryHues } from '@/lib/categoryColors';
import { kindOf, laneKind, overlayKey, pipelineShortName, type LaneForm, type LaneKind } from '@/lib/laneKinds';
import type { Row } from '@/lib/table/rows';

/** Where `t` sits along a lane, as a percentage of its width. */
export const lanePercent = (t: number, duration: number): number =>
  duration > 0 ? Math.min(100, Math.max(0, (t / duration) * 100)) : 0;

/** "m:ss", or "h:mm:ss" from an hour. */
export function formatClock(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = String(s % 60).padStart(2, '0');
  return h > 0 ? `${h}:${String(m).padStart(2, '0')}:${ss}` : `${m}:${ss}`;
}

const STEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];

/** Tick times for an axis: about `target` of them, at a round step. */
export function axisTicks(duration: number, target = 8): number[] {
  if (!(duration > 0)) return [0];
  const step = STEPS.find((s) => duration / s <= target) ?? STEPS[STEPS.length - 1];
  const ticks: number[] = [];
  for (let t = 0; t <= duration + 1e-9; t += step) ticks.push(t);
  return ticks;
}

/** The time a click at `clientX` on `element` points to. */
export function timeAtClick(clientX: number, element: Element, duration: number): number {
  const rect = element.getBoundingClientRect();
  if (rect.width <= 0) return 0;
  return Math.min(duration, Math.max(0, ((clientX - rect.left) / rect.width) * duration));
}

export type MarkerStatus = 'answered' | 'error' | 'skipped';

export interface LaneMarker {
  t: number;
  status: MarkerStatus;
  label?: string | null;
  /** A shaped answer that didn't fit its shape (spec 033): never drawn as a value. */
  fit?: boolean | null;
}

// ---------------------------------------------------------------------------
// The Video page's lanes (spec 028 R6, R7): a view range, lanes built from
// table rows, and what a canvas needs to draw thousands of events quickly.
// ---------------------------------------------------------------------------


/** The stretch of the video the timeline shows, in seconds. */
export interface ViewRange {
  start: number;
  end: number;
}

export const viewPercent = (t: number, view: ViewRange): number => {
  const span = view.end - view.start;
  return span > 0 ? ((t - view.start) / span) * 100 : 0;
};

export function timeAtClickInView(clientX: number, element: Element, view: ViewRange): number {
  const rect = element.getBoundingClientRect();
  if (rect.width <= 0) return view.start;
  const t = view.start + ((clientX - rect.left) / rect.width) * (view.end - view.start);
  return Math.min(view.end, Math.max(view.start, t));
}

const FINE_STEPS = [0.1, 0.2, 0.5, ...STEPS];

/** Tick times inside `view`, at a round step; the step comes back too. */
export function viewTicks(view: ViewRange, target = 8): { ticks: number[]; step: number } {
  const span = view.end - view.start;
  if (!(span > 0)) return { ticks: [view.start], step: 1 };
  const step = FINE_STEPS.find((s) => span / s <= target) ?? FINE_STEPS[FINE_STEPS.length - 1];
  const ticks: number[] = [];
  for (let i = Math.ceil(view.start / step - 1e-9); i * step <= view.end + 1e-9; i++) ticks.push(i * step);
  return { ticks, step };
}

/** "m:ss.s" with tenths, for a zoomed-in axis and the readout. */
export function formatClockTenths(seconds: number): string {
  const t = Math.max(0, seconds);
  const whole = Math.floor(t);
  return `${formatClock(whole)}.${Math.floor((t - whole) * 10 + 1e-6)}`;
}

export interface RunRef {
  /** What a row's `run` holds for this run: the job id, or '' for local files. */
  id: string;
  /** "Peekaboo study (2026-10-12)", or "run 12 Oct". */
  label: string;
  date?: string | null;
}

export interface LaneEvent {
  start: number;
  end: number | null;
  value: number | null;
  label: string | null;
  confidence: number | null;
  frame: number | null;
  /** Index into the rows the lanes were built from (SC-005), the first for a joined bar. */
  row: number;
  /** Samples joined into one presence bar. */
  samples?: number;
}

export interface LaneRow {
  key: string;
  label: string;
  description?: string;
  colour: string;
  form: LaneForm;
  events: LaneEvent[];
  /** Longest event, so a visible slice can include spans that start before the view. */
  maxDuration: number;
  /** Samples further apart than this aren't joined (Principle VI). */
  interval: number | null;
  min?: number;
  max?: number;
  unit?: string;
}

/** `missing`: the pipeline finished, but its result files are gone from the results folder. */
export type LaneStatus = 'ok' | 'failed' | 'missing' | 'empty';

export interface Lane {
  id: string;
  kind: string;
  pipeline: string | null;
  label: string;
  description: string;
  run: RunRef | null;
  form: LaneForm;
  rows: LaneRow[];
  status: LaneStatus;
  error?: string;
  overlay: string | null;
  source: string[];
}

export interface FailedPipeline {
  run: string;
  pipeline: string;
  error: string | null;
  /** Finished, but its result files are no longer there. */
  missing?: boolean;
}

/** The lane kind a pipeline's results go to, for a failed pipeline with no rows. */
const PIPELINE_KINDS: Record<string, string> = {
  speech_recognition: 'speech',
  speaker_diarization: 'speakers',
  person_tracking: 'people',
  face_analysis: 'faces',
  face_openface3_embedding: 'faces',
  scene_detection: 'scenes',
  vlm_annotation: 'ask_ai',
};

export const colourFor = (hue: number) => `hsl(${Math.round(hue * 10) / 10}, 70%, 60%)`;

/** The typical gap between samples: the median gap between distinct times. */
export function samplingInterval(times: number[]): number | null {
  const distinct = [...new Set(times)].sort((a, b) => a - b);
  const gaps: number[] = [];
  for (let i = 1; i < distinct.length; i++) gaps.push(distinct[i] - distinct[i - 1]);
  if (!gaps.length) return null;
  gaps.sort((a, b) => a - b);
  return gaps[Math.floor(gaps.length / 2)];
}

const withinInterval = (gap: number, interval: number | null) =>
  interval !== null && gap <= interval * 1.001 + 1e-6;

function rowLabel(kind: string, key: string, labels: Set<string>): string {
  if (!key) return '';
  if (kind === 'people' || kind === 'faces') {
    const n = /(\d+)$/.exec(key)?.[1];
    const name = n !== undefined ? `${kind === 'people' ? 'person' : 'face'} ${Number(n)}` : key;
    const label = [...labels].filter(Boolean).join(', ');
    return label ? `${name} (${label})` : name;
  }
  return key;
}

function presence(events: LaneEvent[], interval: number | null): LaneEvent[] {
  const joined: LaneEvent[] = [];
  for (const e of events) {
    const last = joined[joined.length - 1];
    if (last && withinInterval(e.start - (last.end ?? last.start), interval)) {
      last.end = e.start;
      last.samples = (last.samples ?? 1) + 1;
    } else {
      joined.push({ ...e, end: e.start, samples: 1 });
    }
  }
  return joined;
}

function finishRow(row: Omit<LaneRow, 'maxDuration'>): LaneRow {
  row.events.sort((a, b) => a.start - b.start);
  let maxDuration = 0;
  let min = Infinity;
  let max = -Infinity;
  for (const e of row.events) {
    if (e.end !== null && e.end - e.start > maxDuration) maxDuration = e.end - e.start;
    if (e.value !== null) {
      if (e.value < min) min = e.value;
      if (e.value > max) max = e.value;
    }
  }
  const numeric = row.form === 'numeric' && min <= max;
  return { ...row, maxDuration, ...(numeric ? { min, max } : {}) };
}

const eventOf = (r: Row, index: number): LaneEvent => ({
  start: r.start,
  end: r.end,
  value: r.value,
  label: r.label,
  confidence: r.confidence,
  frame: r.frame,
  row: index,
});

interface Group {
  kind: string;
  spec: LaneKind;
  pipeline: string;
  run: string;
  rows: { row: Row; index: number }[];
  files: Set<string>;
}

function buildRows(group: Group): LaneRow[] {
  const { spec } = group;
  const times = group.rows.map(({ row }) => row.start);
  const interval = samplingInterval(times);

  if (group.kind === 'person_area') {
    const byTime = new Map<number, { area: number; index: number }>();
    for (const { row, index } of group.rows) {
      const area = (row.width ?? 0) * (row.height ?? 0);
      const at = byTime.get(row.start);
      if (at) at.area += area;
      else byTime.set(row.start, { area, index });
    }
    const events = [...byTime].map(([start, { area, index }]) => ({
      start, end: null, value: area, label: null, confidence: null, frame: null, row: index,
    }));
    return [finishRow({ key: 'area', label: 'box area', colour: colourFor(30), form: 'numeric', events, interval, unit: spec.unit })];
  }

  const keyOf = (row: Row): string => {
    if (group.kind === 'emotion') return row.measure === 'emotion' ? row.label ?? '' : row.measure;
    if (spec.rowsBy === 'track') return row.track;
    if (spec.rowsBy === 'label') return row.label ?? '';
    return '';
  };
  const formOf = (key: string): LaneForm =>
    group.kind === 'emotion' && (key === 'valence' || key === 'arousal') ? 'numeric' : spec.form;

  const byKey = new Map<string, { events: LaneEvent[]; labels: Set<string> }>();
  for (const { row, index } of group.rows) {
    const key = keyOf(row);
    const entry = byKey.get(key) ?? { events: [], labels: new Set<string>() };
    entry.events.push(eventOf(row, index));
    if (group.kind === 'people' && row.label) entry.labels.add(row.label);
    byKey.set(key, entry);
  }
  const hues = categoryHues([...byKey.keys()]);
  const rows = [...byKey]
    .sort(([a], [b]) => a.localeCompare(b, undefined, { numeric: true }))
    .map(([key, { events, labels }]) => {
      const form = formOf(key);
      return finishRow({
        key,
        label: rowLabel(group.kind, key, labels) || spec.label,
        colour: colourFor(hues.get(key) ?? 0),
        form,
        events: form === 'presence' ? presence(events.sort((a, b) => a.start - b.start), interval) : events,
        interval,
        unit: form === 'numeric' ? spec.unit : undefined,
      });
    });

  if (group.kind === 'faces') {
    const counts = new Map<number, { count: number; index: number }>();
    for (const { row, index } of group.rows) {
      const at = counts.get(row.start);
      if (at) at.count += 1;
      else counts.set(row.start, { count: 1, index });
    }
    rows.push(
      finishRow({
        key: 'count',
        label: 'count',
        description: 'Faces found per sampled frame (frames with none found are not in the results)',
        colour: colourFor(200),
        form: 'numeric',
        events: [...counts].map(([start, { count, index }]) => ({
          start, end: null, value: count, label: null, confidence: null, frame: null, row: index,
        })),
        interval,
      }),
    );
  }
  return rows;
}

const ORDER_OTHER = 1000;
const PERSON_AREA = laneKind('person_area')!;

function laneOrder(kind: string): number {
  return kind.startsWith('other:') ? ORDER_OTHER : laneKind(kind)?.order ?? ORDER_OTHER;
}

/**
 * Lanes from table rows (data-model.md, Lane): one per kind, pipeline and run,
 * in the kind table's order; a failed pipeline with no rows gets a lane saying
 * why. `rows[i].run` names the run (a job id; '' for local files).
 */
export function buildLanes(
  rows: Row[],
  runs: RunRef[],
  failed: FailedPipeline[] = [],
  /** `${run}/${pipeline}` of compared runs: their lanes go right below the chosen run's. */
  compared: Set<string> = new Set(),
): Lane[] {
  const groups = new Map<string, Group>();
  const add = (spec: LaneKind, row: Row, index: number) => {
    const id = `${spec.kind}|${row.pipeline}|${row.run}`;
    const group = groups.get(id) ?? { kind: spec.kind, spec, pipeline: row.pipeline, run: row.run, rows: [], files: new Set<string>() };
    group.rows.push({ row, index });
    if (row.file) group.files.add(row.file);
    groups.set(id, group);
  };
  rows.forEach((row, index) => {
    const kind = kindOf(row);
    if (kind) add(kind, row, index);
    if (row.measure === 'person' && row.width !== null && row.height !== null) add(PERSON_AREA, row, index);
  });

  const runById = new Map(runs.map((r) => [r.id, r]));
  const lanes: Lane[] = [...groups.entries()].map(([id, group]) => {
    const { spec } = group;
    return {
      id,
      kind: group.kind,
      // Notes are the researcher's, not a pipeline's output.
      pipeline: group.kind === 'notes' ? null : group.pipeline,
      label: spec.label,
      description: spec.description,
      run: runById.get(group.run) ?? (group.run ? { id: group.run, label: group.run } : null),
      form: spec.form,
      rows: buildRows(group),
      status: 'ok',
      overlay: overlayKey(group.kind, group.pipeline),
      source: [...group.files],
    };
  });

  for (const f of failed) {
    if (lanes.some((l) => l.pipeline === f.pipeline && (l.run?.id ?? '') === f.run)) continue;
    const kind = PIPELINE_KINDS[f.pipeline] ?? `other:${f.pipeline}/result`;
    const spec = laneKind(kind)!;
    lanes.push({
      id: `${kind}|${f.pipeline}|${f.run}`,
      kind,
      pipeline: f.pipeline,
      label: spec.label,
      description: spec.description,
      run: runById.get(f.run) ?? (f.run ? { id: f.run, label: f.run } : null),
      form: spec.form,
      rows: [],
      status: f.missing ? 'missing' : 'failed',
      error: f.error ?? 'failed',
      overlay: null,
      source: [],
    });
  }

  const pipelinesOfKind = new Map<string, Set<string>>();
  for (const lane of lanes) {
    const set = pipelinesOfKind.get(lane.kind) ?? new Set<string>();
    if (lane.pipeline) set.add(lane.pipeline);
    pipelinesOfKind.set(lane.kind, set);
  }
  for (const lane of lanes) {
    if (lane.pipeline && (pipelinesOfKind.get(lane.kind)?.size ?? 0) > 1) {
      lane.label = `${lane.label} · ${pipelineShortName(lane.pipeline)}`;
    }
  }

  return lanes.sort(
    (a, b) =>
      laneOrder(a.kind) - laneOrder(b.kind) ||
      a.kind.localeCompare(b.kind) ||
      (a.pipeline ?? '').localeCompare(b.pipeline ?? '') ||
      Number(compared.has(`${a.run?.id ?? ''}/${a.pipeline}`)) - Number(compared.has(`${b.run?.id ?? ''}/${b.pipeline}`)) ||
      (a.run?.label ?? '').localeCompare(b.run?.label ?? ''),
  );
}

/** First index whose start is at least `t` (events sorted by start). */
function lowerBound(events: LaneEvent[], t: number): number {
  let lo = 0;
  let hi = events.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (events[mid].start < t) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** First index whose start is after `t`. */
function upperBound(events: LaneEvent[], t: number): number {
  let lo = 0;
  let hi = events.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (events[mid].start <= t) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** `[from, to)`: the events that may show in `view`, by two binary searches. */
export function visibleSlice(events: LaneEvent[], view: ViewRange, maxDuration = 0): [number, number] {
  return [lowerBound(events, view.start - maxDuration), upperBound(events, view.end)];
}

/** Events starting in each of `columns` equal parts of `view`. */
export function densityBins(events: LaneEvent[], view: ViewRange, columns: number, maxDuration = 0): Uint32Array {
  const bins = new Uint32Array(Math.max(0, Math.floor(columns)));
  const span = view.end - view.start;
  if (!bins.length || !(span > 0)) return bins;
  const [from, to] = visibleSlice(events, view, maxDuration);
  for (let i = from; i < to; i++) {
    const column = Math.floor(((events[i].start - view.start) / span) * bins.length);
    bins[Math.min(bins.length - 1, Math.max(0, column))] += 1;
  }
  return bins;
}

/** The next (or previous) event start after (before) `t` in any of `rows`. */
export function adjacentEvent(rows: LaneRow[], t: number, direction: 1 | -1): number | null {
  const EPS = 1e-3;
  let best: number | null = null;
  for (const row of rows) {
    const { events } = row;
    if (direction > 0) {
      const i = upperBound(events, t + EPS);
      if (i < events.length && (best === null || events[i].start < best)) best = events[i].start;
    } else {
      const i = lowerBound(events, t - EPS) - 1;
      if (i >= 0 && (best === null || events[i].start > best)) best = events[i].start;
    }
  }
  return best;
}

export const ROW_HEIGHT = 18;
export const NUMERIC_ROW_HEIGHT = 32;

export const rowHeight = (row: Pick<LaneRow, 'form'>): number => (row.form === 'numeric' ? NUMERIC_ROW_HEIGHT : ROW_HEIGHT);

/** More events in view than pixel columns: drawn as density, and labelled so. */
export function showsDensity(row: LaneRow, view: ViewRange, width: number): boolean {
  if (row.form === 'numeric' || width <= 0) return false;
  const [from, to] = visibleSlice(row.events, view, row.maxDuration);
  return to - from > width;
}

/**
 * The event under time `t` in `row`: a span or presence bar containing it, or
 * the nearest moment or sample within `tolerance` seconds.
 */
export function eventAt(row: LaneRow, t: number, tolerance: number): LaneEvent | null {
  const [from, to] = visibleSlice(row.events, { start: t - tolerance, end: t + tolerance }, row.maxDuration);
  let best: LaneEvent | null = null;
  let bestDistance = Infinity;
  for (let i = from; i < to; i++) {
    const e = row.events[i];
    const end = e.end ?? e.start;
    const distance = t < e.start ? e.start - t : t > end ? t - end : 0;
    if (distance <= tolerance && distance < bestDistance) {
      best = e;
      bestDistance = distance;
    }
  }
  return best;
}

export const formatValue = (v: number): string =>
  Math.abs(v) >= 100 || Number.isInteger(v) ? String(Math.round(v)) : v.toFixed(2);

export const MIN_VIEW_SEC = 1;

const clampView = (start: number, span: number, duration: number): ViewRange => {
  const length = Math.min(Math.max(span, Math.min(MIN_VIEW_SEC, duration)), duration);
  const from = Math.min(Math.max(0, start), Math.max(0, duration - length));
  return { start: from, end: from + length };
};

export const fullView = (duration: number): ViewRange => ({ start: 0, end: Math.max(duration, 0) });

/** Zoom by `factor` (< 1 is in) keeping `anchor` where it is; never under a second. */
export function zoomView(view: ViewRange, factor: number, anchor: number, duration: number): ViewRange {
  const span = view.end - view.start;
  const next = Math.min(duration, Math.max(Math.min(MIN_VIEW_SEC, duration), span * factor));
  const ratio = span > 0 ? (anchor - view.start) / span : 0.5;
  return clampView(anchor - ratio * next, next, duration);
}

export const panView = (view: ViewRange, seconds: number, duration: number): ViewRange =>
  clampView(view.start + seconds, view.end - view.start, duration);

/** While playing, the view pages forward when the playhead leaves it (US1.3). */
export function followView(view: ViewRange, t: number, duration: number): ViewRange {
  if (t >= view.start && t <= view.end) return view;
  return clampView(t, view.end - view.start, duration);
}

export const centreView = (view: ViewRange, t: number, duration: number): ViewRange =>
  clampView(t - (view.end - view.start) / 2, view.end - view.start, duration);

export const LANE_HEADER_WIDTH = 200;

/** What a lane shows and what made it, for its label's tooltip (SC-002). */
export function laneTooltip(lane: Lane): string {
  const made = lane.pipeline ? ` Made by ${lane.pipeline}` : '';
  const run = lane.run ? `, ${lane.run.label}` : '';
  return `${lane.label}: ${lane.description}.${made}${run}.`;
}

export function rowTooltip(lane: Lane, row: LaneRow): string {
  const scale =
    row.form === 'numeric' && row.min !== undefined && row.max !== undefined
      ? ` Scale ${formatValue(row.min)} to ${formatValue(row.max)}${row.unit ? ` ${row.unit}` : ''}.`
      : '';
  const what = row.description ?? lane.description;
  return `${row.label} (${lane.label}): ${what}.${scale}${lane.pipeline ? ` From ${lane.pipeline}.` : ''}`;
}

/** One zoom step halves (or doubles) the view. */
export const ZOOM_STEP = 0.5;

/**
 * The video's frame rate as the results record it (frame number over time);
 * null when no result gives both. Not the parsers' default of 30.
 */
export function frameRateFromRows(rows: Pick<Row, 'frame' | 'start'>[]): number | null {
  const rates: number[] = [];
  for (const r of rows) {
    if (r.frame !== null && r.frame > 0 && r.start > 0) rates.push(r.frame / r.start);
    if (rates.length >= 501) break;
  }
  if (!rates.length) return null;
  rates.sort((a, b) => a - b);
  return Math.round(rates[Math.floor(rates.length / 2)] * 1000) / 1000;
}

/** A collapsed lane is its label row only (spec 028 US2.1). */
export const laneHeight = (lane: Pick<Lane, 'rows'>, collapsed = false): number =>
  collapsed ? ROW_HEIGHT : Math.max(ROW_HEIGHT, lane.rows.reduce((sum, row) => sum + rowHeight(row), 0));
