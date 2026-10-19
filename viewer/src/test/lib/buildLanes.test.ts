import { readFileSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

import {
  adjacentEvent,
  buildLanes,
  densityBins,
  samplingInterval,
  visibleSlice,
  type LaneEvent,
} from '@/lib/lanes';
import { rowsFromFiles, type Row } from '@/lib/table/rows';

const FIXTURES = path.resolve(__dirname, '../../../../tests/fixtures/viewer_contract');
const demoRows = () =>
  rowsFromFiles(
    readdirSync(FIXTURES)
      .filter((name) => statSync(path.join(FIXTURES, name)).isFile())
      .map((name) => ({ name, text: readFileSync(path.join(FIXTURES, name), 'utf8') })),
    'demo_clip',
    'job-1',
  );
const RUNS = [{ id: 'job-1', label: 'run 12 Oct' }];

const row = (over: Partial<Row>): Row => ({
  run: 'job-1',
  video: 'v',
  pipeline: 'person_tracking',
  event: 1,
  track: '',
  start: 0,
  end: null,
  frame: null,
  measure: 'person',
  label: null,
  value: null,
  confidence: null,
  x: null,
  y: null,
  width: null,
  height: null,
  model: 'm',
  videoannotator_version: '1',
  file: 'f',
  author: '',
  ...over,
});

describe('buildLanes', () => {
  it('makes one lane per kind present, and none for kinds absent', () => {
    const lanes = buildLanes(demoRows(), RUNS);
    const kinds = new Set(lanes.map((l) => l.kind));
    expect(kinds).toEqual(
      new Set(['speech', 'speakers', 'people', 'faces', 'emotion', 'scenes', 'ask_ai', 'action_units', 'head_pose', 'gaze', 'person_area']),
    );
    expect(lanes.every((l) => l.status === 'ok' && l.run?.label === 'run 12 Oct')).toBe(true);
    expect(lanes[0].kind).toBe('speech');
  });

  it('names the pipeline when two pipelines make one kind', () => {
    const faces = buildLanes(demoRows(), RUNS).filter((l) => l.kind === 'faces');
    expect(faces.map((l) => l.label).sort()).toEqual(['Faces · DeepFace', 'Faces · OpenFace 3']);
    expect(faces.map((l) => l.overlay).sort()).toEqual(['faces', 'faces.openface3']);
  });

  it('gives each speaker a row with its own colour', () => {
    const rows = [
      row({ pipeline: 'speaker_diarization', measure: 'speaker_turn', track: 'SPEAKER_01', label: 'SPEAKER_01', start: 2, end: 3 }),
      row({ pipeline: 'speaker_diarization', measure: 'speaker_turn', track: 'SPEAKER_00', label: 'SPEAKER_00', start: 0, end: 1 }),
      row({ pipeline: 'speaker_diarization', measure: 'speaker_turn', track: 'SPEAKER_00', label: 'SPEAKER_00', start: 4, end: 5 }),
    ];
    const [speakers] = buildLanes(rows, RUNS);
    expect(speakers.rows.map((r) => r.label)).toEqual(['SPEAKER_00', 'SPEAKER_01']);
    expect(speakers.rows[0].events.map((e) => e.row)).toEqual([1, 2]);
    expect(speakers.rows[0].colour).not.toBe(speakers.rows[1].colour);
  });

  it('gives a failed pipeline a lane saying why', () => {
    const lanes = buildLanes([], RUNS, [{ run: 'job-1', pipeline: 'speech_recognition', error: 'No audio' }]);
    expect(lanes).toHaveLength(1);
    expect(lanes[0]).toMatchObject({ kind: 'speech', status: 'failed', error: 'No audio' });
  });

  it('never bridges presence across a gap longer than the sampling interval', () => {
    const times = [0, 1, 2, 3, 7, 8];
    const rows = times.map((t) => row({ track: 'person_1', start: t, width: 1, height: 1 }));
    const people = buildLanes(rows, RUNS).find((l) => l.kind === 'people')!;
    const bars = people.rows[0].events.map((e) => [e.start, e.end]);
    expect(bars).toEqual([
      [0, 3],
      [7, 8],
    ]);
  });

  it('labels a person row by its number and the label the pipeline gave', () => {
    const people = buildLanes([row({ track: 'person_demo_clip_002', label: 'infant' })], RUNS).find((l) => l.kind === 'people')!;
    expect(people.rows[0].label).toBe('person 2 (infant)');
  });

  it('gives a measure it doesn\'t know a generic lane, never drops it', () => {
    const lanes = buildLanes([row({ pipeline: 'blink', measure: 'blink_rate', value: 3 })], RUNS);
    expect(lanes.map((l) => l.kind)).toEqual(['other:blink/blink_rate']);
    expect(lanes[0].form).toBe('numeric');
  });

  it('counts faces per sampled frame', () => {
    const rows = [0, 0, 1].map((t) => row({ pipeline: 'face_analysis', measure: 'face', start: t }));
    const faces = buildLanes(rows, RUNS).find((l) => l.kind === 'faces')!;
    const count = faces.rows.find((r) => r.key === 'count')!;
    expect(count.events.map((e) => [e.start, e.value])).toEqual([
      [0, 2],
      [1, 1],
    ]);
    expect([count.min, count.max]).toEqual([1, 2]);
  });
});

const events = (n: number, duration: number): LaneEvent[] =>
  Array.from({ length: n }, (_, i) => ({
    start: (i / n) * duration,
    end: null,
    value: null,
    label: null,
    confidence: null,
    frame: null,
    row: i,
  }));

describe('the visible slice and density', () => {
  it('finds the events in view by binary search', () => {
    const e = events(100, 100);
    expect(visibleSlice(e, { start: 10, end: 20 })).toEqual([10, 21]);
    expect(visibleSlice(e, { start: 10, end: 20 }, 5)).toEqual([5, 21]);
  });

  it('bins 50,000 events in well under a frame', () => {
    const e = events(50_000, 3600);
    const began = performance.now();
    const [from, to] = visibleSlice(e, { start: 600, end: 1200 });
    const bins = densityBins(e, { start: 0, end: 3600 }, 1200);
    expect(performance.now() - began).toBeLessThan(20);
    expect(to - from).toBeGreaterThan(8000);
    expect(bins.reduce((a, b) => a + b, 0)).toBe(50_000);
  });

  it('steps to the next and previous event', () => {
    const rows = [{ events: events(10, 10) }, { events: [{ ...events(1, 1)[0], start: 2.5 }] }];
    expect(adjacentEvent(rows as never, 2, 1)).toBe(2.5);
    expect(adjacentEvent(rows as never, 2.5, 1)).toBe(3);
    expect(adjacentEvent(rows as never, 2.5, -1)).toBe(2);
    expect(adjacentEvent(rows as never, 0, -1)).toBeNull();
  });

  it('takes the median gap as the sampling interval', () => {
    expect(samplingInterval([0, 1, 2, 3, 10])).toBe(1);
    expect(samplingInterval([5])).toBeNull();
  });
});

describe('zoom, pan and follow', () => {
  const D = 3600;
  it('zooms around an anchor, never under a second', async () => {
    const { zoomView } = await import('@/lib/lanes');
    expect(zoomView({ start: 0, end: 100 }, 0.5, 50, D)).toEqual({ start: 25, end: 75 });
    let view = { start: 0, end: 4 };
    for (let i = 0; i < 10; i++) view = zoomView(view, 0.5, 2, D);
    expect(view.end - view.start).toBe(1);
    expect(zoomView({ start: 0, end: 3000 }, 2, 0, D)).toEqual({ start: 0, end: D });
  });

  it('pans within the video', async () => {
    const { panView } = await import('@/lib/lanes');
    expect(panView({ start: 10, end: 20 }, -50, D)).toEqual({ start: 0, end: 10 });
    expect(panView({ start: 10, end: 20 }, 5000, D)).toEqual({ start: D - 10, end: D });
  });

  it('pages forward when the playhead leaves the view', async () => {
    const { followView } = await import('@/lib/lanes');
    const view = { start: 0, end: 10 };
    expect(followView(view, 5, D)).toBe(view);
    expect(followView(view, 10.1, D)).toEqual({ start: 10.1, end: 20.1 });
  });
});
