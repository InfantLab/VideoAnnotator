/**
 * A proxy for SC-004 (smooth with an hour and 50,000 events): the work a pan
 * step does, on a synthetic hour (tests/fixtures/video_page/long_run). The
 * real check is the quickstart walk-through in a browser.
 */
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import path from 'node:path';
import { beforeAll, describe, expect, it } from 'vitest';

import { buildLanes, densityBins, panView, visibleSlice, type ViewRange } from '@/lib/lanes';
import { rowsFromFiles, type Row } from '@/lib/table/rows';

const ROOT = path.resolve(__dirname, '../../../..');
const LONG_RUN = path.join(ROOT, 'tests/fixtures/video_page/long_run');
const DURATION = 3600;

let rows: Row[] = [];

beforeAll(() => {
  // Too big to commit: written on first use (README there).
  if (!existsSync(path.join(LONG_RUN, 'long_run_person_tracking.json'))) {
    execFileSync('python3', [path.join(ROOT, 'scripts/make_long_run_fixture.py'), LONG_RUN]);
  }
  const files = readdirSync(LONG_RUN)
    .filter((name) => name.startsWith('long_run_'))
    .map((name) => ({ name, text: readFileSync(path.join(LONG_RUN, name), 'utf8') }));
  rows = rowsFromFiles(files, 'long_run', 'job');
}, 60_000);

describe('an hour with 50,000 events', () => {
  it('has the events the fixture promises', () => {
    expect(rows.length).toBeGreaterThan(45_000);
  });

  it('builds its lanes in under half a second', () => {
    const began = performance.now();
    const lanes = buildLanes(rows, [{ id: 'job', label: 'run' }]);
    expect(performance.now() - began).toBeLessThan(500);
    expect(lanes.map((l) => l.kind)).toEqual(['speech', 'speakers', 'people', 'person_area']);
  });

  it('pans 200 steps, slicing and binning every row, in under a second', () => {
    const lanes = buildLanes(rows, [{ id: 'job', label: 'run' }]);
    const rowsOfLanes = lanes.flatMap((l) => l.rows);
    let view: ViewRange = { start: 0, end: 600 };
    const began = performance.now();
    for (let step = 0; step < 200; step++) {
      view = panView(view, 15, DURATION);
      for (const row of rowsOfLanes) {
        visibleSlice(row.events, view, row.maxDuration);
        densityBins(row.events, view, 1200, row.maxDuration);
      }
    }
    expect(performance.now() - began).toBeLessThan(1000);
  });
});
