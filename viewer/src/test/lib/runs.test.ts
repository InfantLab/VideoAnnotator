import { describe, expect, it } from 'vitest';

import { runDuration, runOutcome, runStateOf } from '@/lib/runs';
import { FINISHED_RUN, run } from '@/test/fixtures/videos';

describe('a run in words (viewer design "Run row")', () => {
  it('while it runs', () => {
    expect(runStateOf(run())).toBe('running');
    expect(runOutcome(run())).toBe('31 of 48 done · 2 running · 15 waiting');
    expect(runDuration(run({ started_at: '2026-10-20T09:00:00Z' }), new Date('2026-10-20T09:06:00Z'))).toBe('6 min so far');
  });

  it('once finished', () => {
    expect(runStateOf(FINISHED_RUN)).toBe('failed');
    expect(runOutcome(FINISHED_RUN)).toBe('48 done, 1 failed');
    expect(runDuration({ active: false, started_at: '2026-10-12T09:00:00Z', finished_at: '2026-10-12T10:12:00Z' })).toBe('1 h 12 min');
  });

  it('waiting, stopped and done', () => {
    expect(runStateOf(run({ running: 0 }))).toBe('waiting');
    expect(runStateOf(run({ active: false, running: 0, waiting: 0, cancelled: 17 }))).toBe('stopped');
    expect(runStateOf(run({ active: false, running: 0, waiting: 0, done: 48 }))).toBe('done');
  });
});
