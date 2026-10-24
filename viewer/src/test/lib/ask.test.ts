import { describe, expect, it } from 'vitest';
import { alignRows, entryTitle, framesSentNote, nearest, repeatsDiffer, samplingSummary, tally } from '@/lib/ask';
import { parseMoments } from '@/lib/sampling';
import type { AskAnswer } from '@/types/ask';

const a = (t: number, label: string | null, status: AskAnswer['status'] = 'answered', frames = 1): AskAnswer => ({
  t,
  label,
  status,
  frame_numbers: Array.from({ length: frames }, (_, i) => i),
  frame_times: [],
});

describe('Ask AI helpers', () => {
  it('names an entry by its name or the first line of its question', () => {
    expect(entryTitle({ name: 'touch v2', prompt: 'x' })).toBe('touch v2');
    expect(entryTitle({ name: null, prompt: 'Is the adult touching?\nAnswer touching or not' })).toBe('Is the adult touching?');
  });

  it('summarises sampling', () => {
    expect(samplingSummary({ interval: 5, frames: 1 })).toBe('every 5 s');
    expect(samplingSummary({ interval: 2, start: 10, end: 20, frames: 3, spacing: 0.5 })).toBe(
      'every 2 s · 0:10 to 0:20 · 3 frames 0.5 s apart',
    );
    expect(samplingSummary({ moments: [1, 2, 3] })).toBe('at 3 moments');
  });

  it('aligns entries by sample time, leaving gaps empty', () => {
    const rows = alignRows([[a(0, 'X'), a(5, 'Y')], [a(5, 'Z'), a(10, 'W')]]);
    expect(rows.map((r) => r.t)).toEqual([0, 5, 10]);
    expect(rows[0].cells[1]).toEqual([]);
    expect(rows[1].cells.map((c) => c[0].label)).toEqual(['Y', 'Z']);
  });

  it('marks repeats that differ, not ones that agree', () => {
    expect(repeatsDiffer([a(1, 'X'), a(1, 'X')])).toBe(false);
    expect(repeatsDiffer([a(1, 'X'), a(1, 'Y')])).toBe(true);
    expect(repeatsDiffer([a(1, 'X'), a(1, null, 'error')])).toBe(true);
  });

  it('counts errors and skipped apart from answers', () => {
    const t = tally([a(0, 'X'), a(1, 'X'), a(2, 'Y'), a(3, null, 'error'), a(4, null, 'skipped')]);
    expect(t).toEqual({ answers: [['X', 2], ['Y', 1]], errors: 1, skipped: 1, misfits: 0 });
  });

  it('finds the nearest sample', () => {
    expect(nearest([0, 5, 10], 6.4)).toBe(5);
    expect(nearest([], 3)).toBeNull();
  });

  it('says when a burst was clipped', () => {
    expect(framesSentNote(a(0, 'X', 'answered', 2), 3)).toBe('2 of 3 frames');
    expect(framesSentNote(a(5, 'X', 'answered', 3), 3)).toBeNull();
  });

  it('reads moments as m:ss or seconds', () => {
    expect(parseMoments('0:12, 0:47, 1:30')).toEqual([12, 47, 90]);
    expect(parseMoments('2, 5.5  9,x,-1, 1:2:03')).toEqual([2, 5.5, 9, 3723]);
  });
});
