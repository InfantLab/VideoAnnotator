import { describe, expect, it } from 'vitest';
import { comparisonCsv, pairRuns, summarize, withTruth } from '@/lib/vlmCompare';
import type { ElanTierAnnotation, VLMFrameAnnotation } from '@/types/annotations';

const s = (t: number, label: string, reasoning = ''): VLMFrameAnnotation =>
  ({ timestamp_sec: t, label, reasoning } as VLMFrameAnnotation);

describe('pairRuns', () => {
  it('pairs same-time samples and marks agreement as recorded', () => {
    const pairs = pairRuns([s(0, 'TOUCH'), s(5, 'TOUCH'), s(10, 'touch')], [s(0, 'TOUCH'), s(5, 'NO_TOUCH'), s(10, 'TOUCH')]);
    expect(pairs.map((p) => p.status)).toEqual(['agree', 'disagree', 'disagree']);
    expect(pairRuns([s(10, 'touch')], [s(10, 'TOUCH')], { ignoreCase: true })[0].status).toBe('agree');
  });

  it('pairs different sampling within half the larger interval, and never drops a sample', () => {
    const a = [s(0, 'X'), s(5, 'X'), s(10, 'X')]; // every 5 s
    const b = [s(0, 'X'), s(2, 'X'), s(4, 'X'), s(6, 'X'), s(8, 'X'), s(10, 'X')]; // every 2 s
    const pairs = pairRuns(a, b);
    const summary = summarize(pairs, a.length, b.length);
    expect(summary.compared).toBe(3);
    expect(summary.unpaired).toBe(3);
    expect(summary.compared * 2 + summary.unpaired + summary.errors).toBe(a.length + b.length);
    expect(pairs.find((p) => p.time === 5)?.b?.timestamp_sec).toBe(4);
  });

  it('counts error samples separately, outside the agreement rate', () => {
    const pairs = pairRuns([s(0, 'TOUCH'), s(5, 'ERROR: timeout')], [s(0, 'TOUCH'), s(5, 'TOUCH')]);
    const summary = summarize(pairs, 2, 2);
    expect(summary).toMatchObject({ compared: 1, agree: 1, errors: 1, agreementRate: 1 });
  });

  it('tabulates label pairs', () => {
    const pairs = pairRuns([s(0, 'TOUCH'), s(5, 'TOUCH'), s(10, 'NO_TOUCH')], [s(0, 'NO_TOUCH'), s(5, 'NO_TOUCH'), s(10, 'NO_TOUCH')]);
    expect(summarize(pairs, 3, 3).labelPairs).toEqual([
      { a: 'TOUCH', b: 'NO_TOUCH', count: 2 },
      { a: 'NO_TOUCH', b: 'NO_TOUCH', count: 1 },
    ]);
  });
});

describe('ground truth', () => {
  it('scores each run against the ELAN category', () => {
    const truth = [{ tier: 'Static touch', startSec: 0, endSec: 6, value: '' }] as unknown as ElanTierAnnotation[];
    const pairs = withTruth(pairRuns([s(0, 'TOUCH'), s(10, 'TOUCH')], [s(0, 'NO_TOUCH'), s(10, 'NO_TOUCH')]), truth);
    const summary = summarize(pairs, 2, 2);
    expect(pairs.map((p) => p.truth)).toEqual(['MATERNAL_TOUCH', 'NO_TOUCH']);
    expect(summary.truth).toEqual({ compared: 2, agreeA: 1, agreeB: 1 });
  });
});

it('writes a CSV that quotes free text', () => {
  const csv = comparisonCsv(pairRuns([s(0, 'TOUCH', 'hand, on "arm"')], [s(0, 'TOUCH', 'x')]), 'a1', 'b2');
  expect(csv.split('\n')[0]).toBe('time_sec,status,label_a1,label_b2,ground_truth,reasoning_a1,reasoning_b2');
  expect(csv).toContain('"hand, on ""arm"""');
});
