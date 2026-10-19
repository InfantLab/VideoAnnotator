import fs from 'fs';
import path from 'path';
import { describe, expect, it } from 'vitest';
import { burstOffsetsFor, burstTimes, chooseMoments, samplePoints } from '@/lib/sampling';

interface Case {
  name: string;
  duration: number;
  interval: number;
  start: number;
  end: number | null;
  moments: number[] | null;
  frames: number;
  spacing: number | null;
  count: number;
  expect: { points: number[]; burst_times_at_first_point: Array<[number, number]>; chosen: number[] };
}

const CASES: Case[] = JSON.parse(
  fs.readFileSync(path.resolve(__dirname, '../../../../tests/fixtures/ask_ai/sampling_cases.json'), 'utf-8'),
);

const points = (c: Case) => samplePoints(c.duration, c.interval, c.start, c.end, c.moments);

describe('sampling twin of the pipeline', () => {
  it('has cases', () => expect(CASES.length).toBeGreaterThan(10));

  it.each(CASES.map((c) => [c.name, c] as const))('sample points: %s', (_, c) => {
    expect(points(c)).toEqual(c.expect.points);
  });

  it.each(CASES.map((c) => [c.name, c] as const))('burst times: %s', (_, c) => {
    const got = burstTimes(points(c)[0], burstOffsetsFor(c.frames), c.spacing ?? c.interval, c.duration);
    expect(got).toEqual(c.expect.burst_times_at_first_point);
  });

  it.each(CASES.map((c) => [c.name, c] as const))('chosen moments: %s', (_, c) => {
    expect(chooseMoments(points(c), c.count)).toEqual(c.expect.chosen);
  });

  it('centres burst offsets with one more before the anchor', () => {
    expect(burstOffsetsFor(4)).toEqual([-2, -1, 0, 1]);
    expect(burstOffsetsFor(1)).toEqual([0]);
  });
});
