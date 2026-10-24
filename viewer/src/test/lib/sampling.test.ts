import fs from 'fs';
import path from 'path';
import { describe, expect, it } from 'vitest';
import { burstOffsetsFor, burstTimes, chooseMoments, clipSpan, clipTimes, samplePoints } from '@/lib/sampling';

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
  clip?: { length: number; fps: number };
  expect: {
    points: number[];
    burst_times_at_first_point: Array<[number, number]>;
    chosen: number[];
    clip_times_at_first_point?: number[];
    clip_span?: [number, number];
  };
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

  const clips = CASES.filter((c) => c.clip);
  it('has clip cases', () => expect(clips.length).toBeGreaterThanOrEqual(4));

  it.each(clips.map((c) => [c.name, c] as const))('clip span and times: %s', (_, c) => {
    const t = points(c)[0];
    expect(clipSpan(t, c.clip!.length, c.duration)).toEqual(c.expect.clip_span);
    expect(clipTimes(t, c.clip!.length, c.clip!.fps, c.duration)).toEqual(c.expect.clip_times_at_first_point);
  });

  it('centres burst offsets with one more before the anchor', () => {
    expect(burstOffsetsFor(4)).toEqual([-2, -1, 0, 1]);
    expect(burstOffsetsFor(1)).toEqual([0]);
  });
});
