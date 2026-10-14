import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

import { HOLD_FACTOR, sampledAtTime } from '@/lib/sampledAtTime';

const at = (timestamp: number, id = 0) => ({ timestamp, id });

describe('sampledAtTime', () => {
  // Sampled about once a second, as face analysis is; nothing detected at 3.0.
  const faces = [at(0.0, 1), at(0.0, 2), at(1.0, 1), at(2.0, 1), at(4.0, 1)];

  it('holds a frame\'s detections between samples, not only near the timestamp', () => {
    expect(sampledAtTime(faces, 0.5).map((f) => f.id)).toEqual([1, 2]);
    expect(sampledAtTime(faces, 1.0)).toEqual([at(1.0, 1)]);
    expect(sampledAtTime(faces, 1.9)).toEqual([at(1.0, 1)]);
  });

  it('draws one frame at a time, never several frames stacked', () => {
    expect(sampledAtTime(faces, 1.05)).toHaveLength(1);
  });

  it('drops a detection once it is HOLD_FACTOR × the sample interval old', () => {
    expect(sampledAtTime(faces, 2.0 + HOLD_FACTOR - 0.01)).toEqual([at(2.0, 1)]);
    expect(sampledAtTime(faces, 2.0 + HOLD_FACTOR + 0.01)).toEqual([]);
  });

  it('shows nothing before the first sample', () => {
    expect(sampledAtTime([at(1.0)], 0.5)).toEqual([]);
    expect(sampledAtTime([], 0.5)).toEqual([]);
  });

  it('takes the interval from the data', () => {
    const dense = [at(0.0), at(0.2), at(0.4)];
    expect(sampledAtTime(dense, 0.4 + 0.25)).toEqual([at(0.4)]);
    expect(sampledAtTime(dense, 0.4 + 0.35)).toEqual([]);
  });

  it('keeps a face on screen through a real face-analysis job', () => {
    const fixture = path.resolve(__dirname, '../../../../tests/fixtures/viewer_contract/demo_clip_face_detections.json');
    const annotations: { timestamp: number }[] = JSON.parse(readFileSync(fixture, 'utf8')).annotations;
    const first = Math.min(...annotations.map((a) => a.timestamp));
    const last = Math.max(...annotations.map((a) => a.timestamp));

    // Sampled every ~0.97 s; under the old ±0.1 s rule most of these times showed no face.
    const shown = [];
    for (let t = first; t <= last; t += 0.1) shown.push(sampledAtTime(annotations, t).length > 0);
    expect(shown.filter(Boolean).length / shown.length).toBeGreaterThan(0.9);
  });
});
