/**
 * Where a VLM run samples: the twin of
 * src/videoannotator/pipelines/vlm_annotation/sampling.py (spec 027 R2).
 * Both are pinned by tests/fixtures/ask_ai/sampling_cases.json, so the ticks
 * drawn here are the points the pipeline samples.
 */

const EPS = 1e-9;

const round3 = (t: number): number => Math.round(t * 1000) / 1000;

/** Round half up, as sampling.py does (Python's round() is half-to-even). */
const halfUp = (x: number): number => Math.floor(x + 0.5);

/** Listed moments within the video when given, else the interval grid. */
export function samplePoints(
  duration: number,
  interval: number,
  start = 0,
  end: number | null = null,
  moments: number[] | null = null,
): number[] {
  if (moments !== null) {
    const kept = new Set(moments.filter((t) => t >= 0 && t <= duration + EPS).map(round3));
    return [...kept].sort((a, b) => a - b);
  }
  if (!(interval > 0)) throw new Error('interval must be greater than 0');
  const stop = end === null ? duration : Math.min(end, duration);
  const points: number[] = [];
  for (let k = 0; ; k++) {
    const t = start + k * interval;
    if (t > stop + EPS) break;
    points.push(round3(t));
  }
  return points;
}

/** `frames` consecutive offsets around 0; an even count has one more before. */
export function burstOffsetsFor(frames: number): number[] {
  const before = Math.floor(frames / 2);
  return Array.from({ length: frames }, (_, i) => i - before);
}

/** [offset, time] of each burst frame inside the video. */
export function burstTimes(
  anchor: number,
  offsets: number[],
  spacing: number,
  duration: number,
): Array<[number, number]> {
  const out: Array<[number, number]> = [];
  for (const off of offsets) {
    const t = anchor + off * spacing;
    if (t < -EPS || t > duration + EPS) continue;
    out.push([off, round3(Math.max(0, t))]);
  }
  return out;
}

/** `count` points spread evenly over `points` (research R3). */
export function chooseMoments(points: number[], count: number): number[] {
  const s = points.length;
  if (count <= 0 || s === 0) return [];
  if (s <= count) return [...points];
  if (count === 1) return [points[halfUp((s - 1) / 2)]];
  return Array.from({ length: count }, (_, i) => points[halfUp((i * (s - 1)) / (count - 1))]);
}
