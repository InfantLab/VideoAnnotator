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

/** Seconds from "0:12, 0:47, 1:30" or "12, 47.5"; anything else is left out. */
export function parseMoments(text: string): number[] {
  return text
    .split(/[,;\s]+/)
    .filter(Boolean)
    .map((part) => {
      const pieces = part.split(':');
      if (pieces.length > 3 || pieces.some((p) => p === '' || !/^\d*\.?\d+$/.test(p))) return NaN;
      return pieces.reduce((total, p) => total * 60 + Number(p), 0);
    })
    .filter((n) => Number.isFinite(n) && n >= 0);
}

export interface WindowSampling {
  interval?: number | null;
  start?: number | null;
  end?: number | null;
  moments?: number[] | null;
  frames?: number | null;
  spacing?: number | null;
}

/** The window's sampling as vlm_annotation settings: the twin of ask.py's sampling_to_config. */
export function samplingToConfig(s: WindowSampling): Record<string, unknown> {
  const frames = s.frames ?? 1;
  const config: Record<string, unknown> = {
    frame_interval_sec: s.interval ?? 5,
    start_sec: s.start ?? 0,
    end_sec: s.end ?? null,
    moments_sec: s.moments ? [...new Set(s.moments)].sort((a, b) => a - b) : null,
    sampling_mode: frames === 1 ? 'single_frame' : 'frame_burst',
  };
  if (frames > 1) {
    config.burst_offsets = burstOffsetsFor(frames);
    config.burst_spacing_sec = s.spacing ?? null;
  }
  return config;
}

/** vlm_annotation settings as the window's sampling: the inverse of samplingToConfig. */
export function samplingFromConfig(config: Record<string, unknown>): WindowSampling {
  const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
  const offsets = Array.isArray(config.burst_offsets) ? config.burst_offsets : [-2, -1, 0, 1, 2];
  const burst = config.sampling_mode === 'frame_burst';
  const interval = num(config.frame_interval_sec) ?? 5;
  return {
    interval,
    start: num(config.start_sec) ?? 0,
    end: num(config.end_sec),
    moments: Array.isArray(config.moments_sec) ? config.moments_sec.filter((m): m is number => typeof m === 'number') : null,
    frames: burst ? offsets.length : 1,
    spacing: burst ? num(config.burst_spacing_sec) ?? interval : null,
  };
}
