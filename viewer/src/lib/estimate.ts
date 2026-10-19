/**
 * How long a run would take (spec 027 R14): this model's recent answers in
 * the history, else a rough guess, said to be one.
 */
import type { AskEntry } from '@/types/ask';

export const ROUGH_SECONDS_PER_FRAME = 6;
const RECENT = 20;

export interface Estimate {
  seconds: number;
  rough: boolean;
  overHour: boolean;
}

const median = (values: number[]) => {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

/** Seconds per frame from the model's last answers, newest entries first. */
function measuredPerFrame(model: string, history: AskEntry[]): number | null {
  const perFrame: number[] = [];
  for (const entry of history) {
    if (entry.model !== model) continue;
    for (const a of [...(entry.answers ?? [])].reverse()) {
      if (a.status !== 'answered' || a.total_time == null) continue;
      const frames = a.frame_numbers?.length || entry.sampling.frames || 1;
      perFrame.push(a.total_time / frames);
      if (perFrame.length >= RECENT) return median(perFrame);
    }
  }
  return perFrame.length ? median(perFrame) : null;
}

export function estimate(samples: number, frames: number, model: string, history: AskEntry[]): Estimate {
  const measured = measuredPerFrame(model, history);
  const seconds = samples * Math.max(1, frames) * (measured ?? ROUGH_SECONDS_PER_FRAME);
  return { seconds, rough: measured === null, overHour: seconds > 3600 };
}

/** "about 40 s", "about 3 min", "about 1 h 20 min" */
export function formatEstimate(seconds: number): string {
  if (seconds < 60) return `about ${Math.max(1, Math.round(seconds))} s`;
  if (seconds < 3600) return `about ${Math.round(seconds / 60)} min`;
  const h = Math.floor(seconds / 3600);
  const m = Math.round((seconds % 3600) / 60);
  return m ? `about ${h} h ${m} min` : `about ${h} h`;
}
