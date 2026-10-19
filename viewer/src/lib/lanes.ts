/** Shared by the lane timeline (spec 027 R15; spec 028 extends it). */

/** Where `t` sits along a lane, as a percentage of its width. */
export const lanePercent = (t: number, duration: number): number =>
  duration > 0 ? Math.min(100, Math.max(0, (t / duration) * 100)) : 0;

/** "m:ss", or "h:mm:ss" from an hour. */
export function formatClock(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = String(s % 60).padStart(2, '0');
  return h > 0 ? `${h}:${String(m).padStart(2, '0')}:${ss}` : `${m}:${ss}`;
}

const STEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];

/** Tick times for an axis: about `target` of them, at a round step. */
export function axisTicks(duration: number, target = 8): number[] {
  if (!(duration > 0)) return [0];
  const step = STEPS.find((s) => duration / s <= target) ?? STEPS[STEPS.length - 1];
  const ticks: number[] = [];
  for (let t = 0; t <= duration + 1e-9; t += step) ticks.push(t);
  return ticks;
}

/** The time a click at `clientX` on `element` points to. */
export function timeAtClick(clientX: number, element: Element, duration: number): number {
  const rect = element.getBoundingClientRect();
  if (rect.width <= 0) return 0;
  return Math.min(duration, Math.max(0, ((clientX - rect.left) / rect.width) * duration));
}

export type MarkerStatus = 'answered' | 'error' | 'skipped';

export interface LaneMarker {
  t: number;
  status: MarkerStatus;
  label?: string | null;
}
