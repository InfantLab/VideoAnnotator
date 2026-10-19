import { lanePercent, viewPercent, type ViewRange } from '@/lib/lanes';

interface PlayheadProps {
  time: number;
  duration: number;
  view?: ViewRange;
}

/** Laid over the lanes (the parent is `relative`). */
export function Playhead({ time, duration, view }: PlayheadProps) {
  const percent = view ? viewPercent(time, view) : lanePercent(time, duration);
  if (percent < 0 || percent > 100) return null;
  return (
    <div
      className="pointer-events-none absolute inset-y-0 w-px bg-primary"
      style={{ left: `${percent}%` }}
      data-testid="playhead"
    />
  );
}
