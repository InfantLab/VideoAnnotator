import { lanePercent } from '@/lib/lanes';

interface PlayheadProps {
  time: number;
  duration: number;
}

/** Laid over the lanes (the parent is `relative`). */
export function Playhead({ time, duration }: PlayheadProps) {
  return (
    <div
      className="pointer-events-none absolute inset-y-0 w-px bg-primary"
      style={{ left: `${lanePercent(time, duration)}%` }}
      data-testid="playhead"
    />
  );
}
