import { lanePercent, timeAtClick } from '@/lib/lanes';

interface SampleTicksProps {
  times: number[];
  duration: number;
  highlight?: number | null;
  onSeek?: (t: number) => void;
}

/** Where samples fall: the points a run would send to the model. */
export function SampleTicks({ times, duration, highlight, onSeek }: SampleTicksProps) {
  return (
    <svg
      className="block h-4 w-full cursor-pointer text-muted-foreground"
      data-testid="sample-ticks"
      aria-label={`${times.length} samples`}
      onClick={(e) => onSeek?.(timeAtClick(e.clientX, e.currentTarget, duration))}
    >
      {times.map((t) => (
        <line
          key={t}
          data-testid="sample-tick"
          x1={`${lanePercent(t, duration)}%`}
          x2={`${lanePercent(t, duration)}%`}
          y1="2"
          y2="14"
          stroke={t === highlight ? 'hsl(var(--primary))' : 'currentColor'}
          strokeWidth={t === highlight ? 2.5 : 1}
        />
      ))}
    </svg>
  );
}
