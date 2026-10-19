import { axisTicks, formatClock, formatClockTenths, lanePercent, viewPercent, viewTicks, type ViewRange } from '@/lib/lanes';

interface TimeAxisProps {
  duration: number;
  /** The stretch shown (spec 028); the whole video when absent. */
  view?: ViewRange;
}

export function TimeAxis({ duration, view }: TimeAxisProps) {
  const fine = view ? viewTicks(view) : null;
  const ticks = fine ? fine.ticks : axisTicks(duration);
  const at = (t: number) => (view ? viewPercent(t, view) : lanePercent(t, duration));
  const end = view ? view.end : duration;
  const start = view ? view.start : 0;
  const label = (t: number) => (fine && fine.step < 1 ? formatClockTenths(t) : formatClock(t));
  return (
    <svg className="block h-6 w-full overflow-visible text-muted-foreground" role="presentation" data-testid="time-axis">
      <line x1="0" x2="100%" y1="1" y2="1" stroke="currentColor" strokeOpacity={0.4} />
      {ticks.map((t) => (
        <g key={t}>
          <line x1={`${at(t)}%`} x2={`${at(t)}%`} y1="1" y2="6" stroke="currentColor" />
          <text
            x={`${at(t)}%`}
            y="18"
            fontSize="10"
            fill="currentColor"
            textAnchor={t <= start ? 'start' : t >= end ? 'end' : 'middle'}
          >
            {label(t)}
          </text>
        </g>
      ))}
    </svg>
  );
}
