import { axisTicks, formatClock, lanePercent } from '@/lib/lanes';

interface TimeAxisProps {
  duration: number;
}

export function TimeAxis({ duration }: TimeAxisProps) {
  const ticks = axisTicks(duration);
  return (
    <svg className="block h-6 w-full overflow-visible text-muted-foreground" role="presentation" data-testid="time-axis">
      <line x1="0" x2="100%" y1="1" y2="1" stroke="currentColor" strokeOpacity={0.4} />
      {ticks.map((t) => (
        <g key={t}>
          <line x1={`${lanePercent(t, duration)}%`} x2={`${lanePercent(t, duration)}%`} y1="1" y2="6" stroke="currentColor" />
          <text
            x={`${lanePercent(t, duration)}%`}
            y="18"
            fontSize="10"
            fill="currentColor"
            textAnchor={t === 0 ? 'start' : t >= duration ? 'end' : 'middle'}
          >
            {formatClock(t)}
          </text>
        </g>
      ))}
    </svg>
  );
}
