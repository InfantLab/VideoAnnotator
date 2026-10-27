import { formatClockTenths, lanePercent, timeAtClick, type LaneMarker } from '@/lib/lanes';

interface AnswerLaneProps<M extends LaneMarker> {
  markers: M[];
  duration: number;
  /** Each answer's colour, the same answer the same colour (UAT #23). */
  colourOf: (marker: M) => string;
  highlight?: number | null;
  onSeek?: (t: number) => void;
  onHover?: (marker: M | null) => void;
}

const statusWord = (m: LaneMarker) =>
  m.status === 'error'
    ? 'no answer: the model failed'
    : m.status === 'skipped'
      ? 'skipped: no readable frame'
      : m.fit === false
        ? "didn't fit the answer shape"
        : m.label ?? '';

/**
 * One entry's answers along the video. An error, a skipped sample and an
 * answer that didn't fit its shape are drawn as their own shapes, never as an
 * answer (Principle VI).
 */
export function AnswerLane<M extends LaneMarker>({ markers, duration, colourOf, highlight, onSeek, onHover }: AnswerLaneProps<M>) {
  return (
    <svg
      className="block h-6 w-full overflow-visible"
      data-testid="answer-lane"
      onClick={(e) => {
        if (e.target === e.currentTarget) onSeek?.(timeAtClick(e.clientX, e.currentTarget, duration));
      }}
    >
      <line x1="0" x2="100%" y1="12" y2="12" stroke="currentColor" strokeOpacity={0.15} />
      {markers.map((m, i) => {
        const selected = highlight !== undefined && highlight !== null && Math.abs(m.t - highlight) < 1e-6;
        return (
          <svg
            key={`${m.t}-${i}`}
            x={`${lanePercent(m.t, duration)}%`}
            y="12"
            overflow="visible"
            className="cursor-pointer"
            data-testid="answer-marker"
            data-status={m.status === 'answered' && m.fit === false ? 'misfit' : m.status}
            data-label={m.label ?? ''}
            data-selected={selected ? 'true' : 'false'}
            role="button"
            aria-label={`${formatClockTenths(m.t)}: ${statusWord(m)}`}
            onClick={(e) => {
              e.stopPropagation();
              onSeek?.(m.t);
            }}
            onMouseEnter={() => onHover?.(m)}
            onMouseLeave={() => onHover?.(null)}
          >
            <title>{`${formatClockTenths(m.t)}: ${statusWord(m)}`}</title>
            {m.status === 'answered' && m.fit === false && (
              <circle
                r={selected ? 7 : 5}
                fill="none"
                stroke="hsl(38 92% 50%)"
                strokeWidth={2}
                strokeDasharray="2 2"
              />
            )}
            {m.status === 'answered' && m.fit !== false && (
              <circle r={selected ? 7 : 5} fill={colourOf(m)} stroke={selected ? 'hsl(var(--foreground))' : 'none'} strokeWidth={2} />
            )}
            {m.status === 'error' && (
              <g stroke="currentColor" strokeOpacity={0.6} strokeWidth={selected ? 2.5 : 1.5} fill="none">
                <circle r={selected ? 7 : 5} />
                <path d="M-3,-3L3,3M3,-3L-3,3" />
              </g>
            )}
            {m.status === 'skipped' && (
              <rect x={-4} y={-4} width={8} height={8} fill="none" stroke="currentColor" strokeOpacity={0.6} strokeDasharray="2 1" />
            )}
          </svg>
        );
      })}
    </svg>
  );
}
