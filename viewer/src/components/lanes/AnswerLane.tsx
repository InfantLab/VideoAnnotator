import { lanePercent, timeAtClick, type LaneMarker } from '@/lib/lanes';

interface AnswerLaneProps<M extends LaneMarker> {
  markers: M[];
  duration: number;
  /** Hue per distinct answer, so the same answer has the same colour. */
  hues: Map<string, number>;
  highlight?: number | null;
  onSeek?: (t: number) => void;
  onHover?: (marker: M | null) => void;
}

const statusWord = (m: LaneMarker) =>
  m.status === 'error' ? 'no answer: the model failed' : m.status === 'skipped' ? 'skipped: no readable frame' : m.label ?? '';

/**
 * One entry's answers along the video. An error and a skipped sample are
 * drawn as their own shapes, never as an answer (Principle VI).
 */
export function AnswerLane<M extends LaneMarker>({ markers, duration, hues, highlight, onSeek, onHover }: AnswerLaneProps<M>) {
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
        const hue = hues.get(m.label ?? '') ?? 0;
        return (
          <svg
            key={`${m.t}-${i}`}
            x={`${lanePercent(m.t, duration)}%`}
            y="12"
            overflow="visible"
            className="cursor-pointer"
            data-testid="answer-marker"
            data-status={m.status}
            data-label={m.label ?? ''}
            data-selected={selected ? 'true' : 'false'}
            role="button"
            aria-label={`${m.t.toFixed(1)} s: ${statusWord(m)}`}
            onClick={(e) => {
              e.stopPropagation();
              onSeek?.(m.t);
            }}
            onMouseEnter={() => onHover?.(m)}
            onMouseLeave={() => onHover?.(null)}
          >
            <title>{`${m.t.toFixed(1)} s: ${statusWord(m)}`}</title>
            {m.status === 'answered' && (
              <circle r={selected ? 7 : 5} fill={`hsl(${hue} 70% 50%)`} stroke={selected ? 'hsl(var(--foreground))' : 'none'} strokeWidth={2} />
            )}
            {m.status === 'error' && (
              <path d="M-5,-5L5,5M5,-5L-5,5" stroke="hsl(var(--destructive))" strokeWidth={selected ? 3 : 2} />
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
