import { useRef } from 'react';

import { centreView, panView, type ViewRange } from '@/lib/lanes';

interface RangeBarProps {
  view: ViewRange;
  duration: number;
  onChange: (view: ViewRange) => void;
}

/** Where in the video the timeline's view is; drag the window to move it (spec 028 R7). */
export function RangeBar({ view, duration, onChange }: RangeBarProps) {
  const track = useRef<HTMLDivElement>(null);
  const drag = useRef<{ x: number; view: ViewRange } | null>(null);
  const left = duration > 0 ? (view.start / duration) * 100 : 0;
  const width = duration > 0 ? ((view.end - view.start) / duration) * 100 : 100;

  const secondsPerPixel = () => {
    const rect = track.current?.getBoundingClientRect();
    return rect && rect.width > 0 ? duration / rect.width : 0;
  };

  return (
    <div
      ref={track}
      className="relative h-2 w-40 shrink cursor-pointer rounded-full bg-muted"
      role="scrollbar"
      aria-orientation="horizontal"
      aria-controls="lane-timeline"
      aria-valuemin={0}
      aria-valuemax={Math.round(duration)}
      aria-valuenow={Math.round(view.start)}
      aria-label="Visible range"
      data-testid="range-bar"
      onPointerDown={(e) => {
        const rect = e.currentTarget.getBoundingClientRect();
        if (rect.width > 0) onChange(centreView(view, ((e.clientX - rect.left) / rect.width) * duration, duration));
      }}
    >
      <div
        className="absolute inset-y-0 cursor-grab rounded-full bg-muted-foreground/50 hover:bg-muted-foreground/70"
        style={{ left: `${left}%`, width: `${Math.max(width, 0.5)}%` }}
        data-testid="range-window"
        onPointerDown={(e) => {
          e.stopPropagation();
          drag.current = { x: e.clientX, view };
          e.currentTarget.setPointerCapture?.(e.pointerId);
        }}
        onPointerMove={(e) => {
          if (!drag.current) return;
          onChange(panView(drag.current.view, (e.clientX - drag.current.x) * secondsPerPixel(), duration));
        }}
        onPointerUp={() => {
          drag.current = null;
        }}
      />
    </div>
  );
}
