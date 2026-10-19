import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import { Maximize2, ZoomIn, ZoomOut } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { LaneCanvas } from '@/components/lanes/LaneCanvas';
import { LaneHeader, type LaneToggles } from '@/components/lanes/LaneHeader';
import { Playhead } from '@/components/lanes/Playhead';
import { RangeBar } from '@/components/lanes/RangeBar';
import { RunMenu, type RunControls } from '@/components/lanes/RunMenu';
import { TimeAxis } from '@/components/lanes/TimeAxis';
import {
  LANE_HEADER_WIDTH,
  ZOOM_STEP,
  eventAt,
  formatClockTenths,
  formatValue,
  fullView,
  laneHeight,
  panView,
  rowHeight,
  showsDensity,
  timeAtClickInView,
  zoomView,
  type Lane,
  type LaneEvent,
  type LaneRow,
  type ViewRange,
} from '@/lib/lanes';

interface LaneTimelineProps {
  lanes: Lane[];
  duration: number;
  currentTime: number;
  view: ViewRange;
  onViewChange: (view: ViewRange) => void;
  onSeek: (time: number) => void;
  selectedLaneId: string | null;
  onSelectLane: (laneId: string) => void;
  /** The table row selected in the Data tab (SC-005). */
  selectedRow?: number | null;
  onSelectRow?: (row: number) => void;
  /** Left of the zoom buttons (Layers, spec 028 US2). */
  toolbar?: ReactNode;
  toggles: LaneToggles;
  /** The run menu on each lane; plain run labels without it (local files). */
  runControls?: RunControls;
}

interface Hover {
  x: number;
  y: number;
  lane: Lane;
  row: LaneRow;
  event: LaneEvent;
}

const HOVER_PX = 4;

function rowAt(lane: Lane, y: number): LaneRow | null {
  let top = 0;
  for (const row of lane.rows) {
    const height = rowHeight(row);
    if (y >= top && y < top + height) return row;
    top += height;
  }
  return null;
}

function HoverCard({ hover }: { hover: Hover }) {
  const { lane, row, event } = hover;
  const end = event.end !== null && event.end !== event.start ? `–${formatClockTenths(event.end)}` : '';
  return (
    <div
      role="tooltip"
      className="pointer-events-none absolute z-20 max-w-xs rounded border bg-popover p-2 text-xs text-popover-foreground shadow-md"
      style={{ left: hover.x + 12, top: hover.y + 12 }}
      data-testid="lane-hover"
    >
      <div className="font-medium">
        {lane.label}
        {row.label && row.label !== lane.label ? ` · ${row.label}` : ''}
      </div>
      {event.label && <div className="mt-0.5 whitespace-pre-wrap break-words">{event.label}</div>}
      <div className="mt-0.5 font-mono text-muted-foreground">
        {formatClockTenths(event.start)}
        {end}
        {event.samples && event.samples > 1 ? ` · ${event.samples} samples` : ''}
      </div>
      {event.value !== null && (
        <div className="text-muted-foreground">
          value {formatValue(event.value)}
          {row.unit ? ` ${row.unit}` : ''}
        </div>
      )}
      {event.confidence !== null && <div className="text-muted-foreground">confidence {formatValue(event.confidence)}</div>}
    </div>
  );
}

/** The Video page's lanes on one time axis, with zoom and pan (spec 028 US1, R6, R7). */
export function LaneTimeline({
  lanes,
  duration,
  currentTime,
  view,
  onViewChange,
  onSeek,
  selectedLaneId,
  onSelectLane,
  selectedRow,
  onSelectRow,
  toolbar,
  toggles,
  runControls,
}: LaneTimelineProps) {
  const container = useRef<HTMLDivElement>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const axis = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  const [hover, setHover] = useState<Hover | null>(null);
  const drag = useRef<{ x: number; view: ViewRange; moved: boolean } | null>(null);
  const viewRef = useRef(view);
  viewRef.current = view;

  useLayoutEffect(() => {
    const el = axis.current;
    if (!el) return;
    const measure = () => setWidth(el.getBoundingClientRect().width);
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // Ctrl/⌘+wheel zooms and a sideways wheel pans; a plain wheel scrolls the lanes.
  // Not React's onWheel: it is passive, so the page would zoom too.
  useEffect(() => {
    const el = scroller.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      const body = axis.current;
      if (!body) return;
      const current = viewRef.current;
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        const anchor = timeAtClickInView(e.clientX, body, current);
        onViewChange(zoomView(current, e.deltaY > 0 ? 1 / ZOOM_STEP : ZOOM_STEP, anchor, duration));
      } else if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
        e.preventDefault();
        const rect = body.getBoundingClientRect();
        const delta = e.shiftKey ? e.deltaY : e.deltaX;
        if (rect.width > 0) onViewChange(panView(current, (delta / rect.width) * (current.end - current.start), duration));
      }
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [duration, onViewChange]);

  const secondsPerPixel = (el: Element) => {
    const rect = el.getBoundingClientRect();
    return rect.width > 0 ? (view.end - view.start) / rect.width : 0;
  };

  const hitAt = (lane: Lane, e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const row = rowAt(lane, e.clientY - rect.top);
    if (!row) return null;
    const t = timeAtClickInView(e.clientX, e.currentTarget, view);
    const event = eventAt(row, t, HOVER_PX * secondsPerPixel(e.currentTarget));
    return { row, t, event };
  };

  return (
    <div ref={container} className="relative flex h-full min-h-0 flex-col border-t bg-card" id="lane-timeline">
      <div className="flex shrink-0 items-center gap-2 border-b px-2 py-1">
        {toolbar}
        <Button
          variant="ghost"
          size="icon"
          className="h-6 w-6"
          title="Zoom out (−)"
          aria-label="Zoom out"
          onClick={() => onViewChange(zoomView(view, 1 / ZOOM_STEP, currentTime, duration))}
        >
          <ZoomOut className="h-3.5 w-3.5" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          className="h-6 w-6"
          title="Zoom in (+)"
          aria-label="Zoom in"
          onClick={() => onViewChange(zoomView(view, ZOOM_STEP, currentTime, duration))}
        >
          <ZoomIn className="h-3.5 w-3.5" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          className="h-6 w-6"
          title="Whole video (0)"
          aria-label="Whole video"
          onClick={() => onViewChange(fullView(duration))}
        >
          <Maximize2 className="h-3.5 w-3.5" />
        </Button>
        <RangeBar view={view} duration={duration} onChange={onViewChange} />
        <span className="shrink-0 font-mono text-[10px] text-muted-foreground" data-testid="view-range">
          {formatClockTenths(view.start)}–{formatClockTenths(view.end)}
        </span>
      </div>

      <div ref={scroller} className="relative min-h-0 flex-1 overflow-y-auto overflow-x-hidden">
        <div className="relative">
          <div className="sticky top-0 z-10 flex bg-card">
            <div className="shrink-0" style={{ width: LANE_HEADER_WIDTH }} />
            <div
              ref={axis}
              className="min-w-0 flex-1 cursor-pointer"
              onClick={(e) => onSeek(timeAtClickInView(e.clientX, e.currentTarget, view))}
              data-testid="lane-axis"
            >
              <TimeAxis duration={duration} view={view} />
            </div>
          </div>

          {lanes.length === 0 && (
            <p className="px-3 py-4 text-xs text-muted-foreground" data-testid="lanes-empty">
              No results for this video yet.
            </p>
          )}

          {lanes.map((lane) => {
            const collapsed = toggles.collapsed(lane.kind);
            const densityRows = collapsed ? new Set<string>() : new Set(lane.rows.filter((row) => showsDensity(row, view, width)).map((row) => row.key));
            return (
              <div
                key={lane.id}
                className={lane.kind === 'notes' ? 'flex border-y border-dashed border-amber-500/70' : 'flex border-b border-border/50'}
                role="group"
                aria-label={lane.label}
                data-testid={`lane-${lane.id}`}
                data-kind={lane.kind}
              >
                <LaneHeader
                  lane={lane}
                  selected={lane.id === selectedLaneId}
                  densityRows={densityRows}
                  onSelect={() => onSelectLane(lane.id)}
                  toggles={toggles}
                />
                <div
                  className="relative min-w-0 flex-1 cursor-crosshair"
                  data-testid={`lane-body-${lane.id}`}
                  onPointerDown={(e) => {
                    drag.current = { x: e.clientX, view, moved: false };
                  }}
                  onPointerMove={(e) => {
                    const d = drag.current;
                    if (!d || e.buttons === 0) return;
                    if (Math.abs(e.clientX - d.x) > 3) d.moved = true;
                    if (d.moved) onViewChange(panView(d.view, -(e.clientX - d.x) * secondsPerPixel(e.currentTarget), duration));
                  }}
                  onMouseMove={(e) => {
                    if (collapsed) return;
                    const hit = hitAt(lane, e);
                    const box = container.current?.getBoundingClientRect();
                    setHover(
                      hit?.event && box
                        ? { x: e.clientX - box.left, y: e.clientY - box.top, lane, row: hit.row, event: hit.event }
                        : null,
                    );
                  }}
                  onMouseLeave={() => setHover(null)}
                  onClick={(e) => {
                    const dragged = drag.current?.moved;
                    drag.current = null;
                    if (dragged) return;
                    onSelectLane(lane.id);
                    const hit = collapsed ? null : hitAt(lane, e);
                    if (hit?.event) {
                      onSeek(hit.event.start);
                      onSelectRow?.(hit.event.row);
                    } else {
                      onSeek(timeAtClickInView(e.clientX, e.currentTarget, view));
                    }
                  }}
                >
                  {lane.status === 'failed' ? (
                    <p className="px-2 text-xs leading-[18px] text-destructive" data-testid="lane-failed">
                      Failed: {lane.error}
                    </p>
                  ) : lane.status === 'empty' ? (
                    <p className="px-2 text-[10px] leading-[18px] text-muted-foreground" data-testid="lane-empty">
                      {lane.error}
                    </p>
                  ) : collapsed ? (
                    <div style={{ height: laneHeight(lane, true) }} />
                  ) : (
                    <LaneCanvas lane={lane} view={view} width={width} selectedRow={selectedRow} />
                  )}
                  {lane.run &&
                    (runControls ? (
                      <RunMenu lane={lane} controls={runControls} />
                    ) : (
                      <span className="pointer-events-none absolute right-1 top-0 text-[10px] text-muted-foreground" data-testid="lane-run">
                        {lane.run.label}
                      </span>
                    ))}
                </div>
              </div>
            );
          })}

          <div className="pointer-events-none absolute inset-y-0 right-0" style={{ left: LANE_HEADER_WIDTH }}>
            <Playhead time={currentTime} duration={duration} view={view} />
          </div>
        </div>
      </div>
      {hover && <HoverCard hover={hover} />}
    </div>
  );
}
