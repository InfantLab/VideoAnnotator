import type { ReactNode } from 'react';
import { ChevronDown, ChevronRight, Eye, EyeOff, SlidersHorizontal } from 'lucide-react';

import { Checkbox } from '@/components/ui/checkbox';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { cn } from '@/lib/utils';
import { OPENFACE3_OVERLAYS } from '@/lib/laneOverlays';
import { formatValue, LANE_HEADER_WIDTH, laneHeight, laneTooltip, rowHeight, rowTooltip, type Lane } from '@/lib/lanes';
import { kindColour, kindOfLane } from '@/lib/resultKinds';

export interface LaneToggles {
  collapsed: (kind: string) => boolean;
  overlayOn: (key: string) => boolean;
  toggleCollapsed: (kind: string) => void;
  setOverlay: (key: string, on: boolean) => void;
}

interface LaneHeaderProps {
  lane: Lane;
  selected: boolean;
  /** Keys of rows drawn as density in the current view. */
  densityRows: Set<string>;
  onSelect: () => void;
  toggles: LaneToggles;
  /** The lane's run menu, at the end of its name (UAT #37). */
  run?: ReactNode;
}

/** OpenFace 3's extra drawings, switched from its Faces lane (spec 028 T023). */
function OpenFace3Menu({ toggles }: { toggles: LaneToggles }) {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          className="shrink-0 rounded p-0.5 text-muted-foreground hover:text-foreground"
          aria-label="OpenFace 3 drawings"
          title="What OpenFace 3 draws on the video"
        >
          <SlidersHorizontal className="h-3 w-3" />
        </button>
      </PopoverTrigger>
      <PopoverContent className="w-48 p-2 text-xs" align="start">
        <p className="mb-1 font-medium">Draw on the video</p>
        {OPENFACE3_OVERLAYS.map(({ key, label }) => (
          <label key={key} className="flex items-center gap-2 py-0.5">
            <Checkbox
              checked={toggles.overlayOn(key)}
              onCheckedChange={(on) => toggles.setOverlay(key, on === true)}
              aria-label={label}
            />
            {label}
          </label>
        ))}
      </PopoverContent>
    </Popover>
  );
}

/** A lane's label and its rows' labels, left of its canvas (spec 028 US1, US2). */
/** The lane's kind, as an icon in its colour (viewer design "Icons"). */
function KindIcon({ lane }: { lane: Lane }) {
  const kind = kindOfLane(lane.kind, lane.pipeline);
  if (!kind) return null;
  const Icon = kind.icon;
  return <Icon className="mx-0.5 h-3.5 w-3.5 shrink-0" style={{ color: kindColour(kind) }} aria-hidden="true" />;
}

export function LaneHeader({ lane, selected, densityRows, onSelect, toggles, run }: LaneHeaderProps) {
  const collapsed = toggles.collapsed(lane.kind);
  const overlay = lane.overlay;
  const overlayShown = overlay ? toggles.overlayOn(overlay) : false;
  return (
    <div
      className={cn('flex shrink-0 border-l-2 text-xs', selected ? 'border-l-primary bg-primary/5' : 'border-l-transparent')}
      style={{ width: LANE_HEADER_WIDTH, height: laneHeight(lane, collapsed) }}
    >
      <div className="flex w-[8.5rem] shrink-0 items-start gap-0.5 pl-0.5">
        {overlay ? (
          <button
            type="button"
            className="mt-[3px] shrink-0 text-muted-foreground hover:text-foreground"
            aria-label={`${overlayShown ? 'Hide' : 'Show'} ${lane.label} on the video`}
            aria-pressed={overlayShown}
            title={`${overlayShown ? 'Hide' : 'Show'} on the video`}
            onClick={() => toggles.setOverlay(overlay, !overlayShown)}
            data-testid="lane-eye"
          >
            {overlayShown ? <Eye className="h-3 w-3" /> : <EyeOff className="h-3 w-3" />}
          </button>
        ) : (
          <span className="w-3 shrink-0" />
        )}
        <button
          type="button"
          className="flex min-w-0 flex-1 items-center gap-0.5 text-left font-medium leading-[18px] hover:underline"
          title={`${laneTooltip(lane)} Click to ${collapsed ? 'expand' : 'collapse'}.`}
          aria-expanded={!collapsed}
          onClick={() => toggles.toggleCollapsed(lane.kind)}
          data-testid="lane-label"
        >
          {collapsed ? <ChevronRight className="h-3 w-3 shrink-0" /> : <ChevronDown className="h-3 w-3 shrink-0" />}
          <KindIcon lane={lane} />
          <span className={cn('truncate', lane.kind === 'notes' && 'italic text-amber-700 dark:text-amber-400')}>{lane.label}</span>
        </button>
        {overlay === 'faces.openface3' && <OpenFace3Menu toggles={toggles} />}
        {run}
      </div>
      <div className="min-w-0 flex-1">
        {!collapsed &&
          lane.rows.map((row) => (
            <button
              type="button"
              key={row.key}
              className="flex w-full min-w-0 items-center gap-1 pr-1 text-left text-muted-foreground"
              style={{ height: rowHeight(row) }}
              title={rowTooltip(lane, row)}
              aria-pressed={selected}
              onClick={onSelect}
              data-testid="row-label"
            >
              <span className="h-2 w-2 shrink-0 rounded-sm" style={{ background: row.colour }} aria-hidden="true" />
              <span className="truncate">{row.label}</span>
              {densityRows.has(row.key) ? (
                <span className="ml-auto shrink-0 text-[9px] uppercase">density</span>
              ) : (
                // The scale beside the row's name, not over its line (UAT #37).
                row.form === 'numeric' &&
                row.min !== undefined &&
                row.max !== undefined && (
                  <span className="ml-auto shrink-0 text-[9px] tabular-nums" data-testid="row-scale">
                    {formatValue(row.min)}–{formatValue(row.max)}
                    {row.unit ?? ''}
                  </span>
                )
              )}
            </button>
          ))}
      </div>
    </div>
  );
}
