import { cn } from '@/lib/utils';
import { LANE_HEADER_WIDTH, laneTooltip, rowHeight, rowTooltip, type Lane } from '@/lib/lanes';

interface LaneHeaderProps {
  lane: Lane;
  selected: boolean;
  /** Keys of rows drawn as density in the current view. */
  densityRows: Set<string>;
  onSelect: () => void;
}

/** A lane's label and its rows' labels, left of its canvas (spec 028 US1). */
export function LaneHeader({ lane, selected, densityRows, onSelect }: LaneHeaderProps) {
  const height = Math.max(lane.rows.reduce((sum, row) => sum + rowHeight(row), 0), 18);
  return (
    <div
      className={cn('flex shrink-0 border-l-2 text-xs', selected ? 'border-l-primary bg-primary/5' : 'border-l-transparent')}
      style={{ width: LANE_HEADER_WIDTH, height }}
    >
      <button
        type="button"
        className="w-24 shrink-0 truncate px-1 text-left font-medium leading-[18px] hover:underline"
        title={laneTooltip(lane)}
        aria-pressed={selected}
        onClick={onSelect}
        data-testid="lane-label"
      >
        {lane.label}
      </button>
      <div className="min-w-0 flex-1">
        {lane.rows.map((row) => (
          <button
            type="button"
            key={row.key}
            className="flex w-full min-w-0 items-center gap-1 pr-1 text-left text-muted-foreground"
            style={{ height: rowHeight(row) }}
            title={rowTooltip(lane, row)}
            onClick={onSelect}
            data-testid="row-label"
          >
            <span className="h-2 w-2 shrink-0 rounded-sm" style={{ background: row.colour }} aria-hidden="true" />
            <span className="truncate">{row.label}</span>
            {densityRows.has(row.key) && <span className="ml-auto shrink-0 text-[9px] uppercase">density</span>}
          </button>
        ))}
      </div>
    </div>
  );
}
