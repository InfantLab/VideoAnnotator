import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { LocateFixed } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { filterRows, rowDetails, shownAtTime } from '@/lib/dataTable';
import { LANE_KINDS, tableKindOf } from '@/lib/laneKinds';
import { formatClockTenths, formatValue } from '@/lib/lanes';
import type { Row } from '@/lib/table/rows';
import { cn } from '@/lib/utils';

export const DATA_ROW_HEIGHT = 24;
const OVERSCAN = 10;
const FALLBACK_VIEWPORT = 600;

interface DataTableProps {
  /** Sorted by start, as the lanes were built from them. */
  rows: Row[];
  currentTime: number;
  /** The row picked on the timeline (SC-005). */
  selectedRow: number | null;
  onRowClick: (index: number) => void;
  filter: string;
  onFilterChange: (kind: string) => void;
  runLabels: Record<string, string>;
}

function cellText(row: Row): string {
  if (row.label && row.value !== null) return `${row.label} ${formatValue(row.value)}`;
  if (row.label) return row.label;
  if (row.value !== null) return formatValue(row.value);
  return row.confidence !== null ? `confidence ${formatValue(row.confidence)}` : '';
}

/** The video's results table beside the video, following playback (spec 028 US3). */
export function DataTable({ rows, currentTime, selectedRow, onRowClick, filter, onFilterChange, runLabels }: DataTableProps) {
  const [search, setSearch] = useState('');
  const [following, setFollowing] = useState(true);
  const [scrollTop, setScrollTop] = useState(0);
  const [viewport, setViewport] = useState(FALLBACK_VIEWPORT);
  const scroller = useRef<HTMLDivElement>(null);
  // Where we last scrolled to: a scroll landing elsewhere is the reader's.
  const ourScroll = useRef<number | null>(null);

  const kinds = useMemo(() => {
    const present = new Set(rows.map((r) => tableKindOf(r).kind));
    const known = LANE_KINDS.filter((k) => present.has(k.kind)).map((k) => ({ kind: k.kind, label: k.label }));
    const other = [...present].filter((k) => !known.some((x) => x.kind === k)).map((k) => ({ kind: k, label: k }));
    return [...known, ...other];
  }, [rows]);
  const kind = filter === 'all' || kinds.some((k) => k.kind === filter) ? filter : 'all';
  const shown = useMemo(() => filterRows(rows, kind, search), [rows, kind, search]);
  const playing = shownAtTime(rows, shown, currentTime);
  const selectedPosition = selectedRow === null ? -1 : shown.indexOf(selectedRow);
  const highlight = selectedPosition >= 0 && !following ? selectedPosition : playing;

  useLayoutEffect(() => {
    const el = scroller.current;
    if (el && el.clientHeight > 0) setViewport(el.clientHeight);
  }, []);

  const scrollToPosition = (position: number) => {
    const el = scroller.current;
    if (!el || position < 0) return;
    const top = position * DATA_ROW_HEIGHT;
    const height = el.clientHeight || viewport;
    if (top >= el.scrollTop && top + DATA_ROW_HEIGHT <= el.scrollTop + height) return;
    el.scrollTop = Math.max(0, top - height / 3);
    ourScroll.current = el.scrollTop;
    setScrollTop(el.scrollTop);
  };

  useEffect(() => {
    if (following) scrollToPosition(playing);
  }, [following, playing]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (selectedPosition >= 0) {
      setFollowing(false);
      scrollToPosition(selectedPosition);
    }
  }, [selectedRow]); // eslint-disable-line react-hooks/exhaustive-deps

  const first = Math.max(0, Math.floor(scrollTop / DATA_ROW_HEIGHT) - OVERSCAN);
  const last = Math.min(shown.length, Math.ceil((scrollTop + viewport) / DATA_ROW_HEIGHT) + OVERSCAN);

  return (
    <div className="flex h-full min-h-0 flex-col text-xs" data-testid="data-table">
      <div className="flex items-center gap-2 px-3 pb-2">
        <Select value={kind} onValueChange={onFilterChange}>
          <SelectTrigger className="h-7 w-32 text-xs" aria-label="Lane">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All lanes</SelectItem>
            {kinds.map((k) => (
              <SelectItem key={k.kind} value={k.kind}>
                {k.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Input
          className="h-7 text-xs"
          placeholder="Search"
          aria-label="Search the rows"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>
      <div className="flex items-center justify-between px-3 pb-1 text-muted-foreground">
        <span data-testid="row-count">
          {shown.length === rows.length
            ? `${rows.length.toLocaleString()} rows`
            : `${shown.length.toLocaleString()} of ${rows.length.toLocaleString()} rows`}
        </span>
        {!following && (
          <Button variant="ghost" size="sm" className="h-6 px-2 text-xs" onClick={() => setFollowing(true)}>
            <LocateFixed className="mr-1 h-3 w-3" />
            Follow playback
          </Button>
        )}
      </div>
      <div
        ref={scroller}
        className="relative min-h-0 flex-1 overflow-y-auto border-t"
        role="table"
        aria-label="Results for this video"
        aria-rowcount={shown.length}
        onScroll={(e) => {
          setScrollTop(e.currentTarget.scrollTop);
          const ours = ourScroll.current !== null && Math.abs(e.currentTarget.scrollTop - ourScroll.current) < 2;
          ourScroll.current = null;
          if (!ours) setFollowing(false);
        }}
      >
        <div style={{ height: shown.length * DATA_ROW_HEIGHT, position: 'relative' }}>
          {shown.slice(first, last).map((index, i) => {
            const row = rows[index];
            const position = first + i;
            const text = cellText(row);
            return (
              <div
                key={index}
                role="row"
                aria-rowindex={position + 1}
                aria-selected={position === highlight}
                className={cn(
                  'absolute inset-x-0 flex cursor-pointer items-center gap-2 px-3 hover:bg-muted/60',
                  position === highlight && 'bg-primary/15',
                  selectedRow === index && 'outline outline-1 outline-primary',
                )}
                style={{ top: position * DATA_ROW_HEIGHT, height: DATA_ROW_HEIGHT }}
                title={rowDetails(row, runLabels[row.run])}
                onClick={() => {
                  setFollowing(false);
                  onRowClick(index);
                }}
                data-testid="data-row"
                data-row={index}
              >
                <span role="cell" className="w-12 shrink-0 font-mono text-muted-foreground">
                  {formatClockTenths(row.start)}
                </span>
                <span role="cell" className="w-20 shrink-0 truncate">
                  {tableKindOf(row).label}
                </span>
                <span role="cell" className="w-20 shrink-0 truncate text-muted-foreground">
                  {row.track || row.measure}
                </span>
                <span role="cell" className="min-w-0 flex-1 truncate">
                  {text}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
