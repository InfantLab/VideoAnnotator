import { Layers } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import type { LaneToggles } from '@/components/lanes/LaneHeader';
import type { Lane } from '@/lib/lanes';

interface LayersMenuProps {
  lanes: Lane[];
  toggles: LaneToggles;
  /** Expand (true) or collapse every one of `kinds`. */
  showLanes: (kinds: string[], shown: boolean) => void;
  setOverlays: (keys: string[], on: boolean) => void;
}

/** Every lane with its two switches, and all / none for each (spec 028 US2.3). */
export function LayersMenu({ lanes, toggles, showLanes, setOverlays }: LayersMenuProps) {
  const kinds = [...new Set(lanes.map((l) => l.kind))];
  const overlays = [...new Set(lanes.flatMap((l) => (l.overlay ? [l.overlay] : [])))];
  const allNone = (apply: (on: boolean) => void, what: string) => (
    <span className="flex justify-center gap-1">
      <button type="button" className="underline hover:text-foreground" onClick={() => apply(true)} aria-label={`All ${what}`}>
        all
      </button>
      <button type="button" className="underline hover:text-foreground" onClick={() => apply(false)} aria-label={`No ${what}`}>
        none
      </button>
    </span>
  );
  // Said once when most lanes share a run; a row names its run only where it differs (UAT #31).
  const runCounts = new Map<string, number>();
  lanes.forEach((l) => l.run && runCounts.set(l.run.label, (runCounts.get(l.run.label) ?? 0) + 1));
  const mainRun = [...runCounts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0];
  const grid = 'grid grid-cols-[minmax(0,1fr)_3.75rem_3.75rem] items-center gap-x-1';
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="outline" size="sm" className="h-6 px-2 text-xs">
          <Layers className="mr-1 h-3.5 w-3.5" />
          Layers
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-96 max-w-[calc(100vw-2rem)] p-2 text-xs" align="start" data-testid="layers-menu">
        {lanes.length === 0 && <p className="text-muted-foreground">No lanes yet.</p>}
        {lanes.length > 0 && (
          <div role="table" aria-label="Layers">
            <div role="row" className={`${grid} mb-1 border-b pb-1 text-muted-foreground`}>
              <span role="columnheader" className="truncate">
                {mainRun && <>From {mainRun}</>}
              </span>
              <span role="columnheader" className="text-center">
                Track
              </span>
              <span role="columnheader" className="text-center">
                On video
              </span>
            </div>
            <div role="row" className={`${grid} mb-1 text-muted-foreground`}>
              <span />
              {allNone((on) => showLanes(kinds, on), 'tracks')}
              {allNone((on) => setOverlays(overlays, on), 'on the video')}
            </div>
            {lanes.map((lane) => {
              const otherRun = lane.run && lane.run.label !== mainRun ? lane.run.label : null;
              const full = otherRun ? `${lane.label} · ${otherRun}` : lane.label;
              return (
                <div role="row" key={lane.id} className={`${grid} py-0.5`}>
                  <span role="cell" className="truncate" title={full}>
                    {lane.label}
                    {otherRun && <span className="text-muted-foreground"> · {otherRun}</span>}
                  </span>
                  <span role="cell" className="flex justify-center">
                    <Checkbox
                      checked={!toggles.collapsed(lane.kind)}
                      onCheckedChange={() => toggles.toggleCollapsed(lane.kind)}
                      aria-label={`Show the ${lane.label} track`}
                    />
                  </span>
                  <span role="cell" className="flex justify-center">
                    {lane.overlay && (
                      <Checkbox
                        checked={toggles.overlayOn(lane.overlay)}
                        onCheckedChange={(on) => toggles.setOverlay(lane.overlay!, on === true)}
                        aria-label={`Show ${lane.label} on the video`}
                      />
                    )}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </PopoverContent>
    </Popover>
  );
}
