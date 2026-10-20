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
  const allNone = (label: string, apply: (on: boolean) => void) => (
    <span className="flex items-center gap-1 text-muted-foreground">
      {label}
      <button type="button" className="underline hover:text-foreground" onClick={() => apply(true)}>
        all
      </button>
      <button type="button" className="underline hover:text-foreground" onClick={() => apply(false)}>
        none
      </button>
    </span>
  );
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="outline" size="sm" className="h-6 px-2 text-xs">
          <Layers className="mr-1 h-3.5 w-3.5" />
          Layers
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-80 p-2 text-xs" align="start" data-testid="layers-menu">
        <div className="mb-2 flex justify-between">
          {allNone('Tracks', (on) => showLanes(kinds, on))}
          {allNone('On the video', (on) => setOverlays(overlays, on))}
        </div>
        {lanes.length === 0 && <p className="text-muted-foreground">No lanes yet.</p>}
        <table className="w-full">
          <thead className="sr-only">
            <tr>
              <th>Track</th>
              <th>Shown</th>
              <th>On the video</th>
            </tr>
          </thead>
          <tbody>
            {lanes.map((lane) => (
              <tr key={lane.id}>
                <td className="truncate py-0.5 pr-2">
                  {lane.label}
                  {lane.run && <span className="text-muted-foreground"> · {lane.run.label}</span>}
                </td>
                <td className="w-8">
                  <Checkbox
                    checked={!toggles.collapsed(lane.kind)}
                    onCheckedChange={() => toggles.toggleCollapsed(lane.kind)}
                    aria-label={`Show the ${lane.label} track`}
                  />
                </td>
                <td className="w-8">
                  {lane.overlay && (
                    <Checkbox
                      checked={toggles.overlayOn(lane.overlay)}
                      onCheckedChange={(on) => toggles.setOverlay(lane.overlay!, on === true)}
                      aria-label={`Show ${lane.label} on the video`}
                    />
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </PopoverContent>
    </Popover>
  );
}
