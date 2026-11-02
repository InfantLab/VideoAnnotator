import { useState } from 'react';

import { Button } from '@/components/ui/button';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';

const KEYS: [string, string][] = [
  ['Space', 'Play or pause'],
  ['← →', 'One frame back or on'],
  ['Shift + ← →', 'One second back or on'],
  ['↓ ↑', 'A tenth of a second back or on'],
  ['[ ]', 'Previous or next event in the selected track'],
  ['N', 'A note at this moment; Shift + N twice for a span'],
  ['+ −', 'Zoom the timeline in or out'],
  ['0', 'Show the whole video'],
  ['Ctrl/⌘ + wheel', 'Zoom around the pointer'],
  ['Shift + wheel, drag', 'Move along the timeline'],
  ['Ctrl + Shift + D', 'Debug panel'],
];

/**
 * Help on the Video page, behind its `?` button: the keys (spec 028 R8), the
 * tips again, and reporting a problem (UAT #68, #70).
 */
export function KeysHelp({ onTips, onReport }: { onTips?: () => void; onReport?: () => void }) {
  const [open, setOpen] = useState(false);
  const then = (action: () => void) => () => {
    setOpen(false);
    action();
  };
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button variant="ghost" size="icon" className="h-7 w-7 font-semibold" aria-label="Help and keys" title="Help and keys">
          ?
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-80 text-xs" align="end">
        <p className="mb-2 font-medium">Keys</p>
        <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
          {KEYS.map(([key, what]) => (
            <div key={key} className="contents">
              <dt className="font-mono text-muted-foreground">{key}</dt>
              <dd>{what}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-2 text-muted-foreground">Keys do nothing while you type in a box.</p>
        {(onTips || onReport) && (
          <div className="mt-3 flex flex-wrap gap-2 border-t pt-3">
            {onTips && (
              <Button variant="outline" size="sm" className="h-7 text-xs" onClick={then(onTips)}>
                Show the tips again
              </Button>
            )}
            {onReport && (
              <Button variant="outline" size="sm" className="h-7 text-xs" onClick={then(onReport)}>
                Report a problem
              </Button>
            )}
          </div>
        )}
      </PopoverContent>
    </Popover>
  );
}
