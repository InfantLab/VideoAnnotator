import { Button } from '@/components/ui/button';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';

const KEYS: [string, string][] = [
  ['Space', 'Play or pause'],
  ['← →', 'One frame back or on'],
  ['Shift + ← →', 'One second back or on'],
  ['↓ ↑', 'A tenth of a second back or on'],
  ['[ ]', 'Previous or next event in the selected track'],
  ['+ −', 'Zoom the timeline in or out'],
  ['0', 'Show the whole video'],
  ['Ctrl/⌘ + wheel', 'Zoom around the pointer'],
  ['Shift + wheel, drag', 'Move along the timeline'],
  ['Ctrl + Shift + D', 'Debug panel'],
];

/** The keys, behind a `?` button (spec 028 R8). */
export function KeysHelp() {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="ghost" size="icon" className="h-7 w-7 font-semibold" aria-label="Keyboard keys" title="Keyboard keys (?)">
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
      </PopoverContent>
    </Popover>
  );
}
