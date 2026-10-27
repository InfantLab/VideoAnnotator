import { CheckCircle2, Clock, Loader2, Square, XCircle } from 'lucide-react';

import { cn } from '@/lib/utils';
import type { RunState } from '@/lib/runs';

/** State icons and colours (viewer design "Icons", "Colour"): blue running, grey waiting or stopped, green done, red failed. */
const LOOK = {
  running: { icon: Loader2, word: 'Running', className: 'animate-spin text-blue-600 dark:text-blue-400' },
  waiting: { icon: Clock, word: 'Waiting', className: 'text-muted-foreground' },
  done: { icon: CheckCircle2, word: 'Done', className: 'text-emerald-600 dark:text-emerald-400' },
  failed: { icon: XCircle, word: 'Some failed', className: 'text-destructive' },
  stopped: { icon: Square, word: 'Stopped', className: 'text-muted-foreground' },
} as const;

export function RunStateIcon({ state, className }: { state: RunState; className?: string }) {
  const { icon: Icon, word, className: look } = LOOK[state];
  return (
    <span title={word} aria-label={word} role="img" className="inline-flex shrink-0">
      <Icon className={cn('h-4 w-4', look, className)} aria-hidden="true" />
    </span>
  );
}
