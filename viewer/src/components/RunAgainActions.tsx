import { Pencil, RotateCcw } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { useRunAgain } from '@/hooks/useRunAgain';
import type { RunTarget } from '@/lib/runTarget';

type Rerun = Extract<RunTarget, { kind: 'rerun' }>;

/**
 * The next step after a result (spec 019, spec 029): run it again with its
 * settings to change in New run (the main choice), or unchanged, on the same
 * videos. A run of several videos first checks which are no longer where they
 * were, and offers to look for them.
 */
export const RunAgainActions = ({ target, size = 'sm' }: { target: Rerun; size?: 'sm' | 'default' }) => {
  const { edit, startUnchanged, busy, ui } = useRunAgain(target);
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-2">
        <Button size={size} onClick={edit}>
          <Pencil className="h-4 w-4 mr-2" /> Run again…
        </Button>
        <Button size={size} variant="outline" onClick={startUnchanged} disabled={busy}>
          <RotateCcw className="h-4 w-4 mr-2" /> Run again unchanged
        </Button>
      </div>
      {ui}
    </div>
  );
};
