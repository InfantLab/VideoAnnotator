import { CheckCircle2, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useCurrentUser } from '@/hooks/useCurrentUser';
import { useServerRestart } from '@/hooks/useServerRestart';

/**
 * A pipeline card's "installed, restart to use it", with the restart right
 * there: the step's RestartRequiredBanner can be scrolled out of view. Shares
 * the banner's restart, so both show the same progress.
 */
export const RestartToActivate = () => {
  const { isAdmin } = useCurrentUser();
  const { state, restart } = useServerRestart();

  if (state.phase === 'requesting' || state.phase === 'restarting') {
    return (
      <p className="mt-1 flex items-center gap-1.5 text-xs font-medium text-muted-foreground" role="status">
        <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
        Restarting VideoAnnotator&hellip;
      </p>
    );
  }

  return (
    <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1">
      <p className="flex items-center gap-1.5 text-xs font-medium text-amber-700 dark:text-amber-500">
        <CheckCircle2 className="h-3.5 w-3.5" aria-hidden="true" />
        {isAdmin === false
          ? 'Installed. Ask whoever runs VideoAnnotator to restart it.'
          : 'Installed. Restart VideoAnnotator to use it.'}
      </p>
      {isAdmin !== false && (
        <Button type="button" size="sm" variant="secondary" className="h-7 px-2 text-xs" onClick={() => restart(false)}>
          Restart now
        </Button>
      )}
    </div>
  );
};
