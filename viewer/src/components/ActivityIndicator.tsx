import { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Activity as ActivityIcon, CheckCircle2 } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { useActivity } from '@/hooks/useActivity';
import { finishedCount, videosWord } from '@/lib/runs';
import { notify } from '@/lib/notify';
import { progressLine, type Transition } from '@/lib/activity';
import { hrefOf } from '@/lib/videosScope';
import type { ActivityRun } from '@/types/videos';

function announce({ kind, run }: Transition) {
  const show = { label: 'Show run', to: hrefOf({ run: run.id }) };
  if (kind === 'started') notify({ title: `${run.name} started`, body: `${videosWord(run.total)}. Results fill in as each finishes.`, action: show });
  else if (kind === 'finished')
    notify({
      title: `${run.name} finished`,
      body: run.failed ? `${videosWord(run.done)} done, ${run.failed} failed.` : `${videosWord(run.done)} done.`,
      action: { label: 'Results table', to: hrefOf({ run: run.id }) },
      tone: 'success',
    });
  else if (kind === 'failed') notify({ title: `${run.name} failed`, body: 'None of its videos finished. The run shows why.', action: show, tone: 'error' });
  else notify({ title: `${run.name} stopped`, body: 'Videos already finished keep their results.', action: show });
}

function RunLine({ run, active }: { run: ActivityRun; active: boolean }) {
  return (
    <li>
      <Link to={hrefOf({ run: run.id })} className="block rounded px-2 py-1.5 hover:bg-muted">
        <span className="flex items-center justify-between gap-2">
          <span className="truncate font-medium">{run.name}</span>
          <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
            {finishedCount(run)}/{run.total}
          </span>
        </span>
        {active ? (
          <span className="mt-1 block h-1 overflow-hidden rounded bg-muted">
            <span className="block h-full bg-primary" style={{ width: `${run.total ? (finishedCount(run) / run.total) * 100 : 0}%` }} />
          </span>
        ) : (
          <span className="text-xs text-muted-foreground">
            {run.stopped ? 'stopped' : run.failed ? `finished, ${run.failed} failed` : 'finished'}
          </span>
        )}
        {run.project && <span className="block text-xs text-muted-foreground">{run.project.name}</span>}
      </Link>
    </li>
  );
}

/** "2 runs · 31/48" in the nav, with the runs going on and just finished (spec 029 R17, FR-018). */
export function ActivityIndicator() {
  const { active, recent, transitions, error } = useActivity();

  useEffect(() => {
    transitions.forEach(announce);
  }, [transitions]);

  if (error) return null;
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="ghost" size="sm" className="gap-2" aria-label={active.length ? `Runs going on: ${progressLine(active)}` : 'Runs'} data-testid="activity-indicator">
          {active.length ? <ActivityIcon className="h-4 w-4 animate-pulse text-blue-600" /> : <CheckCircle2 className="h-4 w-4 text-muted-foreground" />}
          {active.length > 0 && <span className="hidden text-xs tabular-nums md:inline">{progressLine(active)}</span>}
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-80 p-2 text-sm" align="end">
        {active.length === 0 && recent.length === 0 && <p className="p-2 text-muted-foreground">No runs going on.</p>}
        {active.length > 0 && (
          <>
            <p className="px-2 pb-1 text-xs font-semibold uppercase text-muted-foreground">Going on</p>
            <ul>{active.map((run) => <RunLine key={run.id} run={run} active />)}</ul>
          </>
        )}
        {recent.length > 0 && (
          <>
            <p className="px-2 pb-1 pt-2 text-xs font-semibold uppercase text-muted-foreground">Finished today</p>
            <ul>{recent.map((run) => <RunLine key={run.id} run={run} active={false} />)}</ul>
          </>
        )}
      </PopoverContent>
    </Popover>
  );
}
