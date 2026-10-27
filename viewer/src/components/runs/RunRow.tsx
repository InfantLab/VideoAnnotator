import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Download, ExternalLink, MoreHorizontal, Play, RotateCcw, Square, Trash2 } from 'lucide-react';

import { RunDeleteDialog } from '@/components/RunDeleteButton';
import { KindIcons } from '@/components/runs/KindIcons';
import { RunStateIcon } from '@/components/runs/RunStateIcon';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Progress } from '@/components/ui/progress';
import { useRunActions } from '@/hooks/useRunActions';
import { finishedCount, runDuration, runOutcome, runStateOf, videosWord } from '@/lib/runs';
import { hrefOf } from '@/lib/videosScope';
import { formatWhen } from '@/lib/when';
import type { RunListItem } from '@/types/videos';

/** "8 videos: a.mp4, b.mp4 +6", "48 videos in peekaboo", or the one video's name. */
function videosOf(run: RunListItem): string {
  if (run.project) return `${videosWord(run.total)} in ${run.project.name}`;
  if (run.total === 1 && run.videos.length === 1) return run.videos[0];
  const more = run.total - run.videos.length;
  return run.videos.length ? `${videosWord(run.total)}: ${run.videos.join(', ')}${more > 0 ? ` +${more}` : ''}` : videosWord(run.total);
}

function RunActions({ run }: { run: RunListItem }) {
  const navigate = useNavigate();
  const [deleting, setDeleting] = useState(false);
  const { single, runAgain, cancel, retry, download } = useRunActions(run);
  // Dialogs and menus are portals, but React bubbles their clicks to the row.
  return (
    <span className="shrink-0" onClick={(event) => event.stopPropagation()}>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" size="icon" className="h-7 w-7" aria-label={`Actions for ${run.name}`}>
            <MoreHorizontal className="h-4 w-4" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem onSelect={() => navigate(hrefOf({ run: run.id }))}>
            <ExternalLink className="mr-2 h-4 w-4" /> Open
          </DropdownMenuItem>
          {!run.active && (
            <DropdownMenuItem onSelect={runAgain}>
              <Play className="mr-2 h-4 w-4" /> Run again…
            </DropdownMenuItem>
          )}
          {!run.active && run.failed > 0 && (
            <DropdownMenuItem onSelect={() => retry.mutate()}>
              <RotateCcw className="mr-2 h-4 w-4" /> Retry failed videos
            </DropdownMenuItem>
          )}
          {run.active && (
            <DropdownMenuItem onSelect={() => cancel.mutate()}>
              <Square className="mr-2 h-4 w-4" /> Stop
            </DropdownMenuItem>
          )}
          {run.done > 0 && (
            <DropdownMenuItem onSelect={() => download('table')}>
              <Download className="mr-2 h-4 w-4" /> Download table
            </DropdownMenuItem>
          )}
          <DropdownMenuSeparator />
          <DropdownMenuItem onSelect={() => setDeleting(true)} className="text-destructive focus:text-destructive">
            <Trash2 className="mr-2 h-4 w-4" /> Delete…
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <RunDeleteDialog runId={run.id} single={single} videoCount={run.total} name={run.name} after={null} open={deleting} onOpenChange={setDeleting} />
    </span>
  );
}

/**
 * One run (viewer design "Run row"): state · name · videos · what it looks for ·
 * outcome · when · how long, a progress bar only while it runs, one menu.
 */
export function RunRow({ run }: { run: RunListItem }) {
  const navigate = useNavigate();
  const state = runStateOf(run);
  const took = runDuration(run);
  const href = hrefOf({ run: run.id });
  return (
    <li
      className="cursor-pointer space-y-1.5 px-3 py-2.5 hover:bg-muted/50"
      onClick={() => navigate(href)}
      data-testid="run-row"
    >
      <div className="flex min-w-0 items-center gap-2">
        <RunStateIcon state={state} />
        <Link to={href} className="truncate font-medium hover:underline" title={run.name} onClick={(event) => event.stopPropagation()}>
          {run.name}
        </Link>
        <span className="min-w-0 flex-1 truncate text-sm text-muted-foreground" title={videosOf(run)}>
          {videosOf(run)}
        </span>
        <KindIcons pipelines={run.pipelines} />
        <RunActions run={run} />
      </div>
      <div className="flex flex-wrap items-center justify-between gap-x-4 pl-6 text-xs text-muted-foreground">
        <span data-testid="run-outcome">
          {runOutcome(run)}
          {run.failure && (
            <>
              {' '}
              <abbr title={run.failure} className="cursor-help underline decoration-dotted underline-offset-2">
                why?
              </abbr>
            </>
          )}
        </span>
        <span className="tabular-nums">
          {formatWhen(run.started_at)}
          {took && ` · ${took}`}
        </span>
      </div>
      {run.active && (
        <div className="pl-6">
          <Progress value={run.total ? (finishedCount(run) / run.total) * 100 : 0} className="h-1.5" aria-label={`${run.name} progress`} />
        </div>
      )}
    </li>
  );
}

export function RunList({ runs }: { runs: RunListItem[] }) {
  return (
    <ul className="divide-y rounded-md border" data-testid="run-list">
      {runs.map((run) => (
        <RunRow key={run.id} run={run} />
      ))}
    </ul>
  );
}
