import { lazy, Suspense, useState, type ReactNode } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { ChevronRight, FileText, GitCompare, Loader2, Play, RotateCcw, Square, Table2, Video } from 'lucide-react';

import { apiClient } from '@/api/client';
import { RunDeleteButton } from '@/components/RunDeleteButton';
import { KindIcons } from '@/components/runs/KindIcons';
import { RunStateIcon } from '@/components/runs/RunStateIcon';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { useRunActions } from '@/hooks/useRunActions';
import { notify } from '@/lib/notify';
import { finishedCount, isVlmRun, runDuration, runOutcome, runStateOf, videosWord } from '@/lib/runs';
import { userMessage } from '@/lib/userMessage';
import { hrefOf } from '@/lib/videosScope';
import type { ExportChoices, RunSummary } from '@/types/videos';
import { formatWhen } from '@/lib/when';

const MethodsDialog = lazy(() => import('@/components/MethodsDialog'));

interface RunHeaderProps {
  run: RunSummary;
  /** A video of the run, for Compare answers. */
  firstVideoPath?: string | null;
  /** From the workflow the run's project uses: what to offer first (spec 029 R7). */
  exportChoices?: ExportChoices | null;
  /** What changed?, Save as workflow (later stories). */
  children?: ReactNode;
}

/** A run's own page (UAT D2): what it ran, how it went, and its results. */
export function RunHeader({ run, firstVideoPath, exportChoices, children }: RunHeaderProps) {
  const navigate = useNavigate();
  const [methodsOpen, setMethodsOpen] = useState(false);
  const { single, scope, table, busy, setBusy, runAgain, cancel, retry, download } = useRunActions(run, exportChoices);
  const finished = finishedCount(run);

  const compare = async () => {
    if (!firstVideoPath) return;
    setBusy('compare');
    try {
      const { runs } = await apiClient.getVideoRuns(firstVideoPath);
      const mine = runs.find((r) => run.job_ids.includes(r.job_id));
      const other = runs.find(
        (r) => r !== mine && r.status === 'completed' && r.pipelines?.some((p) => p.name === 'vlm_annotation'),
      );
      if (mine && other) navigate(`/compare?a=${encodeURIComponent(other.job_id)}&b=${encodeURIComponent(mine.job_id)}`);
      else notify({ title: 'Nothing to compare with yet', body: 'Ask the question on this video again (Run again…), then compare the answers.' });
    } catch (error) {
      notify({ title: "Couldn't find the answers", body: userMessage(error), tone: 'error' });
    } finally {
      setBusy(null);
    }
  };

  const state = runStateOf(run);
  const took = runDuration(run);

  return (
    <section className="space-y-3" data-testid="run-header">
      <nav className="flex items-center gap-1 text-sm text-muted-foreground" aria-label="Breadcrumb">
        <Link to="/runs" className="hover:text-foreground hover:underline">
          Runs
        </Link>
        <ChevronRight className="h-3.5 w-3.5" aria-hidden="true" />
        <span className="truncate text-foreground">{run.name}</span>
      </nav>
      <div className="space-y-3 rounded-lg border p-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0 space-y-1">
            <h1 className="flex min-w-0 items-center gap-2 text-2xl font-semibold">
              <RunStateIcon state={state} className="h-5 w-5" />
              <span className="truncate" title={run.name}>{run.name}</span>
            </h1>
            <p className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm text-muted-foreground">
              <span>
                {videosWord(run.total)}
                {run.project && (
                  <>
                    {' '}in <Link to={hrefOf({ project: run.project.id })} className="text-foreground hover:underline">{run.project.name}</Link>
                  </>
                )}
              </span>
              <span aria-hidden="true">·</span>
              <KindIcons pipelines={run.pipelines} />
              {run.started_at && (
                <>
                  <span aria-hidden="true">·</span>
                  <span>{formatWhen(run.started_at)}</span>
                </>
              )}
              {took && (
                <>
                  <span aria-hidden="true">·</span>
                  <span>{took}</span>
                </>
              )}
              <span aria-hidden="true">·</span>
              <span data-testid="run-state">{runOutcome(run)}</span>
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            {run.active && (
              <Button variant="outline" size="sm" onClick={() => cancel.mutate()} disabled={cancel.isPending}>
                <Square className="mr-1 h-4 w-4" />
                Stop
              </Button>
            )}
            {!run.active && run.failed > 0 && (
              <Button variant="outline" size="sm" onClick={() => retry.mutate()} disabled={retry.isPending}>
                <RotateCcw className="mr-1 h-4 w-4" />
                Retry failed videos
              </Button>
            )}
            {!run.active && (
              <Button variant="outline" size="sm" onClick={runAgain}>
                <Play className="mr-1 h-4 w-4" />
                Run again…
              </Button>
            )}
            {children}
            <RunDeleteButton runId={run.id} single={single} videoCount={run.total} name={run.name} after="/runs" />
          </div>
        </div>

        {run.active && (
          <div className="space-y-1">
            <Progress value={run.total ? (finished / run.total) * 100 : 0} aria-label="Run progress" />
            <p className="text-xs text-muted-foreground">You can close this tab; results will be here.</p>
          </div>
        )}

        {run.done > 0 && (
          <div className="flex flex-wrap items-center gap-2 rounded-md bg-muted/50 p-3" data-testid="run-headline">
            <p className="mr-2 text-sm">
              <span className="font-medium">Results:</span> open a video below to see them on its timeline, or take
              {run.active ? ' those finished so far' : ' all of them'}:
            </p>
            <Button size="sm" onClick={() => download('table')} disabled={busy !== null}>
              {busy === 'table' ? <Loader2 className="mr-1 h-4 w-4 animate-spin" /> : <Table2 className="mr-1 h-4 w-4" />}
              Download table{table === 'parquet' ? ' (Parquet)' : ''}
            </Button>
            {exportChoices?.elan !== false && (
              <Button variant="outline" size="sm" onClick={() => download('elan')} disabled={busy !== null}>
                <Video className="mr-1 h-4 w-4" />
                ELAN
              </Button>
            )}
            {exportChoices?.methods !== false && (
              <Button variant="outline" size="sm" onClick={() => setMethodsOpen(true)}>
                <FileText className="mr-1 h-4 w-4" />
                Methods
              </Button>
            )}
            {isVlmRun(run) && (
              <Button variant="outline" size="sm" onClick={compare} disabled={busy !== null || !firstVideoPath}>
                <GitCompare className="mr-1 h-4 w-4" />
                Compare answers
              </Button>
            )}
          </div>
        )}
      </div>
      {methodsOpen && (
        <Suspense fallback={null}>
          <MethodsDialog scope={scope} open={methodsOpen} onOpenChange={setMethodsOpen} />
        </Suspense>
      )}
    </section>
  );
}
