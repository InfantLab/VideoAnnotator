import { lazy, Suspense, useState, type ReactNode } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { FileText, GitCompare, Loader2, Play, RotateCcw, Square, Table2, Video } from 'lucide-react';

import { apiClient, type ExportScope } from '@/api/client';
import { RunDeleteButton } from '@/components/RunDeleteButton';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { saveBlob } from '@/lib/exports';
import { notify } from '@/lib/notify';
import { finishedCount, isSingle, isVlmRun, runStatus, videosWord } from '@/lib/runs';
import { runState } from '@/lib/runTarget';
import { userMessage } from '@/lib/userMessage';
import { hrefOf } from '@/lib/videosScope';
import type { ExportChoices, RunSummary } from '@/types/videos';

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

/** A run as a filter of Videos (spec 029 R12): its progress, actions and, once finished, its results table. */
export function RunHeader({ run, firstVideoPath, exportChoices, children }: RunHeaderProps) {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [methodsOpen, setMethodsOpen] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const single = isSingle(run);
  const finished = finishedCount(run);
  const scope: ExportScope = { kind: 'run', id: run.id };
  const table = exportChoices?.table ?? 'csv';

  const refresh = () => {
    queryClient.invalidateQueries({ queryKey: ['videos'] });
  };
  const cancel = useMutation({
    mutationFn: async (): Promise<void> => {
      if (single) await apiClient.cancelJob(run.id);
      else await apiClient.cancelBatch(run.id);
    },
    onSuccess: () => {
      refresh();
      notify({ title: `Stopping ${run.name}`, body: 'Videos already finished keep their results.', action: { label: 'Run again…', onClick: () => runAgain() } });
    },
    onError: (error) => notify({ title: "Couldn't stop the run", body: userMessage(error), tone: 'error' }),
  });
  const retry = useMutation({
    mutationFn: async (): Promise<void> => {
      if (single) await apiClient.retryJob(run.id);
      else await apiClient.retryBatch(run.id);
    },
    onSuccess: () => {
      refresh();
      notify({ title: `Running the failed ${run.failed === 1 ? 'video' : 'videos'} again`, action: { label: 'Show run', to: hrefOf({ run: run.id }) } });
    },
    onError: (error) => notify({ title: "Couldn't run them again", body: userMessage(error), tone: 'error' }),
  });

  const runAgain = () =>
    navigate('/run', {
      state: runState({ target: { kind: 'rerun', runId: run.id, label: run.name, single, videoCount: run.total } }),
    });

  const download = async (kind: 'table' | 'elan') => {
    setBusy(kind);
    try {
      const file = await apiClient.exportResults(scope, kind, table);
      saveBlob(file.blob, file.filename ?? (kind === 'table' ? `${run.name}.${table}` : `${run.name}.eaf`));
    } catch (error) {
      notify({ title: "Couldn't export", body: userMessage(error), tone: 'error' });
    } finally {
      setBusy(null);
    }
  };

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

  return (
    <section className="space-y-3 rounded-lg border p-4" data-testid="run-header">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h1 className="truncate text-2xl font-semibold">{run.name}</h1>
          <p className="text-sm text-muted-foreground">
            Run of {videosWord(run.total)}
            {run.project && (
              <>
                {' '}in <Link to={hrefOf({ project: run.project.id })} className="text-foreground hover:underline">{run.project.name}</Link>
              </>
            )}
            {run.started_at && <> · started {new Date(run.started_at).toLocaleString()}</>}
            {' · '}
            <span data-testid="run-state">{runStatus(run)}</span>
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {run.active && (
            <Button variant="outline" size="sm" onClick={() => cancel.mutate()} disabled={cancel.isPending}>
              <Square className="mr-1 h-4 w-4" />
              Cancel
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
          <RunDeleteButton runId={run.id} single={single} videoCount={run.total} name={run.name} />
        </div>
      </div>

      {run.active && (
        <div className="space-y-1">
          <Progress value={run.total ? (finished / run.total) * 100 : 0} aria-label="Run progress" />
          <p className="text-xs text-muted-foreground">
            {finished} of {run.total} finished · {run.running} running · {run.waiting} waiting
            {run.failed > 0 && ` · ${run.failed} failed`}. You can close this tab; results will be here.
          </p>
        </div>
      )}

      {!run.active && run.done > 0 && (
        <div className="flex flex-wrap items-center gap-2 rounded-md bg-muted/50 p-3" data-testid="run-headline">
          <p className="mr-2 font-medium">
            {videosWord(run.done)} → results table
          </p>
          <Button size="sm" onClick={() => download('table')} disabled={busy !== null}>
            {busy === 'table' ? <Loader2 className="mr-1 h-4 w-4 animate-spin" /> : <Table2 className="mr-1 h-4 w-4" />}
            Download{table === 'parquet' ? ' (Parquet)' : ''}
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
      {methodsOpen && (
        <Suspense fallback={null}>
          <MethodsDialog scope={scope} open={methodsOpen} onOpenChange={setMethodsOpen} />
        </Suspense>
      )}
    </section>
  );
}
