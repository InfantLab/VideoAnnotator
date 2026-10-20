import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { Download, GitCompare, Square } from 'lucide-react';

import { apiClient } from '@/api/client';
import { ResultsLocation } from '@/components/ResultsLocation';
import { RunAgainActions } from '@/components/RunAgainActions';
import { RunDeleteButton } from '@/components/RunDeleteButton';
import { SaveAsWorkflow } from '@/components/workflows/SaveAsWorkflow';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { usePipelineTitles } from '@/hooks/usePipelineTitles';
import { saveBlob } from '@/lib/exports';
import { notify } from '@/lib/notify';
import { userMessage } from '@/lib/userMessage';
import { hrefOf } from '@/lib/videosScope';
import type { FolderRef } from '@/types/ingest';
import type { VideoRun } from '@/types/video';

const WORDS: Record<string, { word: string; className: string }> = {
  completed: { word: 'done', className: 'border-emerald-300 text-emerald-700 dark:text-emerald-400' },
  running: { word: 'running', className: 'border-blue-300 text-blue-700 dark:text-blue-400' },
  pending: { word: 'waiting', className: 'border-muted-foreground/40 text-muted-foreground' },
  retrying: { word: 'waiting', className: 'border-muted-foreground/40 text-muted-foreground' },
  failed: { word: 'failed', className: 'border-destructive/50 text-destructive' },
  cancelled: { word: 'stopped', className: 'border-muted-foreground/40 text-muted-foreground' },
};

export function RunStatus({ status }: { status: string }) {
  const look = WORDS[status] ?? WORDS.pending;
  return (
    <Badge variant="outline" className={look.className}>
      {look.word}
    </Badge>
  );
}

const ordinal = (n: number) => `${n}${n % 10 === 1 && n % 100 !== 11 ? 'st' : n % 10 === 2 && n % 100 !== 12 ? 'nd' : n % 10 === 3 && n % 100 !== 13 ? 'rd' : 'th'}`;

const runLabel = (run: VideoRun) =>
  run.run_name || run.batch_name || (run.created_at ? new Date(run.created_at).toLocaleString() : run.job_id);

function settingsLines(config: Record<string, unknown> | undefined, titleOf: (id: string) => string): string[] {
  const lines: string[] = [];
  for (const [pipeline, values] of Object.entries(config ?? {})) {
    if (!values || typeof values !== 'object') continue;
    const set = Object.entries(values as Record<string, unknown>).map(
      ([key, value]) => `${key}: ${typeof value === 'string' ? value : JSON.stringify(value)}`,
    );
    if (set.length) lines.push(`${titleOf(pipeline)} · ${set.join(', ')}`);
  }
  return lines;
}

function CompareWith({ run, runs }: { run: VideoRun; runs: VideoRun[] }) {
  const navigate = useNavigate();
  const [other, setOther] = useState('');
  const isVlm = (r: VideoRun) => r.status === 'completed' && (r.pipelines ?? []).some((p) => p.name === 'vlm_annotation');
  if (!isVlm(run)) return null;
  const candidates = runs.filter((r) => r.job_id !== run.job_id && isVlm(r));
  if (!candidates.length) return null;
  return (
    <div className="flex flex-wrap items-center gap-2">
      <select aria-label="Answers to compare with" value={other} onChange={(e) => setOther(e.target.value)} className="h-8 rounded-md border border-input bg-background px-2 text-xs">
        <option value="">Compare answers with…</option>
        {candidates.map((c) => (
          <option key={c.job_id} value={c.job_id}>
            {runLabel(c)}
          </option>
        ))}
      </select>
      <Button size="sm" variant="outline" disabled={!other} onClick={() => navigate(`/compare?a=${encodeURIComponent(other)}&b=${encodeURIComponent(run.job_id)}`)}>
        <GitCompare className="mr-1 h-4 w-4" />
        Compare
      </Button>
    </div>
  );
}

function RunCard({ run, runs, videoName }: { run: VideoRun; runs: VideoRun[]; videoName: string }) {
  const queryClient = useQueryClient();
  const { titleOf } = usePipelineTitles();
  const [downloading, setDownloading] = useState(false);
  const going = run.status === 'pending' || run.status === 'running' || run.status === 'retrying';
  const cancel = useMutation({
    mutationFn: () => apiClient.cancelJob(run.job_id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['videoRuns'] });
      queryClient.invalidateQueries({ queryKey: ['videos'] });
      notify({ title: `Stopping the run on ${videoName}`, body: 'Pipelines already finished keep their results.' });
    },
    onError: (error) => notify({ title: "Couldn't stop it", body: userMessage(error), tone: 'error' }),
  });
  const downloadAll = async () => {
    setDownloading(true);
    try {
      const response = await apiClient.getJobArtifacts(run.job_id);
      saveBlob(await response.blob(), `${videoName.replace(/\.[^.]+$/, '')} results.zip`);
    } catch (error) {
      notify({ title: "Couldn't download the files", body: userMessage(error), tone: 'error' });
    } finally {
      setDownloading(false);
    }
  };
  const settings = settingsLines(run.config, titleOf);
  const folder = run.results_folder as unknown as (FolderRef & { exists?: boolean }) | null | undefined;

  return (
    <li className="space-y-2 rounded-md border p-3" data-testid="run-card">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="font-medium">{runLabel(run)}</span>
        <RunStatus status={run.status} />
      </div>
      <p className="text-muted-foreground">
        {run.created_at && <>Started {new Date(run.created_at).toLocaleString()}</>}
        {run.completed_at && <> · finished {new Date(run.completed_at).toLocaleString()}</>}
        {run.batch_id && (
          <>
            {' '}· part of{' '}
            <Link to={hrefOf({ run: run.batch_id })} className="text-foreground hover:underline">
              {run.batch_name || run.run_name || 'its run'}
            </Link>
          </>
        )}
      </p>
      {going && (
        <div className="space-y-1">
          <Progress value={run.progress} aria-label="Progress" />
          {run.queue_position ? <p>Waiting its turn: {ordinal(run.queue_position)} in line.</p> : null}
        </div>
      )}
      {run.error && <p className="text-destructive">{run.error}</p>}
      <ul className="space-y-0.5">
        {(run.pipelines ?? []).map((p) => (
          <li key={p.name} className="flex flex-wrap items-baseline gap-2">
            <span>{titleOf(p.name)}</span>
            <RunStatus status={p.status} />
            {p.error && <span className="text-destructive">{p.error}</span>}
          </li>
        ))}
        {!(run.pipelines ?? []).length &&
          (run.selected_pipelines ?? []).map((name) => (
            <li key={name} className="flex gap-2">
              <span>{titleOf(name)}</span>
              <RunStatus status={run.status} />
            </li>
          ))}
      </ul>
      {settings.length > 0 && (
        <details>
          <summary className="cursor-pointer text-muted-foreground">Settings</summary>
          <ul className="mt-1 space-y-0.5 font-mono text-[11px]">
            {settings.map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
        </details>
      )}
      {folder?.path && <ResultsLocation folder={folder} label="This video's results" />}
      <div className="flex flex-wrap gap-2">
        {going ? (
          <Button size="sm" variant="outline" onClick={() => cancel.mutate()} disabled={cancel.isPending}>
            <Square className="mr-1 h-4 w-4" />
            Cancel
          </Button>
        ) : (
          <>
            <Button size="sm" variant="outline" onClick={downloadAll} disabled={downloading}>
              <Download className="mr-1 h-4 w-4" />
              Download all files
            </Button>
            <RunDeleteButton runId={run.job_id} single videoCount={1} name={`the run on ${videoName}`} after={null} />
          </>
        )}
      </div>
      {!going && <RunAgainActions target={{ kind: 'rerun', runId: run.job_id, label: videoName, single: true }} />}
      {!going && (
        <SaveAsWorkflow
          source={{ kind: 'run', id: run.job_id, name: runLabel(run) }}
          settings={async () => ({ selected: run.selected_pipelines ?? [], config: (run.config ?? {}) as Record<string, unknown> })}
        />
      )}
      <CompareWith run={run} runs={runs} />
    </li>
  );
}

/**
 * Every run on this video, newest first, with what Job detail used to show
 * (spec 029 R12): status, progress and place in the queue, each pipeline's
 * outcome, its settings, where its results are, and what can be done to it.
 */
export function RunsPanel({ runs, videoName }: { runs: VideoRun[]; videoName: string }) {
  if (!runs.length) return <p className="text-muted-foreground">No runs on this video yet.</p>;
  return (
    <ul className="space-y-3" data-testid="runs-panel">
      {runs.map((run) => (
        <RunCard key={run.job_id} run={run} runs={runs} videoName={videoName} />
      ))}
    </ul>
  );
}
