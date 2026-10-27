import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useMutation, useQueryClient } from '@tanstack/react-query';

import { apiClient, type ExportScope } from '@/api/client';
import { saveBlob } from '@/lib/exports';
import { notify } from '@/lib/notify';
import { isSingle } from '@/lib/runs';
import { runState } from '@/lib/runTarget';
import { userMessage } from '@/lib/userMessage';
import { hrefOf } from '@/lib/videosScope';
import type { ExportChoices, RunSummary } from '@/types/videos';

/** What can be done to a run, from its page or its row's menu: stop, retry the failed, run again, export. */
export function useRunActions(run: RunSummary, exportChoices?: ExportChoices | null) {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [busy, setBusy] = useState<string | null>(null);
  const single = isSingle(run);
  const scope: ExportScope = { kind: 'run', id: run.id };
  const table = exportChoices?.table ?? 'csv';
  const refresh = () => queryClient.invalidateQueries({ queryKey: ['videos'] });

  const runAgain = () =>
    navigate('/run', {
      state: runState({ target: { kind: 'rerun', runId: run.id, label: run.name, single, videoCount: run.total } }),
    });

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

  return { single, scope, table, busy, setBusy, runAgain, cancel, retry, download };
}
