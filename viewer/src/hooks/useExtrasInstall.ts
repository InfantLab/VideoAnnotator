import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useMutation, useQueries } from '@tanstack/react-query';
import { apiClient } from '@/api/client';
import { APIError } from '@/api/handleError';
import { useRefreshPipelineCatalog } from '@/hooks/usePipelineCatalog';
import type { ExtrasInstallJob, PipelineDescriptor } from '@/types/pipelines';

const STORAGE_KEY = 'videoannotator_extras_install_jobs';
const POLL_INTERVAL_MS = 5000;

interface TrackedJob {
  jobId: string;
  startedAt: string;
  /** Pipeline ids known to belong to this extras group at trigger time, used to clear the tracked job once they're all observed available again (see data-model.md). */
  pipelineIds: string[];
}

type TrackedJobsMap = Record<string, TrackedJob>; // extraName -> job

function readTrackedJobs(): TrackedJobsMap {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as unknown;
    return parsed && typeof parsed === 'object' ? (parsed as TrackedJobsMap) : {};
  } catch {
    return {};
  }
}

function writeTrackedJobs(map: TrackedJobsMap) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(map));
  } catch {
    // Best-effort only - the install still runs server-side even if this browser
    // can't persist tracking (private browsing, storage quota, etc).
  }
}

function isTerminal(status: ExtrasInstallJob['status'] | undefined): boolean {
  return status === 'completed' || status === 'failed';
}

/**
 * Tracks pipeline-extras install jobs the user has triggered in this browser:
 * persists job identity across reloads, polls status while in flight, and stops
 * polling once a job reaches a terminal state. See specs/002-pipeline-extras-install.
 *
 * @param catalogPipelines Current pipeline catalog (if available). Used only to
 * auto-drop a terminal (completed/failed) tracked job once every pipeline it was
 * meant to unlock is observed `available !== false` again - i.e. once the server
 * has restarted and the group is confirmed active - per data-model.md's tracked-job
 * lifecycle. Rendering itself doesn't depend on this: a pipeline that's `available`
 * already renders via the normal selectable path, not LockedPipelineCard, regardless
 * of leftover tracking state.
 */
export function useExtrasInstall(catalogPipelines: PipelineDescriptor[] = []) {
  const [trackedJobs, setTrackedJobs] = useState<TrackedJobsMap>(() => readTrackedJobs());
  const extraNames = useMemo(() => Object.keys(trackedJobs), [trackedJobs]);

  const jobQueries = useQueries({
    queries: extraNames.map((extraName) => {
      const jobId = trackedJobs[extraName].jobId;
      return {
        queryKey: ['videoannotator', 'extras-install-job', jobId] as const,
        queryFn: async (): Promise<ExtrasInstallJob | null> => {
          try {
            return await apiClient.getExtrasInstallJob(jobId);
          } catch (error) {
            // A 404 means the server no longer knows this job (expired, or server
            // restarted and lost in-memory state) - treat as terminal-failed locally
            // rather than retrying indefinitely.
            if (error instanceof APIError && error.status === 404) {
              return null;
            }
            throw error;
          }
        },
        refetchInterval: (query: { state: { data?: ExtrasInstallJob | null } }) => {
          const data = query.state.data;
          if (data === null) return false;
          return isTerminal(data?.status) ? false : POLL_INTERVAL_MS;
        },
        staleTime: 0
      };
    })
  });

  const jobsByExtra = useMemo(() => {
    const map: Record<string, ExtrasInstallJob | null | undefined> = {};
    extraNames.forEach((extraName, index) => {
      map[extraName] = jobQueries[index]?.data;
    });
    return map;
  }, [extraNames, jobQueries]);

  const forgetJob = useCallback((extraName: string) => {
    setTrackedJobs((prev) => {
      if (!(extraName in prev)) return prev;
      const next = { ...prev };
      delete next[extraName];
      writeTrackedJobs(next);
      return next;
    });
  }, []);

  // A stale/expired job (server returned 404) can never progress - drop it so
  // the pipeline goes back to a plain "locked, installable" state.
  useEffect(() => {
    extraNames.forEach((extraName) => {
      if (jobsByExtra[extraName] === null) {
        forgetJob(extraName);
      }
    });
  }, [extraNames, jobsByExtra, forgetJob]);

  // A completed install changes what the server reports (top-level
  // `restart_required`, and later `available`), but nothing else re-fetches the
  // catalog, so the restart banner never appeared. Refresh once per completed job,
  // bypassing the API client's own catalog cache.
  const refreshCatalog = useRefreshPipelineCatalog();
  const refreshedJobIds = useRef(new Set<string>());
  useEffect(() => {
    extraNames.forEach((extraName) => {
      const job = jobsByExtra[extraName];
      if (job?.status !== 'completed' || refreshedJobIds.current.has(job.jobId)) return;
      refreshedJobIds.current.add(job.jobId);
      refreshCatalog({ forceServerRefresh: true }).catch(() => {});
    });
  }, [extraNames, jobsByExtra, refreshCatalog]);

  // Once every pipeline a terminal job was tracking is confirmed available again
  // (server restarted), drop the local tracking entry - it's served its purpose.
  useEffect(() => {
    if (!catalogPipelines.length) return;

    extraNames.forEach((extraName) => {
      const job = jobsByExtra[extraName];
      if (!job || !isTerminal(job.status)) return;

      const pipelineIds = trackedJobs[extraName]?.pipelineIds ?? [];
      if (pipelineIds.length === 0) return;

      const allUnlocked = pipelineIds.every((id) => {
        const pipeline = catalogPipelines.find((p) => p.id === id);
        return pipeline ? pipeline.available !== false : false;
      });

      if (allUnlocked) {
        forgetJob(extraName);
      }
    });
    // trackedJobs is read but intentionally excluded: forgetJob already depends on
    // the latest state via its updater function, and including trackedJobs here
    // would re-run this effect on every job-tracking change rather than only when
    // the catalog itself changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [catalogPipelines, extraNames, jobsByExtra, forgetJob]);

  const triggerMutation = useMutation({
    mutationFn: (variables: { extraName: string; pipelineIds: string[] }) =>
      apiClient.installPipelineExtras(variables.extraName),
    onSuccess: (response, variables) => {
      setTrackedJobs((prev) => {
        const next: TrackedJobsMap = {
          ...prev,
          [variables.extraName]: {
            jobId: response.jobId,
            startedAt: new Date().toISOString(),
            pipelineIds: variables.pipelineIds
          }
        };
        writeTrackedJobs(next);
        return next;
      });
    }
  });

  const install = useCallback(
    (extraName: string, pipelineIds: string[]) =>
      triggerMutation.mutateAsync({ extraName, pipelineIds }),
    [triggerMutation]
  );

  /** Drop a tracked job's local state once its pipelines are confirmed available again, or the user dismisses it. */
  const clearTrackedJob = useCallback((extraName: string) => forgetJob(extraName), [forgetJob]);

  return {
    /** extraName -> job status (undefined while the first poll is in flight; never null here - 404s are dropped automatically). */
    jobsByExtra,
    /** Trigger an install for `extraName`. `pipelineIds` are the currently-locked pipelines this extras group unlocks. */
    install,
    isInstalling: (extraName: string) =>
      triggerMutation.isPending && triggerMutation.variables?.extraName === extraName,
    installError: triggerMutation.error,
    installErrorExtraName: triggerMutation.isError ? triggerMutation.variables?.extraName : undefined,
    /** Pipeline ids tracked for a given extras group's most recent install job. */
    trackedPipelineIds: (extraName: string) => trackedJobs[extraName]?.pipelineIds ?? [],
    clearTrackedJob
  };
}
