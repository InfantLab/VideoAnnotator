import { useCallback, useSyncExternalStore } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/api/client';
import { APIError, apiErrorEnvelope } from '@/api/handleError';
import { useIngestAccess } from '@/hooks/useIngestAccess';
import { useRefreshPipelineCatalog } from '@/hooks/usePipelineCatalog';

const POLL_INTERVAL_MS = 2000;
const GIVE_UP_AFTER_MS = 120_000;

export const MANUAL_RESTART_HINT =
  'Restart it the way it was started, e.g. stop it and run `videoannotator server` again, then refresh this page.';

export const LAUNCHER_RESTART_HINT = 'Run `videoannotator-start restart`, then refresh this page.';

export type ServerRestartState =
  | { phase: 'idle' }
  | { phase: 'requesting' }
  | { phase: 'restarting' }
  | { phase: 'confirm_force'; runningJobCount: number }
  | { phase: 'failed'; message: string; hint?: string }
  | { phase: 'timed_out' }
  | { phase: 'done' };

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

// One restart per page: the banner and each pipeline card offer it, and all of
// them show the same progress. `generation` ends a superseded wait.
let shared: ServerRestartState = { phase: 'idle' };
let generation = 0;
const listeners = new Set<() => void>();

const subscribe = (listener: () => void) => {
  listeners.add(listener);
  return () => listeners.delete(listener);
};

const setShared = (next: ServerRestartState) => {
  shared = next;
  listeners.forEach((listener) => listener());
};

export function resetServerRestart() {
  generation += 1;
  setShared({ phase: 'idle' });
}

/**
 * Restart the VideoAnnotator server from the viewer (VideoAnnotator spec 011,
 * contracts/readiness-contract.md sections 4-5): ask for a restart, then poll
 * /health until `boot_id` changes. Connection errors while polling are expected,
 * because the server is down. On success, everything cached is refetched, since the
 * pipeline catalog (at least) has changed.
 */
export function useServerRestart() {
  const state = useSyncExternalStore(subscribe, () => shared);
  const queryClient = useQueryClient();
  const refreshCatalog = useRefreshPipelineCatalog();
  const { access } = useIngestAccess();
  const manualHint = access?.managed_by_launcher ? LAUNCHER_RESTART_HINT : MANUAL_RESTART_HINT;

  const waitForNewBoot = useCallback(
    async (previousBootId: string, run: number) => {
      const deadline = Date.now() + GIVE_UP_AFTER_MS;
      while (Date.now() < deadline) {
        await sleep(POLL_INTERVAL_MS);
        if (run !== generation) return;
        try {
          const { bootId } = await apiClient.getBootIdentity();
          if (bootId && bootId !== previousBootId) {
            await refreshCatalog({ forceServerRefresh: true }).catch(() => {});
            await queryClient.invalidateQueries();
            if (run === generation) setShared({ phase: 'done' });
            return;
          }
        } catch {
          // Still restarting.
        }
      }
      if (run === generation) setShared({ phase: 'timed_out' });
    },
    [queryClient, refreshCatalog]
  );

  const restart = useCallback(
    async (force = false) => {
      if (shared.phase === 'requesting' || shared.phase === 'restarting') return;
      generation += 1;
      const run = generation;
      const update = (next: ServerRestartState) => {
        if (run === generation) setShared(next);
      };
      update({ phase: 'requesting' });
      try {
        const { bootId } = await apiClient.restartServer(force);
        update({ phase: 'restarting' });
        await waitForNewBoot(bootId, run);
      } catch (error) {
        const { code, hint, details } = apiErrorEnvelope(error);
        if (code === 'JOBS_RUNNING') {
          const jobIds = Array.isArray(details?.job_ids) ? details.job_ids : [];
          update({ phase: 'confirm_force', runningJobCount: jobIds.length });
        } else if (code === 'INSTALL_IN_PROGRESS') {
          update({
            phase: 'failed',
            message: 'A pipeline install is still running.',
            hint: 'Wait for it to finish, then restart.'
          });
        } else if (code === 'RESTART_UNSUPPORTED') {
          update({
            phase: 'failed',
            message: "VideoAnnotator can't restart itself here.",
            hint: hint ?? manualHint
          });
        } else if (error instanceof APIError && error.status === 404) {
          update({
            phase: 'failed',
            message: 'This version of VideoAnnotator has no restart action.',
            hint: manualHint
          });
        } else {
          update({
            phase: 'failed',
            message: error instanceof Error ? error.message : "Couldn't restart VideoAnnotator.",
            hint
          });
        }
      }
    },
    [manualHint, waitForNewBoot]
  );

  const reset = useCallback(() => setShared({ phase: 'idle' }), []);

  return { state, restart, reset, manualHint };
}
