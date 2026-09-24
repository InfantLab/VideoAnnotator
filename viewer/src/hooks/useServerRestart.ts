import { useCallback, useEffect, useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/api/client';
import { APIError, apiErrorEnvelope } from '@/api/handleError';
import { useRefreshPipelineCatalog } from '@/hooks/usePipelineCatalog';

const POLL_INTERVAL_MS = 2000;
const GIVE_UP_AFTER_MS = 120_000;

export const MANUAL_RESTART_HINT =
  'Restart it the way it was started, e.g. stop it and run `videoannotator server` again, then refresh this page.';

export type ServerRestartState =
  | { phase: 'idle' }
  | { phase: 'requesting' }
  | { phase: 'restarting' }
  | { phase: 'confirm_force'; runningJobCount: number }
  | { phase: 'failed'; message: string; hint?: string }
  | { phase: 'timed_out' }
  | { phase: 'done' };

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Restart the VideoAnnotator server from the viewer (VideoAnnotator spec 011,
 * contracts/readiness-contract.md sections 4-5): ask for a restart, then poll
 * /health until `boot_id` changes. Connection errors while polling are expected,
 * because the server is down. On success, everything cached is refetched, since the
 * pipeline catalog (at least) has changed.
 */
export function useServerRestart() {
  const [state, setState] = useState<ServerRestartState>({ phase: 'idle' });
  const queryClient = useQueryClient();
  const refreshCatalog = useRefreshPipelineCatalog();
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const update = useCallback((next: ServerRestartState) => {
    if (mounted.current) setState(next);
  }, []);

  const waitForNewBoot = useCallback(
    async (previousBootId: string) => {
      const deadline = Date.now() + GIVE_UP_AFTER_MS;
      while (Date.now() < deadline) {
        await sleep(POLL_INTERVAL_MS);
        if (!mounted.current) return;
        try {
          const { bootId } = await apiClient.getBootIdentity();
          if (bootId && bootId !== previousBootId) {
            await refreshCatalog({ forceServerRefresh: true }).catch(() => {});
            await queryClient.invalidateQueries();
            update({ phase: 'done' });
            return;
          }
        } catch {
          // Still restarting.
        }
      }
      update({ phase: 'timed_out' });
    },
    [queryClient, refreshCatalog, update]
  );

  const restart = useCallback(
    async (force = false) => {
      update({ phase: 'requesting' });
      try {
        const { bootId } = await apiClient.restartServer(force);
        update({ phase: 'restarting' });
        await waitForNewBoot(bootId);
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
            message: "This server can't restart itself.",
            hint: hint ?? MANUAL_RESTART_HINT
          });
        } else if (error instanceof APIError && error.status === 404) {
          update({
            phase: 'failed',
            message: 'This server version has no restart action.',
            hint: MANUAL_RESTART_HINT
          });
        } else {
          update({
            phase: 'failed',
            message: error instanceof Error ? error.message : "Couldn't restart the server.",
            hint
          });
        }
      }
    },
    [update, waitForNewBoot]
  );

  const reset = useCallback(() => update({ phase: 'idle' }), [update]);

  return { state, restart, reset };
}
