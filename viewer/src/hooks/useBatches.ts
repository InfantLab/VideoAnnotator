// React Query hooks for submission batches (VideoAnnotator spec 008).
//
// Polling only, deliberately: the server's SSE stream is additive, never
// authoritative, so everything here must be correct without it. `useSSE.ts`
// can later cut the poll interval, not replace these.

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/api/client';
import { useToast } from '@/hooks/use-toast';
import { showErrorToast } from '@/lib/toastHelpers';
import { parseApiError } from '@/lib/errorHandling';
import type { BatchListResponse, BatchSummary } from '@/types/batches';

export const BatchQueryKeys = {
  all: ['batches'] as const,
  list: (page: number) => ['batches', 'list', page] as const,
  detail: (batchId: string) => ['batches', batchId] as const,
  jobs: (batchId: string) => ['jobs', 'batch', batchId] as const,
};

/**
 * Poll faster while anything is still moving, slower once it isn't — the same
 * adaptive approach the jobs list already uses, so an idle screen isn't
 * hammering a researcher's laptop while a long batch runs overnight.
 */
const ACTIVE_POLL_MS = 5000;
const IDLE_POLL_MS = 30000;

function hasActiveWork(batches: BatchSummary[] | undefined): boolean {
  if (!batches) return false;
  return batches.some((b) => b.by_status.pending + b.by_status.running > 0);
}

/** All batches, newest submission first. */
export function useBatches(page: number = 1, perPage: number = 20) {
  return useQuery<BatchListResponse>({
    queryKey: BatchQueryKeys.list(page),
    queryFn: () => apiClient.getBatches(page, perPage),
    refetchInterval: (query) =>
      hasActiveWork(query.state.data?.batches) ? ACTIVE_POLL_MS : IDLE_POLL_MS,
    refetchOnWindowFocus: false,
  });
}

/** One batch's aggregate. */
export function useBatch(batchId: string | undefined) {
  return useQuery<BatchSummary>({
    queryKey: BatchQueryKeys.detail(batchId ?? ''),
    queryFn: () => apiClient.getBatch(batchId as string),
    enabled: !!batchId,
    refetchInterval: (query) =>
      hasActiveWork(query.state.data ? [query.state.data] : undefined)
        ? ACTIVE_POLL_MS
        : IDLE_POLL_MS,
    refetchOnWindowFocus: false,
  });
}

/** The member jobs of one batch — the per-video detail behind the aggregate. */
// The server has no cap on page size, but a request per hundred keeps each
// response small for a run of hundreds of videos.
const BATCH_JOBS_PAGE = 100;

/** Every video of one batch (all pages: a batch page that shows only some is wrong). */
export function useBatchJobs(batchId: string | undefined) {
  return useQuery({
    queryKey: BatchQueryKeys.jobs(batchId ?? ''),
    queryFn: async () => {
      const first = await apiClient.getJobs(1, BATCH_JOBS_PAGE, { batchId });
      const jobs = [...first.jobs];
      for (let page = 2; jobs.length < first.total; page++) {
        const next = await apiClient.getJobs(page, BATCH_JOBS_PAGE, { batchId });
        if (next.jobs.length === 0) break;
        jobs.push(...next.jobs);
      }
      return { ...first, jobs, page: 1, per_page: jobs.length };
    },
    enabled: !!batchId,
    refetchInterval: (query) => {
      const jobs = query.state.data?.jobs;
      const active = jobs?.some(
        (job) => job.status === 'pending' || job.status === 'running' || job.status === 'cancelling'
      );
      return active ? ACTIVE_POLL_MS : IDLE_POLL_MS;
    },
    refetchOnWindowFocus: false,
  });
}

/**
 * Cancel or retry a whole batch in one call.
 *
 * Both report per-job skips rather than failing wholesale (a half-finished
 * batch has jobs that can't be cancelled, and a batch being retried has jobs
 * that don't need retrying), so the toast tells the user what actually
 * happened rather than just "done".
 */
export function useBatchActions(batchId: string) {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: BatchQueryKeys.all });
    queryClient.invalidateQueries({ queryKey: ['jobs'] });
  };

  const cancel = useMutation({
    mutationFn: () => apiClient.cancelBatch(batchId),
    onSuccess: (result) => {
      const { cancelled, skipped } = result;
      toast({
        title: cancelled.length
          ? `Cancelled ${cancelled.length} job${cancelled.length === 1 ? '' : 's'}`
          : 'Nothing left to cancel',
        description: skipped.length
          ? `${skipped.length} job${skipped.length === 1 ? '' : 's'} had already finished.`
          : undefined,
      });
      invalidate();
    },
    onError: (error) => showErrorToast(toast, parseApiError(error)),
  });

  const retry = useMutation({
    mutationFn: () => apiClient.retryBatch(batchId),
    onSuccess: (result) => {
      const { retried, skipped } = result;
      toast({
        title: retried.length
          ? `Retrying ${retried.length} job${retried.length === 1 ? '' : 's'}`
          : 'Nothing to retry',
        description: skipped.length
          ? `${skipped.length} job${skipped.length === 1 ? '' : 's'} skipped — ${skipped[0].reason}`
          : undefined,
      });
      invalidate();
    },
    onError: (error) => showErrorToast(toast, parseApiError(error)),
  });

  return {
    cancelBatch: cancel.mutate,
    retryBatch: retry.mutate,
    isCancelling: cancel.isPending,
    isRetrying: retry.isPending,
  };
}
