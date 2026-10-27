import { keepPreviousData, useQuery } from '@tanstack/react-query';

import { apiClient } from '@/api/client';
import type { RunListResponse } from '@/types/videos';

export const RUNS_POLL_MS = 5_000;

/** Every run, those going on first (UAT D2), refreshed every 5 s while any is going on. */
export function useRuns(limit = 50, enabled = true) {
  return useQuery<RunListResponse>({
    queryKey: ['videos', 'runs', limit],
    queryFn: () => apiClient.listRuns(limit),
    enabled,
    placeholderData: keepPreviousData,
    refetchInterval: (q) => (q.state.data?.runs.some((r) => r.active) ? RUNS_POLL_MS : false),
    retry: false,
  });
}
