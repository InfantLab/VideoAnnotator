import { useQuery } from '@tanstack/react-query';

import { apiClient } from '@/api/client';

/** Every run on the video at `path`, with its files and projects (spec 028). */
export function useVideoRuns(path: string | null | undefined) {
  return useQuery({
    queryKey: ['videoRuns', path],
    queryFn: () => apiClient.getVideoRuns(path!),
    enabled: !!path,
    staleTime: 30_000,
    // A run still going: its status and the Runs tab follow it.
    refetchInterval: (q) => (q.state.data?.runs.some((r) => r.status === 'pending' || r.status === 'running') ? 5_000 : false),
    retry: false,
  });
}
