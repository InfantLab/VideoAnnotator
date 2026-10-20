import { keepPreviousData, useQuery } from '@tanstack/react-query';

import { apiClient } from '@/api/client';
import type { VideosQuery, VideosResponse } from '@/types/videos';

export const VIDEOS_POLL_MS = 5_000;

export function videosKey(query: VideosQuery) {
  return ['videos', 'list', query] as const;
}

/** Something on screen is still moving: poll until it settles. */
export function isMoving(data: VideosResponse | undefined): boolean {
  if (!data) return false;
  if (data.runs.some((r) => r.active)) return true;
  return data.videos.some((v) => Object.values(v.results).some((r) => r.state === 'running' || r.state === 'waiting'));
}

/** The videos of one scope (spec 029 R2), refreshed every 5 s while any is running or waiting. */
export function useVideos(query: VideosQuery, enabled = true) {
  return useQuery<VideosResponse>({
    queryKey: videosKey(query),
    queryFn: () => apiClient.listVideos(query),
    enabled,
    staleTime: VIDEOS_POLL_MS,
    placeholderData: keepPreviousData,
    refetchInterval: (q) => (isMoving(q.state.data) ? VIDEOS_POLL_MS : false),
    retry: false,
  });
}
