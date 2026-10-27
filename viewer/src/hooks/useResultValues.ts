import { useQueries } from '@tanstack/react-query';

import { apiClient } from '@/api/client';
import type { ResultValues, VideoItem } from '@/types/videos';

/** Small enough that the first cells fill while the rest are still being read. */
const CHUNK = 40;

/** `<job_id>/<pipeline>`, as the server keys a value. */
export const valueKey = (jobId: string, pipeline: string) => `${jobId}/${pipeline}`;

/**
 * What each shown finished result found (UAT #10), fetched after the list so
 * the table appears at once; cells show their state until their value comes.
 */
export function useResultValues(videos: VideoItem[], pipelines: string[]): ResultValues {
  const pairs: { job_id: string; pipeline: string }[] = [];
  const seen = new Set<string>();
  for (const video of videos) {
    for (const pipeline of pipelines) {
      const result = video.results[pipeline];
      if (result?.state !== 'done') continue;
      const key = valueKey(result.job_id, pipeline);
      if (seen.has(key)) continue;
      seen.add(key);
      pairs.push({ job_id: result.job_id, pipeline });
    }
  }
  const chunks: (typeof pairs)[] = [];
  for (let i = 0; i < pairs.length; i += CHUNK) chunks.push(pairs.slice(i, i + CHUNK));

  const results = useQueries({
    queries: chunks.map((chunk) => ({
      queryKey: ['videoannotator', 'result-values', chunk.map((p) => valueKey(p.job_id, p.pipeline))],
      queryFn: () => apiClient.getResultValues(chunk),
      // A finished result's file doesn't change.
      staleTime: Infinity,
      retry: false,
    })),
  });
  return Object.assign({}, ...results.map((r) => r.data ?? {}));
}
