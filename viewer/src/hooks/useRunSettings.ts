import { useQuery } from '@tanstack/react-query';

import { apiClient } from '@/api/client';
import type { RunSettings } from '@/types/videos';

/** What New run starts from: a workflow, the project's settings, your last run, or the defaults (spec 029 FR-012). */
export function useRunSettings(params: { project?: string | null; workflow?: string | null }, enabled = true) {
  return useQuery<RunSettings>({
    queryKey: ['videos', 'run-settings', params.project ?? null, params.workflow ?? null],
    queryFn: () => apiClient.getRunSettings(params),
    enabled,
    staleTime: 0,
    retry: false,
  });
}
