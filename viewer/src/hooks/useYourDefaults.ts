import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { apiClient } from '@/api/client';
import type { YourDefaults } from '@/types/videos';

const KEY = ['videos', 'run-settings', 'defaults'] as const;

/** Your own settings per pipeline, which New run fills into what its starting point leaves unset (UAT #71). */
export function useYourDefaults(enabled = true) {
  return useQuery<YourDefaults>({ queryKey: KEY, queryFn: () => apiClient.getYourDefaults(), enabled, retry: false });
}

export function useSaveYourDefaults() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (config: YourDefaults['config']) => apiClient.putYourDefaults(config),
    onSuccess: (saved) => client.setQueryData(KEY, saved),
  });
}
