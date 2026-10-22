import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/api/client';
import type { VlmModelsResponse } from '@/types/pipelines';

export const VLM_MODELS_QUERY_KEY = ['videoannotator', 'vlm', 'models'] as const;

/**
 * Models on a model server, for the vlm_annotation model picker + reachability
 * indicator (VideoAnnotator specs 009, 033): the server's default Ollama
 * unless `server` names a listed one. A thrown query error means the server
 * is unreachable (503) — distinct from a successful response with an empty
 * `models` array (reachable, no models yet). No retry: an unreachable
 * server should surface immediately in the UI rather than silently retrying
 * for several seconds first.
 */
export function useVlmModels(options: { enabled?: boolean; server?: string } = {}) {
  const server = options.server ?? 'default';
  return useQuery<VlmModelsResponse>({
    queryKey: [...VLM_MODELS_QUERY_KEY, server],
    queryFn: () => apiClient.getVlmModels(server),
    enabled: options.enabled ?? true,
    staleTime: 60 * 1000,
    retry: false
  });
}
