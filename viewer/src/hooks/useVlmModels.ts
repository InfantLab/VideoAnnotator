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
 *
 * `pollWhileMissing` checks again every few seconds while the server is
 * unreachable or has no models, so a page waiting on someone to start Ollama
 * (or pull a model) notices without a reload.
 */
export const MISSING_MODEL_POLL_MS = 5000;

export function useVlmModels(options: { enabled?: boolean; server?: string; pollWhileMissing?: boolean } = {}) {
  const server = options.server ?? 'default';
  return useQuery<VlmModelsResponse>({
    queryKey: [...VLM_MODELS_QUERY_KEY, server],
    queryFn: () => apiClient.getVlmModels(server),
    enabled: options.enabled ?? true,
    staleTime: 60 * 1000,
    retry: false,
    refetchInterval: options.pollWhileMissing
      ? (query) =>
          query.state.status === 'error' || query.state.data?.models.length === 0 ? MISSING_MODEL_POLL_MS : false
      : false
  });
}
