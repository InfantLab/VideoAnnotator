import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/api/client';
import type { VlmModelsResponse } from '@/types/pipelines';

const VLM_MODELS_QUERY_KEY = ['videoannotator', 'vlm', 'models'] as const;

/**
 * Models pulled on the configured Ollama server, for the vlm_annotation
 * model picker + reachability indicator (VideoAnnotator spec 009). A
 * thrown query error means the server is unreachable (503
 * OLLAMA_UNREACHABLE) — distinct from a successful response with an empty
 * `models` array (reachable, nothing pulled yet). No retry: an
 * unreachable Ollama server should surface immediately in the UI rather
 * than silently retrying for several seconds first.
 */
export function useVlmModels(options: { enabled?: boolean } = {}) {
  return useQuery<VlmModelsResponse>({
    queryKey: VLM_MODELS_QUERY_KEY,
    queryFn: () => apiClient.getVlmModels(),
    enabled: options.enabled ?? true,
    staleTime: 60 * 1000,
    retry: false
  });
}
