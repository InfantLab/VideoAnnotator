import { useCallback } from 'react';
import { useQuery, useQueryClient, type UseQueryOptions } from '@tanstack/react-query';
import { apiClient } from '@/api/client';
import type {
  PipelineCatalogResponse,
  PipelineSchemaResponse,
  VideoAnnotatorServerInfo
} from '@/types/pipelines';

const PIPELINE_CATALOG_QUERY_KEY = ['videoannotator', 'pipelines', 'catalog'] as const;
const PIPELINE_SCHEMA_QUERY_KEY = (pipelineId: string) =>
  ['videoannotator', 'pipelines', 'schema', pipelineId] as const;
const SERVER_INFO_QUERY_KEY = ['videoannotator', 'server-info'] as const;

const DEFAULT_STALE_TIME = 5 * 60 * 1000; // 5 minutes

export const pipelineCatalogQueryOptions = (
  options: { forceRefresh?: boolean } = {}
): UseQueryOptions<PipelineCatalogResponse> => ({
  queryKey: PIPELINE_CATALOG_QUERY_KEY,
  queryFn: () => apiClient.getPipelineCatalog({ forceRefresh: options.forceRefresh }),
  staleTime: DEFAULT_STALE_TIME
});

export const pipelineSchemaQueryOptions = (
  pipelineId: string
): UseQueryOptions<PipelineSchemaResponse> => ({
  queryKey: PIPELINE_SCHEMA_QUERY_KEY(pipelineId),
  queryFn: () => apiClient.getPipelineSchema(pipelineId),
  staleTime: DEFAULT_STALE_TIME
});

export const serverInfoQueryOptions = (): UseQueryOptions<VideoAnnotatorServerInfo | null> => ({
  queryKey: SERVER_INFO_QUERY_KEY,
  queryFn: () => apiClient.getServerInfo(),
  staleTime: DEFAULT_STALE_TIME
});

export function usePipelineCatalog(options: { enabled?: boolean } = {}) {
  return useQuery({
    ...pipelineCatalogQueryOptions(),
    enabled: options.enabled ?? true
  });
}

export function usePipelineSchema(pipelineId: string | undefined, options: { enabled?: boolean } = {}) {
  return useQuery({
    ...pipelineSchemaQueryOptions(pipelineId ?? ''),
    enabled: Boolean(pipelineId) && (options.enabled ?? true)
  });
}

export function useVideoAnnotatorServerInfo(options: { enabled?: boolean } = {}) {
  return useQuery({
    ...serverInfoQueryOptions(),
    enabled: options.enabled ?? true
  });
}

export function useRefreshPipelineCatalog() {
  const queryClient = useQueryClient();

  return useCallback(
    async (options: { forceServerRefresh?: boolean } = {}) => {
      const { forceServerRefresh = false } = options;

      if (forceServerRefresh) {
        apiClient.clearPipelineCache();
        apiClient.clearServerInfoCache();
        const freshCatalog = await apiClient.getPipelineCatalog({ forceRefresh: true });
        await queryClient.setQueryData(PIPELINE_CATALOG_QUERY_KEY, freshCatalog);
        await queryClient.invalidateQueries({ queryKey: SERVER_INFO_QUERY_KEY });
        return freshCatalog;
      }

      await queryClient.invalidateQueries({ queryKey: PIPELINE_CATALOG_QUERY_KEY });
      await queryClient.invalidateQueries({ queryKey: SERVER_INFO_QUERY_KEY });
      return queryClient.getQueryData<PipelineCatalogResponse>(PIPELINE_CATALOG_QUERY_KEY);
    },
    [queryClient]
  );
}

export function useRefreshPipelineSchema() {
  const queryClient = useQueryClient();

  return useCallback(
    async (pipelineId: string) => {
      await queryClient.invalidateQueries({ queryKey: PIPELINE_SCHEMA_QUERY_KEY(pipelineId) });
    },
    [queryClient]
  );
}

export function useClearPipelineCatalogCache() {
  const queryClient = useQueryClient();

  return useCallback(async () => {
    apiClient.clearPipelineCache();
    await queryClient.removeQueries({ queryKey: PIPELINE_CATALOG_QUERY_KEY });
  }, [queryClient]);
}
