/**
 * @file React hooks for managing pipeline catalog and server information.
 * 
 * This module provides hooks for fetching and caching pipeline catalogs,
 * parameter schemas, and server information from the VideoAnnotator API.
 */

import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useCallback } from 'react';
import { apiClient } from '@/api/client';
import type {
    PipelineCatalogResponse,
    PipelineSchemaResponse,
    VideoAnnotatorServerInfo,
} from '@/types/pipelines';

/**
 * Hook for fetching and caching the pipeline catalog from the server.
 * 
 * @param options - Query options for controlling fetch behavior
 * @returns Query result containing the pipeline catalog and server info
 */
export function usePipelineCatalog(options: { enabled?: boolean } = {}) {
    const { enabled = true } = options;

    return useQuery({
        queryKey: ['pipeline-catalog'],
        queryFn: () => apiClient.getPipelineCatalog(),
        staleTime: 5 * 60 * 1000, // 5 minutes
        enabled,
        retry: (failureCount, error) => {
            // Don't retry on 404/405 errors (unsupported endpoints)
            if (error instanceof Error && error.message.includes('404')) {
                return false;
            }
            return failureCount < 2;
        },
    });
}

/**
 * Hook for fetching the parameter schema for a specific pipeline.
 * 
 * @param pipelineId - The ID of the pipeline to fetch schema for
 * @param options - Query options for controlling fetch behavior
 * @returns Query result containing the pipeline schema
 */
export function usePipelineSchema(
    pipelineId: string | null,
    options: { enabled?: boolean } = {}
) {
    const { enabled = true } = options;

    return useQuery({
        queryKey: ['pipeline-schema', pipelineId],
        queryFn: () => {
            if (!pipelineId) {
                throw new Error('Pipeline ID is required');
            }
            return apiClient.getPipelineSchema(pipelineId);
        },
        enabled: enabled && !!pipelineId,
        staleTime: 10 * 60 * 1000, // 10 minutes (schemas change less frequently)
        retry: (failureCount, error) => {
            // Don't retry on 404/405 errors (unsupported endpoints)
            if (error instanceof Error && error.message.includes('404')) {
                return false;
            }
            return failureCount < 2;
        },
    });
}

/**
 * Hook for fetching server information and feature flags.
 * 
 * @param options - Query options for controlling fetch behavior
 * @returns Query result containing server info and capabilities
 */
export function useServerInfo(options: { enabled?: boolean } = {}) {
    const { enabled = true } = options;

    return useQuery({
        queryKey: ['server-info'],
        queryFn: () => apiClient.getServerInfo(),
        staleTime: 2 * 60 * 1000, // 2 minutes
        enabled,
        retry: (failureCount, error) => {
            // Allow retries for server info as it's critical
            return failureCount < 3;
        },
    });
}

/**
 * Hook that provides utilities for refreshing pipeline-related data.
 * 
 * @returns Object with refresh functions for different data types
 */
export function usePipelineUtils() {
    const queryClient = useQueryClient();

    const refreshCatalog = useCallback(async () => {
        // Clear cache and force refetch
        apiClient.clearPipelineCache();
        await queryClient.invalidateQueries({ queryKey: ['pipeline-catalog'] });
        return queryClient.refetchQueries({ queryKey: ['pipeline-catalog'] });
    }, [queryClient]);

    const refreshServerInfo = useCallback(async () => {
        // Clear cache and force refetch
        apiClient.clearServerInfoCache();
        await queryClient.invalidateQueries({ queryKey: ['server-info'] });
        return queryClient.refetchQueries({ queryKey: ['server-info'] });
    }, [queryClient]);

    const clearAllCache = useCallback(() => {
        apiClient.clearPipelineCache();
        apiClient.clearServerInfoCache();
        queryClient.invalidateQueries({ queryKey: ['pipeline-catalog'] });
        queryClient.invalidateQueries({ queryKey: ['server-info'] });
        queryClient.invalidateQueries({ queryKey: ['pipeline-schema'] });
    }, [queryClient]);

    return {
        refreshCatalog,
        refreshServerInfo,
        clearAllCache,
    };
}

/**
 * Hook that combines pipeline catalog and server info for convenience.
 * 
 * @param options - Configuration options
 * @param options.enabled - Whether to automatically fetch data (default: false for lazy loading)
 * @returns Combined state with catalog, server info, and loading states
 */
export function usePipelineData(options: { enabled?: boolean } = { enabled: false }) {
    const catalogQuery = usePipelineCatalog({ enabled: options.enabled });
    const serverQuery = useServerInfo({ enabled: options.enabled });

    return {
        catalog: catalogQuery.data?.catalog,
        server: catalogQuery.data?.server || serverQuery.data,
        isLoading: catalogQuery.isLoading || serverQuery.isLoading,
        isError: catalogQuery.isError || serverQuery.isError,
        error: catalogQuery.error || serverQuery.error,
        refetch: () => {
            catalogQuery.refetch();
            serverQuery.refetch();
        },
    };
}