// React hook for server capability detection
// Provides caching and automatic refresh for VideoAnnotator server capabilities

import { useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/api/client';
import { detectServerCapabilities, areCapabilitiesStale } from '@/api/capabilities';
import type { ServerCapabilities } from '@/types/api';
import { QueryKeys } from '@/types/api';

/**
 * Hook to detect and cache server capabilities
 * 
 * Automatically detects v1.3.0 features by inspecting health endpoint
 * Caches result for 5 minutes
 * Provides manual refresh capability
 * 
 * @param options - React Query options
 * @returns Query result with capabilities data and refresh function
 */
export function useServerCapabilities(options?: {
  refetchInterval?: number;
  enabled?: boolean;
}) {
  const queryClient = useQueryClient();

  const query = useQuery({
    queryKey: QueryKeys.serverCapabilities,
    queryFn: async (): Promise<ServerCapabilities> => {
      // Fetch health response
      const healthResponse = await apiClient.getEnhancedHealth();

      // Detect capabilities from response
      const capabilities = detectServerCapabilities(healthResponse);

      return capabilities;
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
    gcTime: 10 * 60 * 1000, // 10 minutes (formerly cacheTime)
    refetchInterval: options?.refetchInterval,
    enabled: options?.enabled !== false,
    retry: 3,
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
  });

  /**
   * Manually refresh server capabilities
   * Invalidates cache and triggers new fetch
   */
  const refresh = async () => {
    await queryClient.invalidateQueries({
      queryKey: QueryKeys.serverCapabilities,
    });
  };

  /**
   * Check if cached data is stale (> 2 minutes old)
   */
  const isStale = query.data ? areCapabilitiesStale(query.data) : false;

  return {
    ...query,
    capabilities: query.data,
    refresh,
    isStale,
  };
}

/**
 * Hook to check if a specific feature is supported
 * 
 * @param feature - Feature name to check
 * @returns boolean indicating if feature is supported
 */
export function useFeatureSupport(
  feature: keyof Omit<ServerCapabilities, 'version' | 'detectedAt'>
): boolean {
  const { capabilities } = useServerCapabilities();

  if (!capabilities) {
    // Conservative default: assume not supported if unknown
    return false;
  }

  return capabilities[feature];
}

/**
 * Hook to get server version string
 * 
 * @returns Server version or 'unknown'
 */
export function useServerVersion(): string {
  const { capabilities } = useServerCapabilities();
  return capabilities?.version || 'unknown';
}
