import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/api/client';
import type { SystemHealthResponse } from '@/types/system';

export function useSystemHealth(options: { enabled?: boolean } = {}) {
    const { enabled = false } = options; // DISABLED by default - too slow, causes page hang

    return useQuery<SystemHealthResponse>({
        queryKey: ['system', 'health'],
        queryFn: () => apiClient.getSystemHealth(),
        enabled, // Only fetch when explicitly enabled
        // Refetch every 30 seconds to keep GPU status current (when enabled)
        refetchInterval: enabled ? 30000 : false,
        staleTime: 15000,
        retry: 2,
    });
}
