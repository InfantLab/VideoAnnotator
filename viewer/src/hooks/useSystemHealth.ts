import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/api/client';
import type { SystemHealthResponse } from '@/types/system';

export function useSystemHealth() {
    return useQuery<SystemHealthResponse>({
        queryKey: ['system', 'health'],
        queryFn: () => apiClient.getSystemHealth(),
        // Refetch every 30 seconds to keep GPU status current
        refetchInterval: 30000,
        staleTime: 15000,
        retry: 2,
    });
}
