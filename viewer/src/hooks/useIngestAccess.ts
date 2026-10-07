import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/api/client';
import { QueryKeys } from '@/types/api';
import type { IngestAccess } from '@/types/ingest';

/**
 * Whether this browser counts as being on the server's own machine, and so
 * may choose videos where they are (spec 022). Only the server can tell: it
 * sees how the request arrived, which the page's own URL doesn't say behind a
 * proxy or under Docker.
 *
 * A server too old for the endpoint, or any error, reads as "not the same
 * machine": the wizard then offers upload, which works everywhere.
 */
export function useIngestAccess() {
  const query = useQuery<IngestAccess>({
    queryKey: [...QueryKeys.ingestAccess, apiClient.baseURL],
    queryFn: () => apiClient.getIngestAccess(),
    retry: false,
    staleTime: 60 * 1000,
    refetchOnWindowFocus: false,
  });

  return {
    access: query.data ?? null,
    isLoading: query.isLoading,
    sameMachine: query.data?.same_machine ?? false,
    canReadInPlace: query.data?.can_read_in_place ?? false,
    canOpenFolders: query.data?.can_open_folders ?? false,
  };
}
