import { useQuery } from '@tanstack/react-query';
import { apiClient, hasConfiguredApiToken } from '@/api/client';
import { APIError } from '@/api/handleError';
import { QueryKeys } from '@/types/api';

const STALE_TIME = 5 * 60 * 1000; // 5 minutes

/**
 * The current session's own identity/admin status, via GET /api/v1/auth/me
 * (specs/002-pipeline-extras-install addendum). Used to decide up front
 * whether an admin-gated action (like installing a pipeline extras group)
 * will succeed, instead of only finding out via a bare 403.
 *
 * `isAdmin` is `'unknown'` - not `false` - whenever we genuinely can't tell:
 * no token configured, the request is still in flight, or the server predates
 * this endpoint (404). Callers should treat `'unknown'` as "don't block the
 * action, but don't claim admin either" - i.e. fall back to attempting the
 * action and handling its own 403, per Constitution Principle II (graceful
 * degradation against older servers).
 */
export function useCurrentUser() {
  const enabled = hasConfiguredApiToken();

  const query = useQuery({
    queryKey: QueryKeys.currentUser,
    queryFn: () => apiClient.getCurrentUser(),
    enabled,
    staleTime: STALE_TIME,
    retry: false
  });

  const endpointUnsupported =
    query.isError && query.error instanceof APIError && query.error.status === 404;

  const isAdmin: boolean | 'unknown' = query.data ? query.data.isAdmin : 'unknown';

  return {
    currentUser: query.data ?? null,
    isAdmin,
    isLoading: enabled && query.isLoading,
    endpointUnsupported,
    error: query.error
  };
}
