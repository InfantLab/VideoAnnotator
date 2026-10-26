import { hasConfiguredApiToken } from '@/api/client';
import { APIError, isNeedsKey } from '@/api/handleError';
import { useServerCapabilitiesContext } from '@/contexts/ServerCapabilitiesContext';
import { useCurrentUser } from '@/hooks/useCurrentUser';

export type ConnectionState = 'checking' | 'no-server' | 'needs-key' | 'key-refused' | 'connected';

export interface Connection {
  state: ConnectionState;
  /** Who the key belongs to, once the server has accepted it. */
  user: { name: string; isAdmin: boolean } | null;
  version?: string;
}

/**
 * Whether this viewer can use the server, checked rather than assumed
 * (viewer design P1, UAT #38): the server answering its health check isn't
 * enough when it needs a key, so a key is only "connected" once the server
 * has said whose it is.
 */
export function useConnection(): Connection {
  const { capabilities, isLoading } = useServerCapabilitiesContext();
  const { currentUser, isLoading: checkingKey, error, endpointUnsupported } = useCurrentUser();
  const version = capabilities?.version;

  if (!capabilities) return { state: isLoading ? 'checking' : 'no-server', user: null };
  if (!hasConfiguredApiToken()) {
    return { state: capabilities.authRequired ? 'needs-key' : 'connected', user: null, version };
  }
  if (currentUser) {
    const name = currentUser.username || currentUser.email || 'your key';
    return { state: 'connected', user: { name, isAdmin: currentUser.isAdmin }, version };
  }
  // A server from before GET /auth/me can't say whose the key is.
  if (endpointUnsupported) return { state: 'connected', user: null, version };
  if (isNeedsKey(error)) return { state: 'key-refused', user: null, version };
  if (error instanceof APIError && error.status === 0) return { state: 'no-server', user: null };
  if (checkingKey || !error) return { state: 'checking', user: null, version };
  return { state: 'connected', user: null, version };
}
