import { useServerCapabilitiesContext } from '@/contexts/ServerCapabilitiesContext';

/**
 * The server's version, or null when not connected. Also used where no
 * server context exists (an error page, tests), which the context's own hook
 * refuses.
 */
export function useServerVersionShown(): string | null {
  try {
    const version = useServerCapabilitiesContext().capabilities?.version;
    return version && version !== 'unknown' ? version : null;
  } catch {
    return null;
  }
}
