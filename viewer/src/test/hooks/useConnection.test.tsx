/** Connected means checked, not assumed (viewer design P1, UAT #38). */
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { renderHook } from '@testing-library/react';

import { APIError, NEEDS_KEY_MESSAGE, isNeedsKey } from '@/api/handleError';
import { useConnection } from '@/hooks/useConnection';

const server = vi.hoisted(() => ({
  capabilities: null as { version: string; authRequired?: boolean } | null,
  isLoading: false,
}));
const me = vi.hoisted(() => ({
  currentUser: null as { username: string; email: string; isAdmin: boolean } | null,
  isLoading: false,
  error: null as unknown,
  endpointUnsupported: false,
}));
const key = vi.hoisted(() => ({ set: false }));

vi.mock('@/contexts/ServerCapabilitiesContext', () => ({ useServerCapabilitiesContext: () => server }));
vi.mock('@/hooks/useCurrentUser', () => ({ useCurrentUser: () => me }));
vi.mock('@/api/client', () => ({ hasConfiguredApiToken: () => key.set }));

const state = () => renderHook(() => useConnection()).result.current;

describe('useConnection', () => {
  beforeEach(() => {
    Object.assign(server, { capabilities: { version: '1.7.0', authRequired: true }, isLoading: false });
    Object.assign(me, { currentUser: null, isLoading: false, error: null, endpointUnsupported: false });
    key.set = false;
  });

  it('is not connected when the server answers but needs a key nobody gave', () => {
    expect(state().state).toBe('needs-key');
  });

  it('is connected without a key when the server needs none', () => {
    server.capabilities = { version: '1.7.0', authRequired: false };
    expect(state()).toMatchObject({ state: 'connected', user: null });
  });

  it('is connected as the key’s owner once the server accepts it', () => {
    key.set = true;
    me.currentUser = { username: 'caspar', email: 'c@example.com', isAdmin: true };
    expect(state()).toMatchObject({ state: 'connected', user: { name: 'caspar', isAdmin: true } });
  });

  it('says a refused key is refused', () => {
    key.set = true;
    me.error = new APIError('Invalid API key', 401);
    expect(state().state).toBe('key-refused');
  });

  it('is checking while the key is checked, and with no server says so', () => {
    key.set = true;
    me.isLoading = true;
    expect(state().state).toBe('checking');
    server.capabilities = null;
    expect(state().state).toBe('no-server');
  });
});

describe('APIError for a 401', () => {
  it('says to connect with a key, never what the server said', () => {
    const error = new APIError('API key required. Set AUTH_REQUIRED=false to disable authentication.', 401);
    expect(error.message).toBe(NEEDS_KEY_MESSAGE);
    expect(error.serverMessage).toContain('AUTH_REQUIRED');
    expect(isNeedsKey(error)).toBe(true);
    expect(new APIError('Not found', 404).message).toBe('Not found');
  });
});
