// Unit tests for useCurrentUser (specs/002-pipeline-extras-install addendum, GET /api/v1/auth/me)

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import React from 'react';
import { useCurrentUser } from '@/hooks/useCurrentUser';
import { apiClient, hasConfiguredApiToken } from '@/api/client';
import { APIError } from '@/api/handleError';

vi.mock('@/api/client', async () => {
  const actual = await vi.importActual<typeof import('@/api/client')>('@/api/client');
  return {
    ...actual,
    hasConfiguredApiToken: vi.fn(),
    apiClient: {
      getCurrentUser: vi.fn()
    }
  };
});

describe('useCurrentUser', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    vi.clearAllMocks();
  });

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );

  it('reports isAdmin: true for an admin identity', async () => {
    vi.mocked(hasConfiguredApiToken).mockReturnValue(true);
    vi.mocked(apiClient.getCurrentUser).mockResolvedValue({
      id: 1,
      username: 'alice',
      email: 'alice@example.com',
      isAdmin: true
    });

    const { result } = renderHook(() => useCurrentUser(), { wrapper });

    await waitFor(() => {
      expect(result.current.isAdmin).toBe(true);
    });
    expect(result.current.currentUser?.username).toBe('alice');
  });

  it('reports isAdmin: false for a non-admin identity - not "unknown"', async () => {
    vi.mocked(hasConfiguredApiToken).mockReturnValue(true);
    vi.mocked(apiClient.getCurrentUser).mockResolvedValue({
      id: 2,
      username: 'bob',
      email: 'bob@example.com',
      isAdmin: false
    });

    const { result } = renderHook(() => useCurrentUser(), { wrapper });

    await waitFor(() => {
      expect(result.current.isAdmin).toBe(false);
    });
  });

  it('reports isAdmin: "unknown" when no token is configured, without calling the endpoint', () => {
    vi.mocked(hasConfiguredApiToken).mockReturnValue(false);

    const { result } = renderHook(() => useCurrentUser(), { wrapper });

    expect(result.current.isAdmin).toBe('unknown');
    expect(apiClient.getCurrentUser).not.toHaveBeenCalled();
  });

  it('reports isAdmin: "unknown" and endpointUnsupported: true on a 404 (pre-v1.5.1 server)', async () => {
    vi.mocked(hasConfiguredApiToken).mockReturnValue(true);
    vi.mocked(apiClient.getCurrentUser).mockRejectedValue(new APIError('Not found', 404));

    const { result } = renderHook(() => useCurrentUser(), { wrapper });

    await waitFor(() => {
      expect(result.current.endpointUnsupported).toBe(true);
    });
    expect(result.current.isAdmin).toBe('unknown');
  });

  it('reports isAdmin: "unknown" (not endpointUnsupported) on a 401', async () => {
    vi.mocked(hasConfiguredApiToken).mockReturnValue(true);
    vi.mocked(apiClient.getCurrentUser).mockRejectedValue(new APIError('Unauthorized', 401));

    const { result } = renderHook(() => useCurrentUser(), { wrapper });

    await waitFor(() => {
      expect(result.current.error).toBeTruthy();
    });
    expect(result.current.isAdmin).toBe('unknown');
    expect(result.current.endpointUnsupported).toBe(false);
  });
});
