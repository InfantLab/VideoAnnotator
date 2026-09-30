// Unit tests for useVlmModels (VideoAnnotator spec 009, US2/FR-005)
// Covers: reachable server with models, unreachable server, reachable with
// zero models pulled -- the three states the model picker/reachability
// badge must be able to tell apart.

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import React from 'react';
import { useVlmModels } from '@/hooks/useVlmModels';
import { apiClient } from '@/api/client';
import { APIError } from '@/api/handleError';

vi.mock('@/api/client', () => ({
  apiClient: {
    getVlmModels: vi.fn()
  }
}));

describe('useVlmModels', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    });
    vi.clearAllMocks();
  });

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );

  it('returns the models a reachable server reports', async () => {
    vi.mocked(apiClient.getVlmModels).mockResolvedValueOnce({
      baseUrl: 'http://127.0.0.1:11434',
      models: ['qwen3.5:9b', 'gemma4:12b']
    });

    const { result } = renderHook(() => useVlmModels(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.models).toEqual(['qwen3.5:9b', 'gemma4:12b']);
  });

  it('surfaces an unreachable server as a query error, not a value', async () => {
    vi.mocked(apiClient.getVlmModels).mockRejectedValueOnce(
      new APIError('Cannot reach Ollama server', 503)
    );

    const { result } = renderHook(() => useVlmModels(), { wrapper });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(result.current.data).toBeUndefined();
  });

  it('distinguishes reachable-with-zero-models from unreachable', async () => {
    vi.mocked(apiClient.getVlmModels).mockResolvedValueOnce({
      baseUrl: 'http://127.0.0.1:11434',
      models: []
    });

    const { result } = renderHook(() => useVlmModels(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.isError).toBe(false);
    expect(result.current.data?.models).toEqual([]);
  });

  it('does not retry on failure, so an unreachable server surfaces immediately', async () => {
    vi.mocked(apiClient.getVlmModels).mockRejectedValueOnce(
      new APIError('Cannot reach Ollama server', 503)
    );

    const { result } = renderHook(() => useVlmModels(), { wrapper });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(apiClient.getVlmModels).toHaveBeenCalledTimes(1);
  });
});
