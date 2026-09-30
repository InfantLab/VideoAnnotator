// Restart-from-the-viewer flow (VideoAnnotator spec 011): request, the 409
// refusals, and waiting for the server to come back with a new boot id.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import React from 'react';
import { RestartRequiredBanner } from '@/components/RestartRequiredBanner';
import { apiClient } from '@/api/client';
import { APIError } from '@/api/handleError';

vi.mock('@/api/client', async () => {
  const actual = await vi.importActual<typeof import('@/api/client')>('@/api/client');
  return {
    ...actual,
    hasConfiguredApiToken: vi.fn(() => true),
    apiClient: {
      getCurrentUser: vi.fn(),
      restartServer: vi.fn(),
      getBootIdentity: vi.fn(),
      clearPipelineCache: vi.fn(),
      clearServerInfoCache: vi.fn(),
      getPipelineCatalog: vi.fn().mockResolvedValue({ pipelines: [], restartRequired: false })
    }
  };
});

const refusal = (code: string, extra: Record<string, unknown> = {}) =>
  new APIError('refused', 409, undefined, { error: { code, message: 'refused', ...extra } });

const renderBanner = () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <RestartRequiredBanner restartRequired />
    </QueryClientProvider>
  );
};

describe('RestartRequiredBanner', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers({ shouldAdvanceTime: true });
    vi.mocked(apiClient.getCurrentUser).mockResolvedValue({
      id: 'u1',
      username: 'admin',
      email: 'a@b.c',
      isAdmin: true
    });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('renders nothing when no restart is needed', () => {
    const queryClient = new QueryClient();
    const { container } = render(
      <QueryClientProvider client={queryClient}>
        <RestartRequiredBanner restartRequired={false} />
      </QueryClientProvider>
    );
    expect(container).toBeEmptyDOMElement();
  });

  it('tells a non-admin why they cannot restart, instead of offering it', async () => {
    vi.mocked(apiClient.getCurrentUser).mockResolvedValue({
      id: 'u2',
      username: 'irene',
      email: 'i@b.c',
      isAdmin: false
    });
    renderBanner();
    expect(await screen.findByText(/needs an administrator API key/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /restart server/i })).not.toBeInTheDocument();
  });

  it('restarts, waits through connection errors for a new boot id, then refreshes', async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    vi.mocked(apiClient.restartServer).mockResolvedValue({ bootId: 'old' });
    vi.mocked(apiClient.getBootIdentity)
      .mockRejectedValueOnce(new APIError('Network error', 0))
      .mockResolvedValueOnce({ bootId: 'old' })
      .mockResolvedValue({ bootId: 'new', restartMode: 'execv' });

    renderBanner();
    await user.click(await screen.findByRole('button', { name: /restart server/i }));

    expect(await screen.findByText(/Restarting server/)).toBeInTheDocument();
    expect(apiClient.restartServer).toHaveBeenCalledWith(false);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(7000);
    });

    await waitFor(() => {
      expect(apiClient.getPipelineCatalog).toHaveBeenCalledWith({
        forceRefresh: true,
        includeUnavailable: true
      });
    });
    expect(apiClient.getBootIdentity).toHaveBeenCalledTimes(3);
  });

  it('asks before interrupting running jobs, and forces only on confirmation', async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    vi.mocked(apiClient.restartServer)
      .mockRejectedValueOnce(refusal('JOBS_RUNNING', { details: { job_ids: ['a', 'b'] } }))
      .mockResolvedValueOnce({ bootId: 'old' });
    vi.mocked(apiClient.getBootIdentity).mockResolvedValue({ bootId: 'new' });

    renderBanner();
    await user.click(await screen.findByRole('button', { name: /restart server/i }));

    expect(await screen.findByText(/2 annotation jobs are running/)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /restart anyway/i }));

    expect(apiClient.restartServer).toHaveBeenLastCalledWith(true);
  });

  it("shows the server's own instructions when it can't restart itself", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    vi.mocked(apiClient.restartServer).mockRejectedValueOnce(
      refusal('RESTART_UNSUPPORTED', { hint: 'Stop it and run `videoannotator server` again.' })
    );

    renderBanner();
    await user.click(await screen.findByRole('button', { name: /restart server/i }));

    expect(await screen.findByText(/can't restart itself/)).toBeInTheDocument();
    expect(screen.getByText('Stop it and run `videoannotator server` again.')).toBeInTheDocument();
  });

  it('gives the manual route when the server never comes back', async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    vi.mocked(apiClient.restartServer).mockResolvedValue({ bootId: 'old' });
    vi.mocked(apiClient.getBootIdentity).mockRejectedValue(new APIError('Network error', 0));

    renderBanner();
    await user.click(await screen.findByRole('button', { name: /restart server/i }));
    await screen.findByText(/Restarting server/);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(125_000);
    });

    expect(await screen.findByText(/hasn't come back after two minutes/)).toBeInTheDocument();
  });
});
