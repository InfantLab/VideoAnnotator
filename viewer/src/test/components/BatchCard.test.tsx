// Unit tests for BatchCard — the card that replaces N independent job rows.
// Covers the states a researcher actually sees: running with and without an
// ETA, finished, partly failed, and which bulk actions are offered when.

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import React from 'react';
import { BatchCard } from '@/components/BatchCard';
import { apiClient } from '@/api/client';
import type { BatchSummary } from '@/types/batches';

vi.mock('@/api/client', () => ({
  apiClient: {
    cancelBatch: vi.fn(),
    retryBatch: vi.fn(),
  },
}));

vi.mock('@/hooks/use-toast', () => ({
  useToast: () => ({ toast: vi.fn() }),
}));

function makeBatch(overrides: Partial<BatchSummary> = {}): BatchSummary {
  const by_status = {
    pending: 0,
    running: 0,
    completed: 0,
    failed: 0,
    cancelled: 0,
    ...(overrides.by_status ?? {}),
  };
  const total =
    overrides.total ??
    by_status.pending + by_status.running + by_status.completed + by_status.failed + by_status.cancelled;

  // by_status is applied after the spread deliberately: it is already merged
  // from the overrides above, and a raw spread would replace the merged value
  // with a partial one.
  return {
    batch_id: 'batch-1',
    batch_name: 'Irene corpus',
    dataset_id: null,
    created_at: new Date().toISOString(),
    completion_percentage: 0,
    estimated_seconds_remaining: null,
    ...overrides,
    total,
    by_status,
  };
}

describe('BatchCard', () => {
  let queryClient: QueryClient;
  let user: ReturnType<typeof userEvent.setup>;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
    });
    user = userEvent.setup();
    vi.clearAllMocks();
  });

  const renderCard = (batch: BatchSummary) =>
    render(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter>
          <BatchCard batch={batch} />
        </MemoryRouter>
      </QueryClientProvider>
    );

  it('shows the batch name and video count as one unit', () => {
    renderCard(makeBatch({ by_status: { pending: 12, running: 0, completed: 0, failed: 0, cancelled: 0 } }));

    expect(screen.getByText('Irene corpus')).toBeInTheDocument();
    expect(screen.getByText(/12 videos/)).toBeInTheDocument();
  });

  it('falls back to describing the batch when it has no name', () => {
    renderCard(
      makeBatch({
        batch_name: null,
        by_status: { pending: 5, running: 0, completed: 0, failed: 0, cancelled: 0 },
      })
    );

    // Never the raw uuid — that tells a researcher nothing.
    expect(screen.queryByText('batch-1')).not.toBeInTheDocument();
    expect(screen.getAllByText(/5 videos/).length).toBeGreaterThan(0);
  });

  it('says it is estimating rather than inventing a time', () => {
    renderCard(
      makeBatch({
        by_status: { pending: 8, running: 2, completed: 0, failed: 0, cancelled: 0 },
        estimated_seconds_remaining: null,
      })
    );

    expect(screen.getByText(/estimating time left/i)).toBeInTheDocument();
  });

  it('shows a real ETA once the server has one', () => {
    renderCard(
      makeBatch({
        by_status: { pending: 6, running: 2, completed: 4, failed: 0, cancelled: 0 },
        estimated_seconds_remaining: 600,
        completion_percentage: 33.3,
      })
    );

    expect(screen.getByText(/~10 minutes left/)).toBeInTheDocument();
    expect(screen.getByText(/4 of 12 finished/)).toBeInTheDocument();
  });

  it('reports completion when everything is terminal', () => {
    renderCard(
      makeBatch({
        by_status: { pending: 0, running: 0, completed: 11, failed: 1, cancelled: 0 },
        completion_percentage: 100,
      })
    );

    expect(screen.getByText(/All 12 finished/)).toBeInTheDocument();
  });

  it('offers cancel-all only while something can still be cancelled', () => {
    const { unmount } = renderCard(
      makeBatch({ by_status: { pending: 3, running: 1, completed: 0, failed: 0, cancelled: 0 } })
    );
    expect(screen.getByRole('button', { name: /cancel all/i })).toBeInTheDocument();
    unmount();

    renderCard(
      makeBatch({ by_status: { pending: 0, running: 0, completed: 4, failed: 0, cancelled: 0 } })
    );
    expect(screen.queryByRole('button', { name: /cancel all/i })).not.toBeInTheDocument();
  });

  it('offers retry only when something failed or was cancelled, naming the count', () => {
    const { unmount } = renderCard(
      makeBatch({ by_status: { pending: 0, running: 0, completed: 9, failed: 2, cancelled: 1 } })
    );
    expect(screen.getByRole('button', { name: /retry 3/i })).toBeInTheDocument();
    unmount();

    renderCard(
      makeBatch({ by_status: { pending: 0, running: 0, completed: 12, failed: 0, cancelled: 0 } })
    );
    expect(screen.queryByRole('button', { name: /retry/i })).not.toBeInTheDocument();
  });

  it('cancels the whole batch in a single call', async () => {
    vi.mocked(apiClient.cancelBatch).mockResolvedValue({
      batch_id: 'batch-1',
      cancelled: ['a', 'b'],
      skipped: [],
    });

    renderCard(
      makeBatch({ by_status: { pending: 2, running: 0, completed: 0, failed: 0, cancelled: 0 } })
    );
    await user.click(screen.getByRole('button', { name: /cancel all/i }));

    await waitFor(() => {
      expect(apiClient.cancelBatch).toHaveBeenCalledTimes(1);
    });
    expect(apiClient.cancelBatch).toHaveBeenCalledWith('batch-1');
  });

  it('retries the whole batch in a single call', async () => {
    vi.mocked(apiClient.retryBatch).mockResolvedValue({
      batch_id: 'batch-1',
      retried: ['a'],
      skipped: [],
    });

    renderCard(
      makeBatch({ by_status: { pending: 0, running: 0, completed: 0, failed: 1, cancelled: 0 } })
    );
    await user.click(screen.getByRole('button', { name: /retry 1/i }));

    await waitFor(() => {
      expect(apiClient.retryBatch).toHaveBeenCalledTimes(1);
    });
    expect(apiClient.retryBatch).toHaveBeenCalledWith('batch-1');
  });

  it('only shows status chips for states that have jobs in them', () => {
    renderCard(
      makeBatch({ by_status: { pending: 0, running: 3, completed: 1, failed: 0, cancelled: 0 } })
    );

    expect(screen.getByText('3 running')).toBeInTheDocument();
    expect(screen.getByText('1 done')).toBeInTheDocument();
    expect(screen.queryByText(/0 failed/)).not.toBeInTheDocument();
    expect(screen.queryByText(/0 queued/)).not.toBeInTheDocument();
  });

  it('links through to the batch it represents', () => {
    renderCard(
      makeBatch({ by_status: { pending: 1, running: 0, completed: 0, failed: 0, cancelled: 0 } })
    );

    expect(screen.getByRole('link', { name: /Irene corpus/ })).toHaveAttribute(
      'href',
      '/batches/batch-1'
    );
  });
});
