import { render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { APIClient } from '@/api/client';
import { BatchNavigation } from '@/components/BatchNavigation';

describe('BatchNavigation', () => {
  afterEach(() => vi.restoreAllMocks());

  it('shows the batch, n of N, and previous/next to videos with results', async () => {
    vi.spyOn(APIClient.prototype, 'getBatch').mockResolvedValue({
      batch_id: 'b1', batch_name: 'Peekaboo session', total: 3,
      by_status: { pending: 0, running: 0, completed: 2, failed: 1, cancelled: 0 },
    } as never);
    vi.spyOn(APIClient.prototype, 'getJobs').mockResolvedValue({
      jobs: [
        { id: 'j1', status: 'completed', video_filename: 'take1.mp4' },
        { id: 'j2', status: 'completed', video_filename: 'take2.mp4' },
        { id: 'j3', status: 'failed', video_filename: 'take3.mp4' },
      ],
    } as never);

    render(
      <QueryClientProvider client={new QueryClient()}>
        <MemoryRouter>
          <BatchNavigation jobId="j2" batchId="b1" />
        </MemoryRouter>
      </QueryClientProvider>
    );

    const link = await screen.findByRole('link', { name: /Peekaboo session · 2 of 3/ });
    expect(link).toHaveAttribute('href', '/batches/b1');
    expect(screen.getByRole('button', { name: 'Previous video in batch' })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Previous video in batch' })).toHaveAttribute('title', 'Previous: take1.mp4');
    // The only later video failed: nothing to open.
    expect(screen.getByRole('button', { name: 'Next video in batch' })).toBeDisabled();
  });
});
