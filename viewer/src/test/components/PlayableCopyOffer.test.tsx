/** Viewer design D7, UAT #54: a copy this browser can play. */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { getApiClient } from '@/api/client';
import { PlayableCopyOffer } from '@/components/PlayableCopyOffer';

const client = () => Object.getPrototypeOf(getApiClient());

function show(onReady = vi.fn()) {
  const queries = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={queries}>
      <PlayableCopyOffer path="/data/v.mp4" onReady={onReady} />
    </QueryClientProvider>,
  );
  return onReady;
}

afterEach(() => vi.restoreAllMocks());

describe('PlayableCopyOffer', () => {
  it('offers to make a copy, says it is being made, and switches once it is ready', async () => {
    const get = vi.spyOn(client(), 'getPlayableCopy').mockResolvedValue({ state: 'none', error: null });
    const make = vi.spyOn(client(), 'makePlayableCopy').mockResolvedValue({ state: 'making', error: null });
    const onReady = show();
    await userEvent.click(await screen.findByRole('button', { name: /Make a copy this browser can play/ }));
    expect(make).toHaveBeenCalledWith('/data/v.mp4');
    expect(await screen.findByText(/Making a copy this browser can play/)).toBeInTheDocument();
    get.mockResolvedValue({ state: 'ready', error: null });
    await waitFor(() => expect(onReady).toHaveBeenCalled(), { timeout: 4000 });
  });

  it('uses a copy made before straight away', async () => {
    vi.spyOn(client(), 'getPlayableCopy').mockResolvedValue({ state: 'ready', error: null });
    const onReady = show();
    await waitFor(() => expect(onReady).toHaveBeenCalled());
  });

  it('without ffmpeg on the server, says so and how to re-encode', async () => {
    vi.spyOn(client(), 'getPlayableCopy').mockResolvedValue({ state: 'unavailable', error: "ffmpeg isn't installed on the server, so it can't make a copy." });
    show();
    expect(await screen.findByText(/ffmpeg isn't installed/)).toBeInTheDocument();
    expect(screen.getByText(/libx264/)).toBeInTheDocument();
  });
});
