// Spec 024: Settings lists the shared folders and the results folder, and
// offers Stop sharing when VideoAnnotator was started by its launcher.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import React from 'react';
import { VideosAndResultsCard } from '@/components/VideosAndResultsCard';
import { apiClient } from '@/api/client';
import type { IngestAccess } from '@/types/ingest';

vi.mock('@/api/client', async () => {
  const actual = await vi.importActual<typeof import('@/api/client')>('@/api/client');
  return {
    ...actual,
    apiClient: {
      baseURL: 'http://127.0.0.1:18011',
      getIngestAccess: vi.fn(),
      stopSharing: vi.fn(),
    },
  };
});

const STUDIES = 'C:\\Users\\ada\\Studies';
const DRIVE = 'E:\\Data';

const access = (overrides: Partial<IngestAccess> = {}): IngestAccess => ({
  same_machine: true,
  can_read_in_place: true,
  reason: null,
  allowed_folders: [{ path: '/c/Users/ada/Studies', display_path: STUDIES }],
  results_root: { path: '/c/Users/ada/VideoAnnotator', display_path: 'C:\\Users\\ada\\VideoAnnotator' },
  can_open_folders: false,
  in_container: true,
  managed_by_launcher: true,
  shares: [
    { path: '/c/Users/ada/Studies', display_path: STUDIES, present: true, stop_requested: false },
    { path: DRIVE, display_path: DRIVE, present: false, stop_requested: false },
  ],
  ...overrides,
});

const renderCard = () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <VideosAndResultsCard />
    </QueryClientProvider>
  );
};

const row = (text: string) => screen.getByText(text).closest('li') as HTMLElement;

beforeEach(() => {
  vi.clearAllMocks();
});

describe('VideosAndResultsCard', () => {
  it('lists each shared folder, read-only, and the results folder', async () => {
    vi.mocked(apiClient.getIngestAccess).mockResolvedValue(access());
    renderCard();
    expect(await screen.findByText(STUDIES)).toBeInTheDocument();
    expect(within(row(STUDIES)).getByText('read-only')).toBeInTheDocument();
    expect(within(row(DRIVE)).getByText(/not found at the last start/)).toBeInTheDocument();
    expect(screen.getByText('C:\\Users\\ada\\VideoAnnotator')).toBeInTheDocument();
  });

  it('stops sharing at the next start', async () => {
    vi.mocked(apiClient.getIngestAccess).mockResolvedValue(access());
    vi.mocked(apiClient.stopSharing).mockResolvedValue({
      path: '/c/Users/ada/Studies',
      display_path: STUDIES,
      present: true,
      stop_requested: true,
    });
    renderCard();
    await screen.findByText(STUDIES);
    fireEvent.click(within(row(STUDIES)).getByRole('button', { name: 'Stop sharing' }));
    await waitFor(() => expect(apiClient.stopSharing).toHaveBeenCalledWith(STUDIES));
    expect(await within(row(STUDIES)).findByText('Stops when VideoAnnotator next starts')).toBeInTheDocument();
  });

  it('shows a stop already asked for', async () => {
    vi.mocked(apiClient.getIngestAccess).mockResolvedValue(
      access({ shares: [{ path: '/c/Users/ada/Studies', display_path: STUDIES, present: true, stop_requested: true }] })
    );
    renderCard();
    expect(await screen.findByText('Stops when VideoAnnotator next starts')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Stop sharing' })).not.toBeInTheDocument();
  });

  it('without the launcher, points to the compose settings and offers no button', async () => {
    vi.mocked(apiClient.getIngestAccess).mockResolvedValue(
      access({
        managed_by_launcher: false,
        shares: [{ path: '/videos', display_path: '/home/ada/Studies', present: true, stop_requested: false }],
      })
    );
    renderCard();
    expect(
      await screen.findByText('To change shared folders, change the compose settings (see the installation guide).')
    ).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Stop sharing' })).not.toBeInTheDocument();
  });

  it('with nothing shared, says how to share a folder', async () => {
    const reason =
      'VideoAnnotator can only see folders you share with it. To share one, run: videoannotator-start share';
    vi.mocked(apiClient.getIngestAccess).mockResolvedValue(
      access({ can_read_in_place: false, reason, allowed_folders: [], shares: [] })
    );
    renderCard();
    expect(await screen.findByText(reason)).toBeInTheDocument();
  });
});
