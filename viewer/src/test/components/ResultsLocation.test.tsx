// Spec 022: run and job pages say where results are, and how to get at them.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import React from 'react';
import { ResultsLocation } from '@/components/ResultsLocation';
import { apiClient } from '@/api/client';
import { APIError } from '@/api/handleError';
import type { IngestAccess } from '@/types/ingest';

vi.mock('@/api/client', async () => {
  const actual = await vi.importActual<typeof import('@/api/client')>('@/api/client');
  return {
    ...actual,
    apiClient: {
      baseURL: 'http://127.0.0.1:18011',
      getIngestAccess: vi.fn(),
      openResultsFolder: vi.fn(),
      getBatchResultsZip: vi.fn(),
    },
  };
});

const FOLDER = {
  path: '/results/Wave 2 (2026-10-06)',
  display_path: '/home/ada/VideoAnnotator/Wave 2 (2026-10-06)',
};

const access = (canOpen: boolean): IngestAccess => ({
  same_machine: true,
  can_read_in_place: true,
  reason: null,
  allowed_folders: [],
  results_root: { path: '/results', display_path: '/home/ada/VideoAnnotator' },
  can_open_folders: canOpen,
});

const wrap = (ui: React.ReactElement) =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      {ui}
    </QueryClientProvider>
  );

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(apiClient.getIngestAccess).mockResolvedValue(access(true));
});

describe('ResultsLocation', () => {
  it('shows where the results are on this computer', () => {
    wrap(<ResultsLocation folder={FOLDER} label="This run's results" />);
    expect(screen.getByTestId('results-location')).toHaveTextContent(FOLDER.display_path);
  });

  it('renders nothing for jobs from before results folders', () => {
    const { container } = wrap(<ResultsLocation folder={null} label="x" />);
    expect(container).toBeEmptyDOMElement();
  });

  it('opens the folder on this computer', async () => {
    vi.mocked(apiClient.openResultsFolder).mockResolvedValue(undefined);
    wrap(<ResultsLocation folder={FOLDER} label="x" />);
    await userEvent.click(await screen.findByRole('button', { name: /Open folder/ }));
    expect(apiClient.openResultsFolder).toHaveBeenCalledWith(FOLDER.path);
  });

  it('offers no Open folder where the server can’t open one', async () => {
    vi.mocked(apiClient.getIngestAccess).mockResolvedValue(access(false));
    wrap(<ResultsLocation folder={FOLDER} label="x" />);
    await waitFor(() => expect(apiClient.getIngestAccess).toHaveBeenCalled());
    expect(screen.queryByRole('button', { name: /Open folder/ })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Copy location/ })).toBeInTheDocument();
  });

  it('falls back to copying when opening is refused', async () => {
    vi.mocked(apiClient.openResultsFolder).mockRejectedValue(new APIError('no desktop', 409));
    wrap(<ResultsLocation folder={FOLDER} label="x" />);
    await userEvent.click(await screen.findByRole('button', { name: /Open folder/ }));
    expect(await screen.findByRole('status')).toHaveTextContent(/Copy the location instead/);
  });

  it('copies the location the researcher can use', async () => {
    const user = userEvent.setup();
    wrap(<ResultsLocation folder={FOLDER} label="x" />);
    await user.click(screen.getByRole('button', { name: /Copy location/ }));
    expect(await navigator.clipboard.readText()).toBe(FOLDER.display_path);
    expect(screen.getByRole('button', { name: /Copied/ })).toBeInTheDocument();
  });

  it('downloads the whole run as one zip', async () => {
    vi.mocked(apiClient.getBatchResultsZip).mockResolvedValue(new Blob(['zip']));
    URL.createObjectURL = vi.fn(() => 'blob:x');
    URL.revokeObjectURL = vi.fn();
    wrap(<ResultsLocation folder={FOLDER} label="x" runZip={{ batchId: 'b1' }} />);
    await userEvent.click(screen.getByRole('button', { name: /Download results/ }));
    expect(apiClient.getBatchResultsZip).toHaveBeenCalledWith('b1');
    expect(screen.getByText(/without the videos/)).toBeInTheDocument();
  });
});
