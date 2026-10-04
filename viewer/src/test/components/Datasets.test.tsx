import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import Datasets from '@/pages/Datasets';
import { apiClient } from '@/api/client';
import { APIError } from '@/api/handleError';
import type { SavedDataset } from '@/types/datasets';

vi.mock('@/api/client', () => ({
  apiClient: { listDatasets: vi.fn(), createDataset: vi.fn(), updateDataset: vi.fn(), deleteDataset: vi.fn() },
  hasConfiguredApiToken: () => true,
}));
const currentUser = { id: 'me', username: 'me', email: 'me@x', isAdmin: false };
vi.mock('@/hooks/useCurrentUser', () => ({ useCurrentUser: () => ({ currentUser }) }));
vi.mock('@/lib/datasetHandles', () => ({ forgetFolder: vi.fn() }));

const mine: SavedDataset = {
  id: 'd1',
  name: 'Pilot',
  owner_user_id: 'me',
  owner_name: 'me',
  video_manifest: [{ filename: 'a.mp4', size_bytes: 1048576, relative_path: 'p/a.mp4' }],
  created_at: '2026-10-01T00:00:00Z',
};
const theirs: SavedDataset = { ...mine, id: 'd2', name: 'Shared', owner_user_id: 'other', owner_name: 'bob' };

const renderPage = () =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <MemoryRouter>
        <Datasets />
      </MemoryRouter>
    </QueryClientProvider>,
  );

const upload = async (text: string) => {
  const input = document.querySelector('input[type="file"]') as HTMLInputElement;
  await userEvent.upload(input, new File([text], 'pilot.dataset.json', { type: 'application/json' }));
};

describe('Datasets page', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(apiClient.listDatasets).mockResolvedValue({ datasets: [mine, theirs], total: 2 });
  });

  it('lists datasets with owner, and offers edit and delete only on your own', async () => {
    renderPage();
    expect(await screen.findByText('Pilot')).toBeInTheDocument();
    expect(screen.getByText(/saved by bob/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Delete Pilot' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Delete Shared' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Export Shared' })).toBeInTheDocument();
  });

  it('says deleting leaves videos and past jobs alone', async () => {
    renderPage();
    await userEvent.click(await screen.findByRole('button', { name: 'Delete Pilot' }));
    expect(await screen.findByText(/jobs already run from it keep their results/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Delete' }));
    await waitFor(() => expect(apiClient.deleteDataset).toHaveBeenCalledWith('d1'));
  });

  it('explains what a dataset is when there are none', async () => {
    vi.mocked(apiClient.listDatasets).mockResolvedValue({ datasets: [], total: 0 });
    renderPage();
    expect(await screen.findByText('No saved datasets yet')).toBeInTheDocument();
    expect(screen.queryByText(/Coming Soon/i)).not.toBeInTheDocument();
  });

  it('imports under a free name when the name is taken', async () => {
    vi.mocked(apiClient.createDataset)
      .mockRejectedValueOnce(new APIError('taken', 409))
      .mockResolvedValueOnce({ ...mine, name: 'Pilot (imported)' });
    renderPage();
    await screen.findByText('Pilot');
    await upload(JSON.stringify({ name: 'Pilot', video_manifest: mine.video_manifest }));
    expect(await screen.findByText(/Imported as “Pilot \(imported\)”/)).toBeInTheDocument();
    expect(vi.mocked(apiClient.createDataset).mock.calls[1][0].name).toBe('Pilot (imported)');
  });

  it('rejects a file that is not an export, creating nothing', async () => {
    renderPage();
    await screen.findByText('Pilot');
    await upload('{"hello": 1}');
    expect(await screen.findByText(/missing name, video_manifest/)).toBeInTheDocument();
    expect(apiClient.createDataset).not.toHaveBeenCalled();
  });
});
