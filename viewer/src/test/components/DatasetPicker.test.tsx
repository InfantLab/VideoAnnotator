import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import React from 'react';
import { DatasetPicker } from '@/components/DatasetPicker';
import { apiClient } from '@/api/client';
import * as handles from '@/lib/datasetHandles';
import type { SavedDataset } from '@/types/datasets';

vi.mock('@/api/client', () => ({
  apiClient: { listDatasets: vi.fn(), scanServerFolder: vi.fn(), updateDataset: vi.fn() },
  hasConfiguredApiToken: () => false,
}));
vi.mock('@/hooks/useCurrentUser', () => ({ useCurrentUser: () => ({ currentUser: null }) }));
vi.mock('@/lib/datasetHandles', () => ({
  rememberedFolder: vi.fn(),
  rememberFolder: vi.fn(),
  supportsFolderHandles: () => true,
  videosIn: vi.fn(),
}));

const base = { owner_user_id: 'u1', owner_name: 'ada', created_at: '2026-10-04T00:00:00Z' };
const serverDataset: SavedDataset = {
  ...base,
  id: 'srv',
  name: 'Session 1',
  server_folder: '/data/s1',
  server_folder_recursive: true,
  video_manifest: [{ filename: 'a.mp4', size_bytes: 1, relative_path: 'a.mp4' }],
};
const filesDataset: SavedDataset = {
  ...base,
  id: 'up',
  name: 'Pilot',
  video_manifest: [
    { filename: 'a.mp4', size_bytes: 1, relative_path: 'a.mp4' },
    { filename: 'b.mp4', size_bytes: 2, relative_path: 'b.mp4' },
  ],
};

const renderPicker = (props: Partial<React.ComponentProps<typeof DatasetPicker>> = {}) => {
  const onUseFiles = vi.fn();
  const onUseServerFolder = vi.fn();
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <DatasetPicker onUseFiles={onUseFiles} onUseServerFolder={onUseServerFolder} {...props} />
    </QueryClientProvider>,
  );
  return { onUseFiles, onUseServerFolder };
};

const file = (name: string, size: number) => {
  const f = new File(['x'], name);
  Object.defineProperty(f, 'size', { value: size });
  return f;
};

describe('DatasetPicker', () => {
  beforeEach(() => {
    vi.mocked(apiClient.listDatasets).mockResolvedValue({ datasets: [serverDataset, filesDataset], total: 2 });
  });

  it('lists datasets with their source and owner', async () => {
    renderPicker();
    expect(await screen.findByText('Session 1')).toBeInTheDocument();
    expect(screen.getByText(/server folder \/data\/s1 · saved by ada/)).toBeInTheDocument();
    expect(screen.getByText(/2 videos · uploaded from a browser/)).toBeInTheDocument();
  });

  it('uses an unchanged server folder at once', async () => {
    vi.mocked(apiClient.scanServerFolder).mockResolvedValue({
      path: '/data/s1',
      recursive: true,
      videos: [{ relative_path: 'a.mp4', name: 'a.mp4', size_bytes: 1 }],
    });
    const { onUseServerFolder } = renderPicker();
    await userEvent.click((await screen.findAllByRole('button', { name: 'Use' }))[0]);
    await waitFor(() =>
      expect(onUseServerFolder).toHaveBeenCalledWith({ path: '/data/s1', recursive: true, videoCount: 1 }, serverDataset),
    );
  });

  it('shows what changed before using a remembered folder, then continues with the matches', async () => {
    vi.mocked(handles.rememberedFolder).mockResolvedValue({} as FileSystemDirectoryHandle);
    const a = file('a.mp4', 1);
    vi.mocked(handles.videosIn).mockResolvedValue([
      { name: 'a.mp4', size: 1, relativePath: 'a.mp4', item: a },
      { name: 'c.mp4', size: 3, relativePath: 'c.mp4', item: file('c.mp4', 3) },
    ]);
    const { onUseFiles } = renderPicker();
    await userEvent.click((await screen.findAllByRole('button', { name: 'Use' }))[1]);

    expect(await screen.findByText(/have changed since “Pilot” was saved/)).toBeInTheDocument();
    expect(screen.getByText('Missing (1)')).toBeInTheDocument();
    expect(screen.getByText('New, not in the dataset (1)')).toBeInTheDocument();
    expect(onUseFiles).not.toHaveBeenCalled();

    await userEvent.click(screen.getByRole('button', { name: /Continue with the 1 matching video$/ }));
    expect(onUseFiles).toHaveBeenCalledWith([{ file: a, relativePath: 'a.mp4' }], filesDataset, {});
  });

  it('asks for the folder when this browser does not remember it', async () => {
    vi.mocked(handles.rememberedFolder).mockResolvedValue(null);
    renderPicker();
    await userEvent.click((await screen.findAllByRole('button', { name: 'Use' }))[1]);
    expect(await screen.findByText(/Choose the folder the videos of “Pilot” are in/)).toBeInTheDocument();
  });
});
