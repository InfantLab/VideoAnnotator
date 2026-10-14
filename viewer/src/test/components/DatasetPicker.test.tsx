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
  apiClient: { listDatasets: vi.fn(), scanServerFolder: vi.fn(), updateDataset: vi.fn(), getStoredVideos: vi.fn() },
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

const stored = (a: string | null, b: string | null) => {
  const videos = [
    { filename: 'a.mp4', size_bytes: 1, job_id: a },
    { filename: 'b.mp4', size_bytes: 2, job_id: b },
  ];
  const count = videos.filter((v) => v.job_id).length;
  return { dataset_id: 'up', videos, stored: count, missing: videos.length - count };
};

const renderPicker = (props: Partial<React.ComponentProps<typeof DatasetPicker>> = {}) => {
  const onUseFiles = vi.fn();
  const onUseServerFolder = vi.fn();
  const onUseStored = vi.fn();
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <DatasetPicker
        onUseFiles={onUseFiles}
        onUseServerFolder={onUseServerFolder}
        onUseStored={onUseStored}
        {...props}
      />
    </QueryClientProvider>,
  );
  return { onUseFiles, onUseServerFolder, onUseStored };
};

const file = (name: string, size: number) => {
  const f = new File(['x'], name);
  Object.defineProperty(f, 'size', { value: size });
  return f;
};

describe('DatasetPicker', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(apiClient.listDatasets).mockResolvedValue({ datasets: [serverDataset, filesDataset], total: 2 });
    // By default the server has none of the uploaded videos any more.
    vi.mocked(apiClient.getStoredVideos).mockResolvedValue(stored(null, null));
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
    expect(await screen.findByText(/server no longer has the videos of “Pilot”/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /on the server/ })).not.toBeInTheDocument();
  });

  it('uses the server\'s stored copies without asking for anything', async () => {
    vi.mocked(apiClient.getStoredVideos).mockResolvedValue(stored('job-a', 'job-b'));
    const { onUseStored, onUseFiles } = renderPicker();
    await userEvent.click((await screen.findAllByRole('button', { name: 'Use' }))[1]);

    await waitFor(() => expect(onUseStored).toHaveBeenCalledWith(filesDataset, stored('job-a', 'job-b')));
    expect(handles.rememberedFolder).not.toHaveBeenCalled();
    expect(onUseFiles).not.toHaveBeenCalled();
    expect(screen.queryByRole('button', { name: 'Choose folder' })).not.toBeInTheDocument();
  });

  it('offers the stored ones, or the folder, when some were deleted', async () => {
    vi.mocked(handles.rememberedFolder).mockResolvedValue(null);
    vi.mocked(apiClient.getStoredVideos).mockResolvedValue(stored('job-a', null));
    const { onUseStored } = renderPicker();
    await userEvent.click((await screen.findAllByRole('button', { name: 'Use' }))[1]);

    expect(await screen.findByText(/1 of the 2 videos of “Pilot” are still on/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Choose folder' })).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Run the 1 on the server' }));
    expect(onUseStored).toHaveBeenCalledWith(filesDataset, stored('job-a', null));
  });

  it('falls back to the folder on a server without stored-video lookup', async () => {
    vi.mocked(handles.rememberedFolder).mockResolvedValue(null);
    vi.mocked(apiClient.getStoredVideos).mockRejectedValue(new Error('404'));
    renderPicker();
    await userEvent.click((await screen.findAllByRole('button', { name: 'Use' }))[1]);
    expect(await screen.findByRole('button', { name: 'Choose folder' })).toBeInTheDocument();
  });

  describe('a dataset of chosen videos (spec 022)', () => {
    const pair: SavedDataset = {
      ...base,
      id: 'pair',
      name: 'Pair',
      server_folder: '/data/s1',
      server_selection: true,
      video_manifest: [
        { filename: 'a.mp4', size_bytes: 1, relative_path: 'a.mp4' },
        { filename: 'c.mp4', size_bytes: 3, relative_path: 'site_a/c.mp4' },
      ],
    };
    const scanned = (names: [string, number][]) => ({
      path: '/data/s1',
      recursive: true,
      videos: names.map(([rel, size]) => ({ relative_path: rel, name: rel.split('/').pop()!, size_bytes: size })),
    });

    beforeEach(() => {
      vi.mocked(apiClient.listDatasets).mockResolvedValue({ datasets: [pair], total: 1 });
    });

    it('runs exactly its videos, ignoring other files in the folder, without asking', async () => {
      vi.mocked(apiClient.scanServerFolder).mockResolvedValue(
        scanned([['a.mp4', 1], ['b.mp4', 2], ['site_a/c.mp4', 3], ['site_a/new.mp4', 4]])
      );
      const { onUseServerFolder } = renderPicker();
      expect(await screen.findByText(/chosen videos in \/data\/s1/)).toBeInTheDocument();
      await userEvent.click(screen.getByRole('button', { name: /Use/ }));
      await waitFor(() =>
        expect(onUseServerFolder).toHaveBeenCalledWith(
          { path: '/data/s1', recursive: true, videoCount: 2, files: ['a.mp4', 'site_a/c.mp4'] },
          pair
        )
      );
      expect(apiClient.scanServerFolder).toHaveBeenCalledWith('/data/s1', true);
      expect(screen.queryByText(/have changed/)).not.toBeInTheDocument();
    });

    it('says which of its videos moved, then continues with the rest', async () => {
      vi.mocked(apiClient.scanServerFolder).mockResolvedValue(scanned([['a.mp4', 1], ['b.mp4', 2]]));
      const { onUseServerFolder } = renderPicker();
      await userEvent.click(await screen.findByRole('button', { name: /Use/ }));
      expect(await screen.findByText(/have changed since/)).toBeInTheDocument();
      expect(screen.getByText('site_a/c.mp4')).toBeInTheDocument();
      expect(screen.queryByRole('button', { name: /Update the dataset/ })).not.toBeInTheDocument();
      await userEvent.click(screen.getByRole('button', { name: /Continue with the 1 matching video/ }));
      expect(onUseServerFolder).toHaveBeenCalledWith(
        { path: '/data/s1', recursive: false, videoCount: 1, files: ['a.mp4'] },
        pair
      );
    });
  });
});
