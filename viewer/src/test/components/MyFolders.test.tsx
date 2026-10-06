// Spec 022: on the server's own machine, videos are chosen in "My folders" and
// read where they are; upload is a link for videos on another computer.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import React from 'react';
import { VideoUploadStep } from '@/pages/NewJob';
import { ServerFolderPicker } from '@/components/ServerFolderPicker';
import { apiClient } from '@/api/client';
import type { IngestAccess } from '@/types/ingest';

vi.mock('@/api/client', async () => {
  const actual = await vi.importActual<typeof import('@/api/client')>('@/api/client');
  return {
    ...actual,
    apiClient: {
      baseURL: 'http://127.0.0.1:18011',
      getIngestAccess: vi.fn(),
      browseServerFolders: vi.fn(),
      scanServerFolder: vi.fn(),
    },
  };
});

const HOME = '/home/ada';
const STUDY = `${HOME}/Studies/BabyJokes`;

const access = (overrides: Partial<IngestAccess> = {}): IngestAccess => ({
  same_machine: true,
  can_read_in_place: true,
  reason: null,
  allowed_folders: [{ path: HOME, display_path: HOME }],
  results_root: { path: `${HOME}/VideoAnnotator`, display_path: `${HOME}/VideoAnnotator` },
  can_open_folders: true,
  ...overrides,
});

const wrap = (ui: React.ReactElement) => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>{ui}</MemoryRouter>
    </QueryClientProvider>
  );
};

const step = (props: Partial<React.ComponentProps<typeof VideoUploadStep>>) => (
  <VideoUploadStep
    selectedFiles={[]}
    setSelectedFiles={() => {}}
    videoSource="server"
    setVideoSource={() => {}}
    serverFolder={null}
    setServerFolder={() => {}}
    dataset={{
      fromDataset: null,
      setFromDataset: () => {},
      relativePaths: new Map(),
      setRelativePaths: () => {},
      folderHandle: null,
      setFolderHandle: () => {},
    }}
    storedRun={null}
    setStoredRun={() => {}}
    access={access()}
    accessLoading={false}
    {...props}
  />
);

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(apiClient.getIngestAccess).mockResolvedValue(access());
  vi.mocked(apiClient.browseServerFolders).mockImplementation(async (path?: string) => ({
    path: path ?? null,
    parent: path === STUDY ? `${HOME}/Studies` : null,
    roots: [HOME],
    directories: path === STUDY ? [{ name: 'site_a', path: `${STUDY}/site_a`, video_count: 1 }] : [],
    videos: [],
    video_count: 3,
    truncated: false,
  }));
  vi.mocked(apiClient.scanServerFolder).mockImplementation(async (path: string, recursive: boolean) => ({
    path,
    recursive,
    videos: [
      { relative_path: 'child01.mp4', name: 'child01.mp4', size_bytes: 1048576 },
      { relative_path: 'child02.mp4', name: 'child02.mp4', size_bytes: 2097152 },
      { relative_path: 'child03.mp4', name: 'child03.mp4', size_bytes: 3145728 },
      ...(recursive ? [{ relative_path: 'site_a/child04.mp4', name: 'child04.mp4', size_bytes: 1 }] : []),
    ],
  }));
});

describe('Choose Videos, step 1', () => {
  it('opens on My folders on the same machine, with upload as a link', async () => {
    const setVideoSource = vi.fn();
    wrap(step({ setVideoSource }));

    expect(screen.getByRole('tab', { name: /My folders/ })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('tab', { name: /Saved datasets/ })).toBeInTheDocument();
    expect(screen.queryByRole('tab', { name: /Upload/ })).not.toBeInTheDocument();
    expect(await screen.findByText(/used where they are, never copied/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /Videos on another computer\? Upload them/ }));
    expect(setVideoSource).toHaveBeenCalledWith('upload');
  });

  it('offers upload, not My folders, when the server is on another computer', () => {
    wrap(step({ videoSource: 'upload', access: access({ same_machine: false, can_read_in_place: false }) }));
    expect(screen.getByRole('tab', { name: /Upload videos/ })).toHaveAttribute('aria-selected', 'true');
    expect(screen.queryByRole('tab', { name: /My folders/ })).not.toBeInTheDocument();
  });

  it('says why when this computer’s folders can’t be read', () => {
    wrap(
      step({
        access: access({ can_read_in_place: false, reason: 'No video folder is set up. Set VIDEOS_DIR' }),
      })
    );
    expect(screen.getByText(/Set VIDEOS_DIR/)).toBeInTheDocument();
    expect(apiClient.browseServerFolders).not.toHaveBeenCalled();
  });

  it('waits for the server before choosing a default', () => {
    wrap(step({ accessLoading: true }));
    expect(screen.queryByRole('tab')).not.toBeInTheDocument();
  });
});

describe('My folders picker', () => {
  it('starts in the first allowed folder and selects nothing until asked', async () => {
    const onSelect = vi.fn();
    wrap(<ServerFolderPicker selection={null} onSelect={onSelect} />);
    await waitFor(() => expect(apiClient.browseServerFolders).toHaveBeenCalledWith(HOME));
    expect(await screen.findByText('child01.mp4')).toBeInTheDocument();
    expect(screen.getByText(/0 videos selected/)).toBeInTheDocument();
  });

  it('runs exactly the ticked videos', async () => {
    const onSelect = vi.fn();
    wrap(<ServerFolderPicker selection={{ path: STUDY, videoCount: 0, recursive: false, files: [] }} onSelect={onSelect} />);
    await userEvent.click(await screen.findByRole('checkbox', { name: /child02.mp4/ }));
    await userEvent.click(screen.getByRole('checkbox', { name: /child03.mp4/ }));
    expect(screen.getByText(/2 videos selected/)).toBeInTheDocument();
    expect(onSelect).toHaveBeenLastCalledWith({
      path: STUDY,
      videoCount: 2,
      recursive: false,
      files: ['child02.mp4', 'child03.mp4'],
    });
  });

  it('Select all sends the whole folder, as before', async () => {
    const onSelect = vi.fn();
    wrap(<ServerFolderPicker selection={{ path: STUDY, videoCount: 0, recursive: false, files: [] }} onSelect={onSelect} />);
    await screen.findByText('child01.mp4');
    await userEvent.click(screen.getByRole('button', { name: 'Select all' }));
    expect(onSelect).toHaveBeenLastCalledWith({ path: STUDY, videoCount: 3, recursive: false });
    expect(screen.getByRole('button', { name: 'Clear selection' })).toBeInTheDocument();
  });

  it('lists subfolder videos by their path when asked', async () => {
    const onSelect = vi.fn();
    wrap(<ServerFolderPicker selection={{ path: STUDY, videoCount: 0, recursive: false, files: [] }} onSelect={onSelect} />);
    await screen.findByText('child01.mp4');
    await userEvent.click(screen.getByRole('checkbox', { name: /Include subfolders/ }));
    await waitFor(() => expect(apiClient.scanServerFolder).toHaveBeenCalledWith(STUDY, true));
    await userEvent.click(await screen.findByRole('checkbox', { name: /site_a\/child04.mp4/ }));
    expect(onSelect).toHaveBeenLastCalledWith({
      path: STUDY,
      videoCount: 1,
      recursive: true,
      files: ['site_a/child04.mp4'],
    });
  });

  it('keeps a whole-folder selection from a saved dataset whole', async () => {
    const onSelect = vi.fn();
    wrap(<ServerFolderPicker selection={{ path: STUDY, videoCount: 3, recursive: false }} onSelect={onSelect} />);
    await screen.findByText('child01.mp4');
    expect(screen.getByText(/3 videos selected/)).toBeInTheDocument();
    // Unchanged, so the parent isn't told: that would unlink the dataset.
    expect(onSelect).not.toHaveBeenCalled();
  });

  it('shows host paths under Docker', async () => {
    vi.mocked(apiClient.getIngestAccess).mockResolvedValue(
      access({ allowed_folders: [{ path: '/videos', display_path: '/home/ada/Studies' }] })
    );
    vi.mocked(apiClient.browseServerFolders).mockResolvedValue({
      path: '/videos', parent: null, roots: ['/videos'], directories: [], videos: [], video_count: 0, truncated: false,
    });
    wrap(<ServerFolderPicker selection={null} onSelect={() => {}} />);
    expect(await screen.findByText('/home/ada/Studies')).toBeInTheDocument();
  });
});
