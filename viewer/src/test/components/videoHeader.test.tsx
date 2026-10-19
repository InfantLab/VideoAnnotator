/** Spec 028 US7: where this video belongs; the header's actions (FR-021). */
import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { getApiClient } from '@/api/client';
import { VIDEO_PATH, demoRun, demoVideoRuns, lanesLoaded, mockVideoApi, renderVideoPage, wideLayout } from '@/test/videoPageHarness';

const client = () => Object.getPrototypeOf(getApiClient());
const open = (from: string, run?: string) =>
  renderVideoPage(`?${new URLSearchParams({ path: VIDEO_PATH, from, ...(run ? { run } : {}) })}`);
const lastPathAsked = (api: ReturnType<typeof mockVideoApi>) => api.getVideoRuns.mock.calls.at(-1)?.[0];

let api: ReturnType<typeof mockVideoApi>;

beforeEach(() => {
  api = mockVideoApi();
  wideLayout();
});
afterEach(() => vi.restoreAllMocks());

describe('the list a video was opened from', () => {
  it('a run: "3 of 5", and › opens the next video with results, keeping the list', async () => {
    const job = (id: string, name: string, status = 'completed') => ({
      id,
      status,
      video_filename: name,
      video_path: `/data/pk/${name}`,
    });
    vi.spyOn(client(), 'getBatch').mockResolvedValue({ batch_id: 'b1', batch_name: 'Pilot run' });
    vi.spyOn(client(), 'getJobs').mockResolvedValue({
      jobs: [job('j1', 'a.mp4'), job('j2', 'b.mp4'), job('job-1', 'demo_clip.mp4'), job('j4', 'd.mp4', 'failed'), job('j5', 'e.mp4')],
      total: 5,
    });
    open('run:b1', 'job-1');
    await lanesLoaded();
    expect(await screen.findByTestId('list-position')).toHaveTextContent('Pilot run · 3 of 5');
    fireEvent.click(screen.getByRole('button', { name: 'Next video' }));
    // d.mp4 failed: the next with results is e.mp4.
    await waitFor(() => expect(lastPathAsked(api)).toBe('/data/pk/e.mp4'));
    expect(screen.getByRole('button', { name: 'Back to the list' })).toBeInTheDocument();
  });

  it('a project in a server folder: its manifest order', async () => {
    vi.spyOn(client(), 'getDataset').mockResolvedValue({
      id: 'd1',
      name: 'Peekaboo study',
      server_folder: '/data/pk',
      video_manifest: [
        { filename: 'a.mp4', size_bytes: 1 },
        { filename: 'demo_clip.mp4', size_bytes: 1 },
        { filename: 'c.mp4', size_bytes: 1, relative_path: 'more/c.mp4' },
      ],
    });
    open('project:d1');
    await lanesLoaded();
    expect(await screen.findByTestId('list-position')).toHaveTextContent('Peekaboo study · 2 of 3');
    fireEvent.click(screen.getByRole('button', { name: 'Next video' }));
    await waitFor(() => expect(lastPathAsked(api)).toBe('/data/pk/more/c.mp4'));
  });

  it('an uploaded project: the stored copy is looked up only when opened', async () => {
    vi.spyOn(client(), 'getDataset').mockResolvedValue({ id: 'd2', name: 'Uploads', server_folder: null, video_manifest: [] });
    vi.spyOn(client(), 'getStoredVideos').mockResolvedValue({
      dataset_id: 'd2',
      videos: [
        { filename: 'demo_clip.mp4', size_bytes: 1, job_id: 'job-1' },
        { filename: 'gone.mp4', size_bytes: 1, job_id: null },
        { filename: 'z.mp4', size_bytes: 1, job_id: 'jz' },
      ],
      stored: 2,
      missing: 1,
    });
    const getJob = vi.spyOn(client(), 'getJob').mockResolvedValue({ id: 'jz', video_path: '/storage/videos/ab/z.mp4' });
    open('project:d2');
    await lanesLoaded();
    expect(await screen.findByTestId('list-position')).toHaveTextContent('Uploads · 1 of 3');
    expect(getJob).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: 'Next video' }));
    await waitFor(() => expect(lastPathAsked(api)).toBe('/storage/videos/ab/z.mp4'));
  });

  it('a folder: its videos in name order', async () => {
    vi.spyOn(client(), 'browseServerFolders').mockResolvedValue({
      path: '/data/pk',
      parent: '/data',
      roots: ['/data'],
      directories: [],
      videos: [
        { name: 'p10.mp4', path: '/data/pk/p10.mp4', size_bytes: 1 },
        { name: 'demo_clip.mp4', path: VIDEO_PATH, size_bytes: 1 },
        { name: 'p2.mp4', path: '/data/pk/p2.mp4', size_bytes: 1 },
      ],
      video_count: 3,
      truncated: false,
    });
    open('folder:/data/pk');
    await lanesLoaded();
    expect(await screen.findByTestId('list-position')).toHaveTextContent('pk · 1 of 3');
    expect(screen.getByRole('button', { name: 'Previous video' })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: 'Next video' }));
    await waitFor(() => expect(lastPathAsked(api)).toBe('/data/pk/p2.mp4'));
  });
});

describe('the header', () => {
  it('lists the video\'s projects when it wasn\'t opened from a list', async () => {
    api.getVideoRuns.mockResolvedValue(demoVideoRuns({ projects: [{ id: 'd1', name: 'Peekaboo study' }] }));
    renderVideoPage();
    await lanesLoaded();
    const link = within(screen.getByTestId('video-projects')).getByRole('link', { name: 'Peekaboo study' });
    expect(new URL(link.getAttribute('href')!, 'http://x').searchParams.get('from')).toBe('project:d1');
    expect(screen.queryByTestId('list-navigation')).toBeNull();
  });

  it('opens Ask AI on this video', async () => {
    renderVideoPage();
    await lanesLoaded();
    const link = screen.getByRole('link', { name: /Ask AI/ });
    expect(new URL(link.getAttribute('href')!, 'http://x').searchParams.get('video')).toBe(VIDEO_PATH);
  });

  it('offers an export for each run the lanes show', async () => {
    const older = demoRun('job-0', { run_name: 'Older run', created_at: '2026-10-01T09:00:00Z' });
    api.getVideoRuns.mockResolvedValue(demoVideoRuns({ runs: [demoRun('job-1', { run_name: 'Newer run' }), older] }));
    renderVideoPage();
    await lanesLoaded();
    fireEvent.click(screen.getByRole('button', { name: 'Export' }));
    expect(within(await screen.findByTestId('export-menu')).queryByText('Newer run')).toBeNull();

    const speech = screen.getAllByRole('group').find((g) => g.dataset.kind === 'speech')!;
    fireEvent.click(within(speech).getByTestId('lane-run'));
    fireEvent.click(within(await screen.findByTestId('run-menu')).getAllByText('Older run')[0]);
    await waitFor(() => {
      const lane = screen.getAllByRole('group').find((g) => g.dataset.kind === 'speech')!;
      expect(within(lane).getByTestId('lane-run')).toHaveTextContent('Older run');
    });
    fireEvent.click(screen.getByRole('button', { name: 'Export' }));
    const menu = await screen.findByTestId('export-menu');
    expect(within(menu).getByText('Newer run')).toBeInTheDocument();
    expect(within(menu).getByText('Older run')).toBeInTheDocument();
  });
});
