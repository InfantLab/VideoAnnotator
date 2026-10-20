/** Spec 028 US7: where this video belongs; the header's actions (FR-021). */
import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { getApiClient } from '@/api/client';
import { VIDEO_PATH, demoRun, demoVideoRuns, lanesLoaded, mockVideoApi, renderVideoPage, wideLayout } from '@/test/videoPageHarness';
import { EMPTY, video } from '@/test/fixtures/videos';
import type { VideoItem } from '@/types/videos';

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
  const item = (name: string, over: Partial<VideoItem> = {}): VideoItem =>
    video(name, { path: `/data/pk/${name}`, paths: [`/data/pk/${name}`], ...over });
  const listing = (name: string, videos: VideoItem[], kind = 'run') =>
    vi.spyOn(client(), 'listVideos').mockResolvedValue({
      ...EMPTY,
      scope: { kind, id: 'x', name, path: null, recursive: false },
      videos,
      total: videos.length,
    });

  it('a run: "3 of 5", and › opens the next video that has something to show, keeping the list', async () => {
    const asked = listing('Pilot run', [
      item('a.mp4', { run_job_id: 'j1' }),
      item('b.mp4', { run_job_id: 'j2' }),
      item('demo_clip.mp4', { run_job_id: 'job-1' }),
      item('d.mp4', { run_job_id: 'j4', available: false, results: {} }),
      item('e.mp4', { run_job_id: 'j5' }),
    ]);
    open('run:b1', 'job-1');
    await lanesLoaded();
    expect(await screen.findByTestId('list-position')).toHaveTextContent('Pilot run · 3 of 5');
    expect(asked).toHaveBeenCalledWith({ run: 'b1' });
    fireEvent.click(screen.getByRole('button', { name: 'Next video' }));
    // d.mp4 is gone and has no results: the next is e.mp4, on its job in this run.
    await waitFor(() => expect(lastPathAsked(api)).toBe('/data/pk/e.mp4'));
    expect(screen.getByRole('button', { name: 'Back to the list' })).toBeInTheDocument();
  });

  it('a project: the Videos page\'s order', async () => {
    listing('Peekaboo study', [item('a.mp4'), item('demo_clip.mp4'), item('c.mp4', { path: '/data/pk/more/c.mp4' })], 'project');
    open('project:d1');
    await lanesLoaded();
    expect(await screen.findByTestId('list-position')).toHaveTextContent('Peekaboo study · 2 of 3');
    fireEvent.click(screen.getByRole('button', { name: 'Next video' }));
    await waitFor(() => expect(lastPathAsked(api)).toBe('/data/pk/more/c.mp4'));
  });

  it('a folder, all videos, and not in a project', async () => {
    const asked = listing('/data/pk', [item('demo_clip.mp4'), item('p2.mp4')], 'folder');
    open('folder:/data/pk');
    await lanesLoaded();
    expect(await screen.findByTestId('list-position')).toHaveTextContent('pk · 1 of 2');
    expect(screen.getByRole('button', { name: 'Previous video' })).toBeDisabled();
    expect(asked).toHaveBeenCalledWith({ folder: '/data/pk' });
  });

  it.each([
    ['all', 'All videos', {}],
    ['unassigned', 'Not in a project', { notInProject: true }],
  ])('from=%s', async (from, name, query) => {
    const asked = listing('', [item('demo_clip.mp4'), item('p2.mp4')], 'all');
    open(from);
    await lanesLoaded();
    expect(await screen.findByTestId('list-position')).toHaveTextContent(`${name} · 1 of 2`);
    expect(asked).toHaveBeenCalledWith(query);
  });
});

describe('the header', () => {
  it('lists the video\'s projects when it wasn\'t opened from a list', async () => {
    api.getVideoRuns.mockResolvedValue(demoVideoRuns({ projects: [{ id: 'd1', name: 'Peekaboo study' }] }));
    renderVideoPage();
    await lanesLoaded();
    const link = within(screen.getByTestId('video-projects')).getByRole('link', { name: 'Peekaboo study' });
    expect(link).toHaveAttribute('href', '/videos?project=d1');
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
