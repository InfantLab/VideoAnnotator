/** The Videos page (spec 029 US1, FR-001, FR-002). */
import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { APIError } from '@/api/handleError';
import Videos from '@/pages/Videos';
import { EMPTY, PK, PROJECT_ID, VIDEOS, mockVideosApi, run, video, videosResponse } from '@/test/fixtures/videos';
import { landed, renderAt } from '@/test/fixtures/renderPage';

const show = (address = '/videos') => renderAt(address, [{ path: '/videos', element: <Videos /> }]);
const rows = () => screen.getAllByTestId('video-row');

afterEach(() => vi.restoreAllMocks());

describe('the left side', () => {
  it('lists all, not in a project, projects and folders with counts', async () => {
    mockVideosApi();
    show();
    const nav = await screen.findByRole('navigation', { name: 'Videos by place' });
    expect(within(nav).getByRole('link', { name: /All videos\s*212/ })).toHaveAttribute('aria-current', 'page');
    expect(within(nav).getByRole('link', { name: /Not in a project\s*37/ })).toHaveAttribute('href', '/videos?not_in_project=1');
    expect(within(nav).getByRole('link', { name: /Peekaboo study\s*48/ })).toHaveAttribute('href', `/videos?project=${PROJECT_ID}`);
    expect(within(nav).getByRole('link', { name: /Uploaded videos\s*3/ })).toHaveAttribute('href', '/videos?folder=uploads');
    expect(within(nav).getByRole('link', { name: /\/data\/pk\s*48/ })).toHaveAttribute('href', `/videos?folder=${encodeURIComponent(PK)}`);
  });

  it('shows the owner of projects that share a name', async () => {
    mockVideosApi({
      videos: (q) => {
        const base = videosResponse(q);
        return {
          ...base,
          counts: {
            ...base.counts,
            projects: [
              { id: 'a', name: 'QS', owner_name: 'caspar', count: 1, follows_folder: false, server_folder: null },
              { id: 'b', name: 'QS', owner_name: 'ana', count: 2, follows_folder: false, server_folder: null },
            ],
          },
        };
      },
    });
    show();
    expect(await screen.findByRole('link', { name: /QS \(caspar\)/ })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /QS \(ana\)/ })).toBeInTheDocument();
  });

  it("a folder offers the folders inside it", async () => {
    mockVideosApi();
    show(`/videos?folder=${encodeURIComponent(PK)}`);
    const inside = await screen.findByTestId('subfolders');
    expect(within(inside).getByRole('link', { name: 'site_a' })).toHaveAttribute('href', `/videos?folder=${encodeURIComponent(`${PK}/site_a`)}`);
  });

  it('asks for the scope in the address', async () => {
    const api = mockVideosApi();
    show(`/videos?folder=${encodeURIComponent(PK)}&recursive=1`);
    await screen.findAllByTestId('video-row');
    expect(api.listVideos).toHaveBeenCalledWith({ folder: PK, recursive: true });
  });
});

describe('the table', () => {
  it('has a column per pipeline with each result, and the reason on hover', async () => {
    mockVideosApi();
    show();
    await screen.findAllByTestId('video-row');
    const header = screen.getAllByRole('columnheader').map((h) => h.textContent);
    // Named by what runs (D1, UAT #9); the task is on hover.
    expect(header).toEqual(['', 'Video', 'Folder', 'Projects', 'DeepFace', 'Whisper large-v3-turbo']);
    expect(screen.getByRole('columnheader', { name: 'DeepFace' })).toHaveAttribute('title', 'DeepFace · where they are and their expression');
    const first = rows()[0];
    // What was found, not a tick (UAT #10).
    expect(await within(first).findByLabelText('DeepFace: 2 faces (most in one frame)')).toHaveTextContent('2 faces');
    expect(within(first).getByLabelText('Whisper large-v3-turbo: failed: No audio track')).toHaveTextContent('✕ failed');
    expect(within(rows()[1]).getByLabelText('DeepFace: running')).toHaveTextContent('●');
    expect(within(rows()[3]).getByLabelText('DeepFace: not run')).toHaveTextContent('–');
  });

  it('marks a video whose file is gone, keeping its results', async () => {
    mockVideosApi();
    show();
    await screen.findAllByTestId('video-row');
    expect(within(rows()[2]).getByText('not on the server')).toBeInTheDocument();
    expect(await within(rows()[2]).findByLabelText(/^DeepFace: 2 faces/)).toBeInTheDocument();
  });

  it('shows a video in two projects with both chips', async () => {
    mockVideosApi();
    show();
    await screen.findAllByTestId('video-row');
    expect(within(rows()[0]).getByRole('link', { name: 'Peekaboo study' })).toHaveAttribute('href', `/videos?project=${PROJECT_ID}`);
    // The second on hover, so the row stays one line (UAT #45).
    expect(within(rows()[0]).getByText('+1').parentElement).toHaveAttribute('title', 'Peekaboo study, QS2');
  });

  it("in a project, shows only a video's other projects", async () => {
    mockVideosApi();
    show(`/videos?project=${PROJECT_ID}`);
    await screen.findAllByTestId('video-row');
    expect(screen.getByRole('columnheader', { name: 'Also in' })).toBeInTheDocument();
    expect(within(rows()[0]).getByRole('link', { name: 'QS2' })).toBeInTheDocument();
    expect(within(rows()[0]).queryByRole('link', { name: 'Peekaboo study' })).toBeNull();
  });

  it('sorts by what a column found, failed and not run last (UAT #10)', async () => {
    mockVideosApi();
    show();
    await screen.findAllByTestId('video-row');
    await within(rows()[0]).findByText('2 faces');
    fireEvent.click(within(screen.getByRole('columnheader', { name: 'Whisper large-v3-turbo' })).getByRole('button'));
    expect(screen.getByRole('columnheader', { name: 'Whisper large-v3-turbo' })).toHaveAttribute('aria-sort', 'descending');
    // Only p01 ran Whisper, and it failed: it comes before the three never run.
    expect(rows()[0]).toHaveTextContent('p01_t1.mp4');
    fireEvent.click(within(screen.getByRole('columnheader', { name: 'Video' })).getByRole('button'));
    expect(rows().map((r) => within(r).getByTestId('file-name').textContent)).toEqual([
      'p01_t1.mp4',
      'p02_t1.mp4',
      'p03_t1.mp4',
      'p04_t1.mp4',
    ]);
  });

  it('keeps both ends of a long name, and says when an upload came (UAT #7, #8)', async () => {
    const long = (n: number) =>
      video(`2UWdXP.joke1.rep${n}.take1.Peekaboo_h265.mp4`, { folder: 'uploads', added_at: '2026-09-26T19:26:48Z' });
    mockVideosApi({ videos: (q) => videosResponse(q, { videos: [long(2), long(3)], total: 2 }) });
    show();
    await screen.findAllByTestId('video-row');
    const names = screen.getAllByTestId('file-name');
    expect(names[0].lastElementChild).toHaveTextContent('.Peekaboo_h265.mp4');
    expect(names[0].firstElementChild).toHaveTextContent('2UWdXP.joke1.rep2.take1');
    expect(rows()[0]).toHaveTextContent('Uploaded 26 Sep');
  });

  it('opens a row on the Video page, moving through the same list', async () => {
    mockVideosApi();
    const view = show(`/videos?project=${PROJECT_ID}`);
    await screen.findAllByTestId('video-row');
    fireEvent.click(rows()[1]);
    const where = new URL(landed(view.container)!.address, 'http://x');
    expect(where.pathname).toBe('/video');
    expect(where.searchParams.get('path')).toBe(`${PK}/p02_t1.mp4`);
    expect(where.searchParams.get('from')).toBe(`project:${PROJECT_ID}`);
  });

  it('filters by name; / focuses the filter; Space selects and Enter opens', async () => {
    mockVideosApi();
    const view = show();
    await screen.findAllByTestId('video-row');
    await userEvent.keyboard('/');
    expect(screen.getByLabelText('Filter videos by name')).toHaveFocus();
    await userEvent.keyboard('p03');
    await waitFor(() => expect(rows()).toHaveLength(1));
    rows()[0].focus();
    await userEvent.keyboard(' ');
    expect(screen.getByTestId('selection-bar')).toHaveTextContent('1 selected');
    await userEvent.keyboard('{Enter}');
    expect(landed(view.container)!.address).toMatch(/^\/video\?path=/);
  });

  it('draws only the rows on screen past 200', async () => {
    const many = Array.from({ length: 2000 }, (_, i) => video(`v${String(i).padStart(4, '0')}.mp4`));
    mockVideosApi({ videos: (q) => ({ ...videosResponse(q), videos: many, total: many.length }) });
    show();
    await screen.findAllByTestId('video-row');
    expect(rows().length).toBeLessThan(100);
  });
});

describe('empty, error and no server', () => {
  it('says what to do when there are no videos', async () => {
    mockVideosApi({ videos: () => EMPTY });
    show();
    expect(await screen.findByTestId('videos-empty')).toHaveTextContent('No videos yet');
  });

  it('says it needs a server when there is none', async () => {
    const api = mockVideosApi();
    api.listVideos.mockRejectedValue(new APIError('Network error: Failed to fetch', 0));
    show();
    expect(await screen.findByTestId('needs-server')).toBeInTheDocument();
  });

  it('words an error from the server', async () => {
    const api = mockVideosApi();
    api.listVideos.mockRejectedValue(new APIError('Dataset x not found', 404, undefined, { error: { code: 'DATASET_NOT_FOUND' } }));
    show('/videos?project=x');
    expect(await screen.findByRole('alert')).toHaveTextContent("That project isn't on this server any more.");
  });
});

describe('progress while a run is going (T032)', () => {
  it("shows a project's active run in its header", async () => {
    mockVideosApi();
    show(`/videos?project=${PROJECT_ID}`);
    const progress = await screen.findByTestId('scope-progress');
    expect(progress).toHaveTextContent('Peekaboo faces: running 31/48');
    expect(within(progress).getByRole('link')).toHaveAttribute('href', '/runs/b2');
  });

  it("sums a folder's rows still moving", async () => {
    mockVideosApi();
    show(`/videos?folder=${encodeURIComponent(PK)}`);
    expect(await screen.findByTestId('scope-progress')).toHaveTextContent('1 video running');
  });

  it('polls while anything moves', async () => {
    const api = mockVideosApi();
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      show(`/videos?project=${PROJECT_ID}`);
      await screen.findAllByTestId('video-row');
      const before = api.listVideos.mock.calls.length;
      await vi.advanceTimersByTimeAsync(5_500);
      await waitFor(() => expect(api.listVideos.mock.calls.length).toBeGreaterThan(before));
    } finally {
      vi.useRealTimers();
    }
  });
});

it('does not list a run on the left', async () => {
  mockVideosApi();
  show();
  const nav = await screen.findByRole('navigation', { name: 'Videos by place' });
  expect(within(nav).queryByText(run().name)).toBeNull();
  expect(VIDEOS.length).toBe(4);
});

describe('status in place (UAT #6, #43)', () => {
  it('a newer run going on shows over an older result, linked to the run', async () => {
    const rerun = video('p01_t1.mp4', {
      results: {
        face_analysis: { state: 'done', run_id: 'b1', job_id: 'j1', finished_at: '2026-10-12T10:00:00Z', error: null, now: 'running', now_run_id: 'b2' },
      },
    });
    mockVideosApi({ videos: (q) => videosResponse(q, { videos: [rerun], total: 1 }) });
    show();
    const cell = await screen.findByTestId('result-now');
    expect(cell).toHaveTextContent('running');
    expect(cell).toHaveAttribute('href', '/runs/b2');
  });

  it('the sidebar folds away', async () => {
    mockVideosApi();
    show();
    await userEvent.click(await screen.findByTestId('sidebar-toggle'));
    expect(screen.getByRole('button', { name: 'Show folders and projects' })).toHaveAttribute('aria-expanded', 'false');
    await userEvent.click(screen.getByTestId('sidebar-toggle'));
    expect(screen.getByRole('button', { name: 'Hide folders and projects' })).toBeInTheDocument();
  });
});
