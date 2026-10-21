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
    expect(header).toEqual(['', 'Video', 'Folder', 'Projects', 'Faces', 'Speech']);
    const first = rows()[0];
    expect(within(first).getByLabelText('Faces: done')).toHaveTextContent('✓');
    expect(within(first).getByLabelText('Speech: failed: No audio track')).toHaveTextContent('✕');
    expect(within(rows()[1]).getByLabelText('Faces: running')).toHaveTextContent('●');
    expect(within(rows()[3]).getByLabelText('Faces: not run')).toHaveTextContent('–');
  });

  it('marks a video whose file is gone, keeping its results', async () => {
    mockVideosApi();
    show();
    await screen.findAllByTestId('video-row');
    expect(within(rows()[2]).getByText('not on the server')).toBeInTheDocument();
    expect(within(rows()[2]).getByLabelText('Faces: done')).toBeInTheDocument();
  });

  it('shows a video in two projects with both chips', async () => {
    mockVideosApi();
    show();
    await screen.findAllByTestId('video-row');
    expect(within(rows()[0]).getByRole('link', { name: 'Peekaboo study' })).toHaveAttribute('href', `/videos?project=${PROJECT_ID}`);
    expect(within(rows()[0]).getByRole('link', { name: 'QS2' })).toBeInTheDocument();
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
    expect(within(progress).getByRole('link')).toHaveAttribute('href', '/videos?run=b2');
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
