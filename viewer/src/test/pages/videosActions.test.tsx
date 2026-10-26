/** Run… and Export for a scope or a selection (spec 029 FR-003, T026). */
import { fireEvent, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import Videos from '@/pages/Videos';
import { PK, PROJECT_ID, mockVideosApi } from '@/test/fixtures/videos';
import { landed, renderAt } from '@/test/fixtures/renderPage';

const show = (address = '/videos') => renderAt(address, [{ path: '/videos', element: <Videos /> }]);

vi.mock('@/lib/exports', async (original) => ({ ...(await original<typeof import('@/lib/exports')>()), saveBlob: vi.fn() }));

afterEach(() => vi.restoreAllMocks());

describe('Run… opens New run on what is shown', () => {
  it.each([
    [`/videos?project=${PROJECT_ID}`, { kind: 'project', id: PROJECT_ID, name: 'Peekaboo study' }],
    [`/videos?folder=${encodeURIComponent(PK)}`, { kind: 'folder', path: PK, recursive: false }],
  ])('%s', async (address, target) => {
    mockVideosApi();
    const view = show(address);
    await screen.findAllByTestId('video-row');
    fireEvent.click(within(screen.getByTestId('scope-header')).getByRole('button', { name: /Run on/ }));
    const where = landed(view.container)!;
    expect(where.address).toBe('/run');
    expect(where.state).toEqual({ runStart: { target } });
  });

  it('all videos: the ones on the server', async () => {
    mockVideosApi();
    const view = show();
    await screen.findAllByTestId('video-row');
    fireEvent.click(within(screen.getByTestId('scope-header')).getByRole('button', { name: /Run on/ }));
    const target = (landed(view.container)!.state as { runStart: { target: { kind: string; paths: string[] } } }).runStart.target;
    expect(target.kind).toBe('videos');
    expect(target.paths).toEqual([`${PK}/p01_t1.mp4`, `${PK}/p02_t1.mp4`, `${PK}/p04_t1.mp4`]);
  });
});

describe('Export for what is shown', () => {
  it("exports a project's latest results", async () => {
    const api = mockVideosApi();
    show(`/videos?project=${PROJECT_ID}`);
    await screen.findAllByTestId('video-row');
    await userEvent.click(within(screen.getByTestId('scope-header')).getByRole('button', { name: /Export/ }));
    await userEvent.click(within(await screen.findByTestId('export-panel')).getByRole('button', { name: 'Results table (CSV)' }));
    expect(api.exportResults).toHaveBeenCalledWith({ kind: 'project', id: PROJECT_ID, results: 'latest' }, 'table', 'csv');
  });
});

describe('the selection bar', () => {
  it('runs and exports exactly the rows selected', async () => {
    const api = mockVideosApi();
    const view = show(`/videos?project=${PROJECT_ID}`);
    await screen.findAllByTestId('video-row');
    await userEvent.click(screen.getByRole('checkbox', { name: 'Select p01_t1.mp4' }));
    await userEvent.click(screen.getByRole('checkbox', { name: 'Select p04_t1.mp4' }));
    const bar = screen.getByTestId('selection-bar');
    expect(bar).toHaveTextContent('2 selected');
    // One Run while rows are selected, and it says its scope (UAT #42, P5).
    expect(within(screen.getByTestId('scope-header')).queryByRole('button', { name: /Run on/ })).toBeNull();
    expect(within(bar).getByRole('button', { name: /Run on/ })).toHaveTextContent('Run on 2 videos…');

    await userEvent.click(within(bar).getByRole('button', { name: /Export/ }));
    await userEvent.click(within(await screen.findByTestId('export-panel')).getByRole('button', { name: 'Results table (CSV)' }));
    expect(api.exportResults).toHaveBeenCalledWith(
      { kind: 'videos', paths: [`${PK}/p01_t1.mp4`, `${PK}/p04_t1.mp4`], results: 'latest' },
      'table',
      'csv',
    );

    fireEvent.click(within(bar).getByRole('button', { name: /Run on/ }));
    expect(landed(view.container)!.state).toEqual({
      runStart: {
        target: {
          kind: 'videos',
          paths: [`${PK}/p01_t1.mp4`, `${PK}/p04_t1.mp4`],
          label: '2 videos',
          projectId: PROJECT_ID,
          projectName: 'Peekaboo study',
        },
      },
    });
  });

  it('selects every row shown, and clears', async () => {
    mockVideosApi();
    show();
    await screen.findAllByTestId('video-row');
    await userEvent.click(screen.getByRole('checkbox', { name: 'Select every video shown' }));
    expect(screen.getByTestId('selection-bar')).toHaveTextContent('4 selected');
    await userEvent.click(screen.getByRole('button', { name: 'Clear the selection' }));
    expect(screen.queryByTestId('selection-bar')).toBeNull();
  });
});
