/** The Runs page (UAT D2, #1). */
import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import Runs from '@/pages/Runs';
import { listedRun, mockVideosApi } from '@/test/fixtures/videos';
import { landed, renderAt } from '@/test/fixtures/renderPage';

const show = () => renderAt('/runs', [{ path: '/runs', element: <Runs /> }]);

afterEach(() => vi.restoreAllMocks());

describe('Runs', () => {
  it('one row per run: state, name, videos, outcome, and why one failed', async () => {
    const failed = listedRun({
      id: 'b1',
      name: 'First faces',
      active: false,
      running: 0,
      waiting: 0,
      done: 46,
      failed: 2,
      project: null,
      total: 48,
      finished_at: '2026-10-20T10:12:00Z',
      failure: 'No faces model',
    });
    mockVideosApi({ runs: [listedRun(), failed] });
    show();
    const rows = await screen.findAllByTestId('run-row');
    expect(rows).toHaveLength(2);
    expect(rows[0]).toHaveTextContent('Peekaboo faces');
    expect(rows[0]).toHaveTextContent('48 videos in Peekaboo study');
    expect(within(rows[0]).getByRole('img', { name: 'Running' })).toBeInTheDocument();
    expect(within(rows[0]).getByRole('progressbar')).toBeInTheDocument();
    expect(within(rows[1]).getByTestId('run-outcome')).toHaveTextContent('46 done, 2 failed why?');
    expect(within(rows[1]).getByTitle('No faces model')).toBeInTheDocument();
    expect(rows[1]).toHaveTextContent('48 videos: p01_t1.mp4, p02_t1.mp4 +46');
    expect(within(rows[1]).queryByRole('progressbar')).toBeNull();
  });

  it('a row opens its run', async () => {
    mockVideosApi({ runs: [listedRun()] });
    const view = show();
    await userEvent.click((await screen.findAllByTestId('run-row'))[0]);
    await waitFor(() => expect(landed(view.container)?.address).toBe('/runs/b2'));
  });

  it('with no runs, says so and offers to start one', async () => {
    mockVideosApi();
    show();
    expect(await screen.findByTestId('runs-empty')).toHaveTextContent('No runs yet.');
    expect(screen.getByRole('link', { name: /Run pipelines on videos/ })).toHaveAttribute('href', '/run');
  });
});
