/** The activity indicator and its messages (spec 029 R17, FR-018, FR-025). */
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ActivityIndicator } from '@/components/ActivityIndicator';
import { ACTIVITY, activityRun, mockVideosApi } from '@/test/fixtures/videos';
import { renderAt } from '@/test/fixtures/renderPage';

const toast = vi.hoisted(() => Object.assign(vi.fn(), { success: vi.fn(), error: vi.fn() }));
vi.mock('sonner', () => ({ toast }));

const show = () => renderAt('/', [{ path: '/', element: <ActivityIndicator /> }]);

beforeEach(() => vi.clearAllMocks());
afterEach(() => vi.restoreAllMocks());

describe('ActivityIndicator', () => {
  it('shows the runs going on, linking to each', async () => {
    mockVideosApi({ activity: ACTIVITY });
    show();
    const button = await screen.findByRole('button', { name: 'Runs going on: 1 run · 31/48' });
    await userEvent.click(button);
    expect(await screen.findByRole('link', { name: /Peekaboo faces/ })).toHaveAttribute('href', '/videos?run=b2');
    expect(screen.getByRole('link', { name: /First faces/ })).toHaveAttribute('href', '/videos?run=b1');
  });

  it('says when a run finished, with the results table next', async () => {
    mockVideosApi({ activity: { active: [], recent: [activityRun({ id: 'b7', name: 'QS faces', done: 3, total: 3, running: 0, waiting: 0, finished_at: '2026-10-20T10:00:00Z' })] } });
    show();
    await waitFor(() =>
      expect(toast.success).toHaveBeenCalledWith(
        'QS faces finished',
        expect.objectContaining({ action: expect.objectContaining({ label: 'Results table' }) }),
      ),
    );
  });

  it('says when a run failed or was stopped', async () => {
    mockVideosApi({
      activity: {
        active: [],
        recent: [
          activityRun({ id: 'b8', name: 'All failed', done: 0, failed: 2, total: 2, running: 0, waiting: 0, finished_at: 'x' }),
          activityRun({ id: 'b9', name: 'Halted', stopped: true, cancelled: 2, running: 0, waiting: 0, finished_at: 'x' }),
        ],
      },
    });
    show();
    await waitFor(() => expect(toast.error).toHaveBeenCalledWith('All failed failed', expect.anything()));
    expect(toast).toHaveBeenCalledWith('Halted stopped', expect.anything());
  });

  it('is quiet when nothing is going on', async () => {
    mockVideosApi();
    show();
    await userEvent.click(await screen.findByRole('button', { name: 'Runs' }));
    expect(screen.getByText('No runs going on.')).toBeInTheDocument();
  });
});
