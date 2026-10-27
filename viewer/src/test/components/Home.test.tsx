/** Home: the start points, the demos and what's left to set up (spec 029 US7, T058). */
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import Home from '@/pages/Home';
import { CATALOG, EMPTY, listedRun, mockVideosApi, videosResponse } from '@/test/fixtures/videos';
import { landed, renderAt } from '@/test/fixtures/renderPage';

const server = vi.hoisted(() => ({ capabilities: { version: '1.7.0' } as { version: string; authRequired?: boolean } | null }));
vi.mock('@/contexts/ServerCapabilitiesContext', () => ({
  useServerCapabilitiesContext: () => ({ capabilities: server.capabilities, isLoading: false }),
}));

const show = () => renderAt('/home', [{ path: '/home', element: <Home /> }]);

describe('Home', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    server.capabilities = { version: '1.7.0' };
  });

  it('offers three start points', async () => {
    mockVideosApi();
    const view = show();
    const points = await screen.findByTestId('start-points');
    expect(within(points).getAllByRole('button').map((b) => b.querySelector('.font-semibold')?.textContent)).toEqual([
      'Ask AI about a video',
      'Run pipelines on videos',
      'Open files from this computer',
    ]);
    await userEvent.click(within(points).getByRole('button', { name: /Run pipelines on videos/ }));
    expect(landed(view.container)?.address).toBe('/run');
  });

  it('shows the setup checklist on a new server', async () => {
    const api = mockVideosApi({ videos: () => EMPTY });
    api.getPipelineCatalog.mockResolvedValue({ ...CATALOG, catalog: { pipelines: [{ id: 'a', name: 'A', available: false }] } });
    show();
    const checklist = await screen.findByTestId('setup-checklist');
    expect(checklist).toHaveTextContent('Install at least one pipeline');
    expect(within(checklist).getByRole('link', { name: 'Videos' })).toHaveAttribute('href', '/videos');
  });

  it('points to Videos once there are videos', async () => {
    mockVideosApi({ videos: (q) => videosResponse(q) });
    show();
    expect(await screen.findByRole('link', { name: /All 212 videos/ })).toHaveAttribute('href', '/videos');
    expect(screen.getAllByTestId('recent-video').length).toBeGreaterThan(0);
  });

  it('shows the runs going on and just finished, then all runs (D3)', async () => {
    const going = listedRun();
    const old = listedRun({ id: 'b0', name: 'Long ago', active: false, running: 0, waiting: 0, done: 48, finished_at: '2026-01-01T10:00:00Z' });
    mockVideosApi({ videos: (q) => videosResponse(q), runs: [going, old] });
    show();
    const rows = await screen.findAllByTestId('run-row');
    expect(rows).toHaveLength(1);
    expect(rows[0]).toHaveTextContent('Peekaboo faces');
    expect(screen.getByRole('link', { name: /All runs/ })).toHaveAttribute('href', '/runs');
  });

  it('with nothing going on, shows the last run', async () => {
    const old = listedRun({ id: 'b0', name: 'Long ago', active: false, running: 0, waiting: 0, done: 48, finished_at: '2026-01-01T10:00:00Z' });
    mockVideosApi({ videos: (q) => videosResponse(q), runs: [old] });
    show();
    expect(await screen.findByText('Nothing running. Your last run:')).toBeInTheDocument();
    expect(screen.getByTestId('run-row')).toHaveTextContent('Long ago');
  });

  it("doesn't claim a connection the key hasn't made (UAT #38)", async () => {
    server.capabilities = { version: '1.7.0', authRequired: true };
    mockVideosApi();
    show();
    expect(screen.getByText('The server needs your key')).toBeInTheDocument();
    expect(screen.queryByText('Server connected')).not.toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Connect with your key' })).toHaveAttribute('href', '/settings');
    expect(screen.getByRole('button', { name: /Ask AI about a video/ })).toBeDisabled();
  });

  it('works without a server: open files and demos', async () => {
    server.capabilities = null;
    mockVideosApi();
    show();
    expect(screen.getByRole('button', { name: /Open files from this computer/ })).toBeEnabled();
    expect(screen.getByRole('button', { name: /Ask AI about a video/ })).toBeDisabled();
    expect(screen.getByRole('link', { name: /^Peekaboo/ })).toHaveAttribute('href', '/viewer?demo=peekaboo-rep3-v1.1.1');
  });
});
