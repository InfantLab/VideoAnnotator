/** Every address, old and new, lands somewhere (spec 029 R13, SC-006; R18). */
import { render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { getApiClient } from '@/api/client';
import App from '@/App';
import { queryClient } from '@/lib/queryClient';
import { EMPTY, mockVideosApi } from '@/test/fixtures/videos';

const server = vi.hoisted(() => ({ capabilities: { version: '1.7.0' } as { version: string } | null }));
vi.mock('@/contexts/ServerCapabilitiesContext', () => ({
  useServerCapabilitiesContext: () => ({ capabilities: server.capabilities, isLoading: false, error: null, refresh: vi.fn() }),
  useServerVersion: () => '1.7.0',
}));
vi.mock('@/contexts/ServerCapabilitiesProvider', () => ({ ServerCapabilitiesProvider: ({ children }: { children: unknown }) => children }));

const client = () => Object.getPrototypeOf(getApiClient());

function visit(address: string, state?: unknown) {
  window.history.pushState(state ?? null, '', address);
  render(<App />);
}
const where = () => window.location.pathname + window.location.search;

beforeEach(() => {
  server.capabilities = { version: '1.7.0' };
  mockVideosApi();
  vi.spyOn(client(), 'getJob').mockImplementation(async (id: string) => {
    if (id === 'gone') throw new Error('not found');
    return { id, video_path: '/data/pk/p01_t1.mp4', batch_id: null };
  });
  vi.spyOn(client(), 'getVideoRuns').mockRejectedValue(new Error('not in this test'));
});
afterEach(() => {
  vi.restoreAllMocks();
  queryClient.clear();
  window.history.pushState(null, '', '/');
});

describe('old addresses', () => {
  it.each([
    ['/jobs', '/videos'],
    ['/batches', '/videos'],
    ['/batches/b1', '/runs/b1'],
    ['/datasets', '/videos'],
    ['/results', '/videos'],
    ['/library', '/videos'],
    ['/getting-started', '/help'],
    ['/jobs/new', '/run'],
    ['/jobs/j1', `/video?path=${encodeURIComponent('/data/pk/p01_t1.mp4')}&run=j1`],
    ['/view/j1', `/video?path=${encodeURIComponent('/data/pk/p01_t1.mp4')}&run=j1`],
    ['/jobs/gone', '/videos'],
    ['/prompts', '/ask?history=open'],
    ['/workbench', '/ask'],
  ])('%s → %s', async (from, to) => {
    visit(from);
    await waitFor(() => expect(where()).toBe(to));
  });

  it('/jobs/new keeps the old wizard state for New run', async () => {
    visit('/jobs/new', { usr: { wizardStart: { mode: 'dataset', datasetId: 'd1', name: 'QS' } }, key: 'x', idx: 0 });
    await waitFor(() => expect(where()).toBe('/run'));
    expect((window.history.state as { usr?: unknown })?.usr).toEqual({ wizardStart: { mode: 'dataset', datasetId: 'd1', name: 'QS' } });
  });
});

describe('new addresses', () => {
  it.each([
    ['/home', 'start-points'],
    ['/videos', 'scope-header'],
    ['/help', 'glossary'],
  ])('%s', async (address, testId) => {
    visit(address);
    expect(await screen.findByTestId(testId, {}, { timeout: 5000 })).toBeInTheDocument();
  });
});

describe('the root (R18)', () => {
  it('opens Home, its dashboard, even once there are results (D3)', async () => {
    visit('/');
    expect(await screen.findByTestId('start-points')).toBeInTheDocument();
    expect(where()).toBe('/');
  });

  it('opens Home on a new server', async () => {
    vi.spyOn(client(), 'listVideos').mockResolvedValue(EMPTY);
    visit('/');
    expect(await screen.findByTestId('start-points')).toBeInTheDocument();
    expect(where()).toBe('/');
  });

  it('opens Home without a server', async () => {
    server.capabilities = null;
    visit('/');
    expect(await screen.findByTestId('start-points')).toBeInTheDocument();
  });
});
