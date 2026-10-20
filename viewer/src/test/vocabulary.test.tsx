/**
 * No route shows the API's words: job, batch, dataset, preset (spec 029 R15,
 * SC-002). Each route is rendered loaded, empty and with the server failing;
 * visible text, titles, aria-labels and placeholders are read.
 */
import { render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { getApiClient } from '@/api/client';
import { APIError } from '@/api/handleError';
import App from '@/App';
import { queryClient } from '@/lib/queryClient';
import { EMPTY, PROJECT_ID, mockVideosApi } from '@/test/fixtures/videos';
import { mockVideoApi, VIDEO_PATH } from '@/test/videoPageHarness';

const server = vi.hoisted(() => ({ capabilities: { version: '1.7.0' } as { version: string } | null }));
vi.mock('@/contexts/ServerCapabilitiesContext', () => ({
  useServerCapabilitiesContext: () => ({ capabilities: server.capabilities, isLoading: false, error: null, refresh: vi.fn() }),
  useServerVersion: () => '1.7.0',
}));
vi.mock('@/contexts/ServerCapabilitiesProvider', () => ({ ServerCapabilitiesProvider: ({ children }: { children: unknown }) => children }));

const API_WORDS = /\b(jobs?|batch(es)?|datasets?|presets?)\b/i;

const ROUTES = [
  '/home',
  '/videos',
  `/videos?project=${PROJECT_ID}`,
  '/videos?run=b1',
  '/videos?folder=uploads',
  '/videos?not_in_project=1',
  '/run',
  `/video?path=${encodeURIComponent(VIDEO_PATH)}`,
  '/viewer',
  '/ask',
  '/pipelines',
  '/settings',
  '/help',
  '/compare',
  '/compare?a=j1&b=j2',
  '/no-such-page',
];

type State = 'loaded' | 'empty' | 'error';

function mockFor(state: State) {
  const client = Object.getPrototypeOf(getApiClient());
  if (state === 'error') {
    const failure = new APIError('Job j1 failed: dataset x', 500, undefined, { error: { code: 'INTERNAL_ERROR' } });
    for (const name of Object.getOwnPropertyNames(client)) {
      if (name === 'constructor' || typeof client[name] !== 'function') continue;
      if (/^(get|list|stream|validate|browse|scan|health|detailed|is)/.test(name)) vi.spyOn(client, name).mockRejectedValue(failure);
    }
    return;
  }
  mockVideoApi();
  const api = mockVideosApi(state === 'empty' ? { videos: () => EMPTY, presets: [], datasets: [] } : {});
  vi.spyOn(client, 'getCurrentUser').mockResolvedValue({ id: 'u1', is_admin: true });
  vi.spyOn(client, 'getExtrasGroups').mockResolvedValue([]);
  vi.spyOn(client, 'getIngestAccess').mockResolvedValue({ can_read_in_place: false, same_machine: false, reason: 'Not shared.' });
  vi.spyOn(client, 'getJob').mockResolvedValue({ id: 'j1', status: 'completed', selected_pipelines: ['vlm_annotation'], config: {}, video_path: VIDEO_PATH });
  return api;
}

function visibleWords(): string {
  const parts = [document.body.textContent ?? ''];
  for (const el of document.body.querySelectorAll('[title],[aria-label],[placeholder]')) {
    for (const attr of ['title', 'aria-label', 'placeholder']) {
      const value = el.getAttribute(attr);
      if (value) parts.push(value);
    }
  }
  return parts.join('\n');
}

afterEach(() => {
  vi.restoreAllMocks();
  queryClient.clear();
  window.history.pushState(null, '', '/');
});

describe.each<State>(['loaded', 'empty', 'error'])('%s', (state) => {
  it.each(ROUTES)('%s says none of the API words', async (route) => {
    server.capabilities = { version: '1.7.0' };
    mockFor(state);
    window.history.pushState(null, '', route);
    render(<App />);
    await screen.findByRole('contentinfo', {}, { timeout: 1500 }).catch(() => undefined);
    await new Promise((r) => setTimeout(r, 400));
    await waitFor(() => expect(document.body.textContent).not.toMatch(/Loading…$/));
    const found = visibleWords().match(new RegExp(API_WORDS.source, 'gi'));
    const where = found ? visibleWords().split('\n').filter((line) => API_WORDS.test(line)).slice(0, 5) : [];
    expect(where).toEqual([]);
  }, 15_000);
});
