/** Spec 028 US4: plays at once, no folder question. */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { getApiClient } from '@/api/client';
import { APIError } from '@/api/handleError';
import { LocalVideo } from '@/components/video/LocalVideo';
import { JobRedirect } from '@/components/Redirects';
import {
  FIXTURES,
  VIDEO_PATH,
  demoRun,
  demoVideoRuns,
  lanesLoaded,
  mockVideoApi,
  renderVideoPage,
  wideLayout,
} from '@/test/videoPageHarness';
import type { StandardAnnotationData } from '@/types/annotations';

// The page's error message keeps the app's nav, which reads the connection.
vi.mock('@/contexts/ServerCapabilitiesContext', () => ({
  useServerCapabilitiesContext: () => ({ capabilities: { version: '1.7.0' }, isLoading: false, error: null, refresh: vi.fn() }),
}));

afterEach(() => vi.restoreAllMocks());

const fixture = (name: string) => new File([readFileSync(path.join(FIXTURES, name), 'utf8')], name);

describe('loading a video from the server', () => {
  it('streams the video: no zip, no folder question', async () => {
    const api = mockVideoApi();
    wideLayout();
    const { container } = renderVideoPage();
    await lanesLoaded();
    expect(container.querySelector('video')?.getAttribute('src')).toMatch(/\/api\/v1\/videos\/stream\?path=.+&token=tok/);
    expect(api.streamToken).toHaveBeenCalledTimes(1);
    const fetched = vi.mocked(global.fetch).mock.calls.map(([url]) => String(url));
    expect(fetched.some((u) => u.includes('/artifacts'))).toBe(false);
    expect(screen.queryByText(/Choose a folder/i)).toBeNull();
  });

  it('fetches only the chosen runs\' result and provenance files', async () => {
    const older = demoRun('job-1', { created_at: '2026-10-01T09:00:00Z', completed_at: '2026-10-01T09:30:00Z' });
    const newer = demoRun('job-2', { created_at: '2026-10-12T09:00:00Z' });
    newer.pipelines = newer.pipelines.map((p) => (p.name === 'face_analysis' ? p : { ...p, status: 'failed', error: 'stopped' }));
    const api = mockVideoApi(demoVideoRuns({ runs: [newer, older] }));
    wideLayout();
    renderVideoPage();
    await lanesLoaded();
    const urls = api.downloadResultFile.mock.calls.map(([url]) => String(url));
    expect(urls.filter((u) => u.includes('/face_analysis'))).toEqual([expect.stringContaining('/jobs/job-2/')]);
    expect(urls.filter((u) => u.includes('/jobs/job-1/')).length).toBe(7);
    // Files that restate another (person_tracks, openface3_detailed) aren't fetched.
    expect(urls.some((u) => /person_tracks|openface3_detailed/.test(u))).toBe(false);
  });

  it('gets a new token when the stream says 401, keeping the time and play state', async () => {
    const api = mockVideoApi();
    api.streamToken
      .mockResolvedValueOnce({ token: 'old', expires_at: new Date(Date.now() + 3600_000).toISOString(), path: VIDEO_PATH, name: 'demo_clip.mp4', duration_sec: 20 })
      .mockResolvedValueOnce({ token: 'new', expires_at: new Date(Date.now() + 3600_000).toISOString(), path: VIDEO_PATH, name: 'demo_clip.mp4', duration_sec: 20 });
    vi.mocked(global.fetch).mockResolvedValue(new Response(null, { status: 401 }));
    wideLayout();
    const { container } = renderVideoPage();
    await lanesLoaded();
    const video = container.querySelector('video')!;
    await waitFor(() => expect(video.getAttribute('src')).toContain('token=old'));
    video.currentTime = 7;
    Object.defineProperty(video, 'paused', { configurable: true, value: false });
    const play = vi.spyOn(video, 'play');
    await act(async () => {
      video.dispatchEvent(new Event('error'));
    });
    await waitFor(() => expect(video.getAttribute('src')).toContain('token=new'));
    video.currentTime = 0;
    act(() => {
      video.dispatchEvent(new Event('loadedmetadata'));
    });
    expect(video.currentTime).toBe(7);
    expect(play).toHaveBeenCalled();
  });

  it('renews a token within five minutes of its expiry', async () => {
    const api = mockVideoApi();
    const soon = { token: 'soon', expires_at: new Date(Date.now() + 4 * 60_000).toISOString(), path: VIDEO_PATH, name: 'demo_clip.mp4', duration_sec: 20 };
    const later = { ...soon, token: 'later', expires_at: new Date(Date.now() + 3600_000).toISOString() };
    api.streamToken.mockResolvedValueOnce(soon).mockResolvedValueOnce(later);
    wideLayout();
    const { container } = renderVideoPage();
    await lanesLoaded();
    await waitFor(() => expect(container.querySelector('video')?.getAttribute('src')).toContain('token=later'));
    expect(api.streamToken).toHaveBeenCalledTimes(2);
  });

  it('shows the results with the missing-video message when the video is gone', async () => {
    const api = mockVideoApi(demoVideoRuns({ available: false, unavailable_reason: 'moved since the run' }));
    wideLayout();
    const { container } = renderVideoPage();
    await lanesLoaded();
    expect(screen.getByRole('alert')).toHaveTextContent('moved since the run');
    expect(container.querySelector('video')?.getAttribute('src')).toBeNull();
    expect(api.streamToken).not.toHaveBeenCalled();
    expect(screen.getAllByTestId('lane-label').length).toBeGreaterThan(5);
  });
});

function Where() {
  const location = useLocation();
  return <div data-testid="where">{location.pathname + location.search}</div>;
}

describe('old addresses', () => {
  it('sends /view/<job id> to the Video page, starting on that run', async () => {
    const client = Object.getPrototypeOf(getApiClient());
    vi.spyOn(client, 'getJob').mockResolvedValue({ id: 'job-9', video_path: VIDEO_PATH, batch_id: 'b1' });
    render(
      <QueryClientProvider client={new QueryClient()}>
        <MemoryRouter initialEntries={['/view/job-9']}>
          <Routes>
            <Route path="/view/:jobId" element={<JobRedirect />} />
            <Route path="/video" element={<Where />} />
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>,
    );
    const where = await screen.findByTestId('where');
    const url = new URL(where.textContent!, 'http://x');
    expect(url.pathname).toBe('/video');
    expect(url.searchParams.get('path')).toBe(VIDEO_PATH);
    expect(url.searchParams.get('run')).toBe('job-9');
    expect(url.searchParams.get('from')).toBe('run:b1');
  });

  it('goes to Videos when the run is not found', async () => {
    const client = Object.getPrototypeOf(getApiClient());
    vi.spyOn(client, 'getJob').mockRejectedValue(new Error('Job job-0 not found'));
    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <MemoryRouter initialEntries={['/view/job-0']}>
          <Routes>
            <Route path="/view/:jobId" element={<JobRedirect />} />
            <Route path="/videos" element={<Where />} />
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>,
    );
    expect(await screen.findByTestId('where')).toHaveTextContent('/videos');
  });
});

describe('standalone /viewer', () => {
  it('shows local files on the Video page, with no server', async () => {
    const api = mockVideoApi();
    wideLayout();
    const results = ['demo_clip_speech_recognition.vtt', 'demo_clip_speaker_diarization.rttm'].map(fixture);
    const data: StandardAnnotationData = { video_info: { filename: 'demo_clip.mp4', duration: 20, width: 0, height: 0 } };
    const video = new File(['video'], 'demo_clip.mp4', { type: 'video/mp4' });
    URL.createObjectURL = vi.fn(() => 'blob:local');
    URL.revokeObjectURL = vi.fn();
    render(
      <QueryClientProvider client={new QueryClient()}>
        <MemoryRouter>
          <LocalVideo loaded={{ video, results, data }} onClose={vi.fn()} />
        </MemoryRouter>
      </QueryClientProvider>,
    );
    await lanesLoaded();
    expect(new Set(screen.getAllByRole('group').map((g) => g.dataset.kind))).toEqual(new Set(['speech', 'speakers', 'notes']));
    expect(screen.getByTestId('lane-empty')).toHaveTextContent('Notes need a server');
    fireEvent.keyDown(window, { key: 'n' });
    expect(screen.queryByTestId('note-editor')).toBeNull();
    expect(screen.queryAllByTestId('lane-run')).toHaveLength(0);
    expect(api.getVideoRuns).not.toHaveBeenCalled();
    expect(api.streamToken).not.toHaveBeenCalled();
    expect(screen.queryByRole('link', { name: /Ask AI/ })).toBeNull();
    fireEvent.mouseDown(screen.getByRole('tab', { name: 'Notes' }));
    expect(await screen.findByTestId('notes-tab')).toHaveTextContent(/Notes need a server/);
  });
});

describe('empty and error states', () => {
  it('a video with no runs shows the video, an empty timeline and Run…', async () => {
    const api = mockVideoApi(demoVideoRuns({ runs: [] }));
    wideLayout();
    const { container } = renderVideoPage();
    expect(await screen.findByTestId('lanes-empty')).toHaveTextContent('No results for this video yet');
    expect(screen.getByRole('button', { name: /Run…/ })).toBeInTheDocument();
    await waitFor(() => expect(container.querySelector('video')?.getAttribute('src')).toContain('/api/v1/videos/stream'));
    expect(api.downloadResultFile).not.toHaveBeenCalled();
  });

  it("a failed pipeline's lane shows why it failed", async () => {
    const run = demoRun('job-1');
    run.pipelines = run.pipelines.map((p) =>
      p.name === 'scene_detection' ? { ...p, status: 'failed', error: 'CUDA out of memory', files: [] } : p,
    );
    mockVideoApi(demoVideoRuns({ runs: [run] }));
    wideLayout();
    renderVideoPage();
    await lanesLoaded();
    expect(screen.getByTestId('lane-failed')).toHaveTextContent('Failed: CUDA out of memory');
  });

  it("shows the server's message and hint when the video can't be opened", async () => {
    const api = mockVideoApi();
    api.getVideoRuns.mockRejectedValue(
      new APIError("/elsewhere/x.mp4 isn't in a folder shared with VideoAnnotator", 403, undefined, {
        error: { code: 'PATH_NOT_ALLOWED', hint: 'Upload the video, or choose one in a folder shared with the server.' },
      }),
    );
    renderVideoPage();
    expect(await screen.findByRole('alert')).toHaveTextContent("isn't shared with VideoAnnotator");
    expect(screen.getByText(/Upload the video, or choose one/)).toBeInTheDocument();
    // Not a dead end (UAT #61).
    expect(screen.getByRole('link', { name: /Videos/ })).toHaveAttribute('href', '/videos');
  });
});
