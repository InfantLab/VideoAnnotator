/** Demos open as files in the standalone Video page (spec 029 R14, T045). */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import Index from '@/pages/Index';
import { renderAt } from '@/test/fixtures/renderPage';
import { FIXTURES, mockVideoApi, wideLayout } from '@/test/videoPageHarness';
import { DEMO_DATA_SETS } from '@/utils/debugUtils';

// The file picker sits inside the app's nav, which asks the server about itself.
vi.mock('@/contexts/ServerCapabilitiesContext', () => ({
  useServerCapabilitiesContext: () => ({ capabilities: null, isLoading: false, error: null, refresh: vi.fn() }),
  useServerVersion: () => null,
}));

// jsdom can't read a video's metadata, which the merger asks the browser for.
vi.mock('@/lib/parsers/merger', async (original) => ({
  ...(await original<typeof import('@/lib/parsers/merger')>()),
  mergeAnnotationData: vi.fn(async () => ({
    data: {
      video_info: { filename: 'demo.mp4', duration: 20, width: 640, height: 360, frame_rate: 25 },
      person_tracking: [],
      speech_recognition: [],
      speaker_diarization: [],
      scene_detection: [],
    },
    warnings: [],
    errors: [],
  })),
}));

const files: Record<string, string> = {
  vtt: readFileSync(path.join(FIXTURES, 'demo_clip_speech_recognition.vtt'), 'utf8'),
  rttm: readFileSync(path.join(FIXTURES, 'demo_clip_speaker_diarization.rttm'), 'utf8'),
};

function serveDemo() {
  URL.createObjectURL = vi.fn(() => 'blob:demo');
  URL.revokeObjectURL = vi.fn();
  return vi.spyOn(global, 'fetch').mockImplementation(async (input) => {
    const url = String(input);
    const body = url.endsWith('.vtt') ? files.vtt : url.endsWith('.rttm') ? files.rttm : url.endsWith('.json') ? '{}' : 'video';
    return new Response(body, { status: 200 });
  });
}

const show = (address: string) => renderAt(address, [{ path: '/viewer', element: <Index /> }]);

afterEach(() => vi.restoreAllMocks());

describe('/viewer?demo=', () => {
  it("fetches the demo's files and opens them on the Video page", async () => {
    mockVideoApi();
    wideLayout();
    const fetched = serveDemo();
    show('/viewer?demo=peekaboo-rep3-v1.1.1');

    expect(await screen.findAllByTestId('lane-label', {}, { timeout: 5000 })).not.toHaveLength(0);
    const urls = fetched.mock.calls.map(([u]) => String(u));
    expect(urls).toContain(DEMO_DATA_SETS['peekaboo-rep3-v1.1.1'].video);
    expect(urls).toContain(DEMO_DATA_SETS['peekaboo-rep3-v1.1.1'].webvtt);
  });

  it('says so when the demo cannot be fetched', async () => {
    vi.spyOn(global, 'fetch').mockRejectedValue(new TypeError('Failed to fetch'));
    show('/viewer?demo=peekaboo-rep3-v1.1.1');
    expect(await screen.findByRole('alert')).toHaveTextContent('internet connection');
  });

  it('offers the demos beside opening files', async () => {
    serveDemo();
    mockVideoApi();
    wideLayout();
    show('/viewer');
    expect(screen.getByRole('heading', { name: 'Open files from this computer' })).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Peekaboo' }));
    await waitFor(() => expect(screen.queryByRole('heading', { name: 'Open files from this computer' })).toBeNull());
  });
});
