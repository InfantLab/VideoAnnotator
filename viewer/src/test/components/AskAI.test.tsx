import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import AskAI from '@/pages/AskAI';
import { AskRedirect } from '@/components/ask/AskRedirect';
import { apiClient } from '@/api/client';
import { APIError } from '@/api/handleError';
import type { AskAnswer, AskEntry } from '@/types/ask';

vi.mock('@/api/client', () => ({
  apiClient: {
    baseURL: 'http://127.0.0.1:18011',
    getVlmModels: vi.fn(),
    getEndpoint: vi.fn(),
    streamToken: vi.fn(),
    streamUrl: vi.fn((path: string, token: string) => `http://server/stream?path=${path}&token=${token}`),
    listEntries: vi.fn(),
    getEntry: vi.fn(),
    createTry: vi.fn(),
    createDraft: vi.fn(),
    createRun: vi.fn(),
    getRunAnswers: vi.fn(),
    getFrames: vi.fn(),
    confirmEndpoint: vi.fn(),
    cancelJob: vi.fn(),
    createPreset: vi.fn(),
    updateEntry: vi.fn(),
    updatePrompt: vi.fn(),
    deletePrompt: vi.fn(),
    exportEntryTable: vi.fn(),
    getJobs: vi.fn(),
    uploadVideo: vi.fn(),
    getIngestAccess: vi.fn(),
  },
}));

const server = { capabilities: { version: '1.7.0' } as unknown, isLoading: false, error: null as unknown };
vi.mock('@/contexts/ServerCapabilitiesContext', () => ({
  useServerCapabilitiesContext: () => server,
}));

const VIDEO = '/data/p01_t1.mp4';
const LOCAL = { baseUrl: 'http://127.0.0.1:11434', host: '127.0.0.1', class: 'local' as const, confirmed: true };
const REMOTE = { baseUrl: 'https://api.example.com', host: 'api.example.com', class: 'remote' as const, confirmed: false };

const answer = (t: number, label: string, over: Partial<AskAnswer> = {}): AskAnswer => ({
  t,
  status: 'answered',
  label,
  reasoning: `because ${label}`,
  frame_numbers: [Math.round(t * 25)],
  frame_times: [t],
  total_time: 1.5,
  ...over,
});

const entry = (over: Partial<AskEntry> = {}): AskEntry =>
  ({
    id: 'e1',
    kind: 'try',
    status: 'done',
    prompt: 'Is the adult touching the infant?',
    prompt_sha256: 'a'.repeat(64),
    model: 'qwen3.5:9b',
    base_url: LOCAL.baseUrl,
    endpoint_class: 'local',
    sampling: { interval: 5, start: 0, end: null, moments: null, frames: 1, spacing: null },
    think: false,
    video_path: VIDEO,
    video_name: 'p01_t1.mp4',
    moments: [0],
    answers: [answer(0, 'TOUCH')],
    user_id: 'irene',
    name: null,
    starred: false,
    created_at: '2026-10-19T10:00:00Z',
    ...over,
  }) as AskEntry;

let location = '';
const Spy = () => {
  const l = useLocation();
  location = `${l.pathname}${l.search}`;
  (window as unknown as { lastState: unknown }).lastState = l.state;
  return null;
};

const renderAsk = (url = `/ask?video=${encodeURIComponent(VIDEO)}`) =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <MemoryRouter initialEntries={[url]}>
        <Routes>
          <Route path="/ask" element={<AskAI />} />
          <Route path="/workbench" element={<AskRedirect />} />
          <Route path="/prompts" element={<AskRedirect history />} />
          <Route path="/run" element={<p>run setup</p>} />
        </Routes>
        <Spy />
      </MemoryRouter>
    </QueryClientProvider>,
  );

/** Requests made so far; streamUrl only builds a string. */
const callCount = () =>
  Object.entries(apiClient as unknown as Record<string, unknown>)
    .filter(([name]) => name !== 'streamUrl')
    .map(([, f]) => f)
    .filter((f): f is { mock: { calls: unknown[] } } => typeof f === 'function' && 'mock' in (f as object))
    .reduce((n, f) => n + f.mock.calls.length, 0);

describe('Ask AI', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    server.capabilities = { version: '1.7.0' };
    server.error = null;
    try {
      localStorage.clear();
    } catch {
      // jsdom always has it; nothing to clear otherwise.
    }
    vi.mocked(apiClient.getVlmModels).mockResolvedValue({ baseUrl: LOCAL.baseUrl, models: ['qwen3.5:9b'], endpoint: LOCAL });
    vi.mocked(apiClient.getEndpoint).mockResolvedValue(LOCAL);
    vi.mocked(apiClient.streamToken).mockResolvedValue({
      token: 'tok',
      expires_at: '2026-10-19T11:00:00Z',
      path: VIDEO,
      name: 'p01_t1.mp4',
      duration_sec: 32,
      playable_copy: 'none',
    });
    vi.mocked(apiClient.listEntries).mockResolvedValue({ entries: [], library: [], next_before: null });
    vi.mocked(apiClient.getJobs).mockResolvedValue({ jobs: [], total: 0 } as never);
    vi.mocked(apiClient.getIngestAccess).mockResolvedValue({ can_read_in_place: false } as never);
    vi.mocked(apiClient.getFrames).mockResolvedValue({ t: 0, frames: [], requested: 1 });
  });

  describe('first answer (US1)', () => {
    it('plays the video from the query string through a stream URL', async () => {
      renderAsk();
      const video = await screen.findByTestId('ask-video');
      expect(video.getAttribute('src')).toBe(`http://server/stream?path=${VIDEO}&token=tok`);
      expect(apiClient.streamToken).toHaveBeenCalledWith(VIDEO);
    });

    it('shows the answer once the try is done, with a marker on the timeline', async () => {
      vi.mocked(apiClient.createTry).mockResolvedValue(entry({ status: 'running', answers: [] }));
      vi.mocked(apiClient.getEntry).mockResolvedValue(entry());
      renderAsk();
      const question = await screen.findByLabelText('Question');
      await userEvent.type(question, 'Is the adult touching the infant?');
      await userEvent.click(await screen.findByRole('button', { name: /Try on this frame/ }));
      expect(apiClient.createTry).toHaveBeenCalledWith(
        expect.objectContaining({ video_path: VIDEO, model: 'qwen3.5:9b', at: 0, prompt: 'Is the adult touching the infant?' }),
      );
      const featured = await screen.findByTestId('featured-answer', {}, { timeout: 3000 });
      expect(within(featured).getByText('TOUCH')).toBeInTheDocument();
      expect(screen.getAllByTestId('answer-marker')).toHaveLength(1);
    });

    it("says what's missing when the model server can't be reached", async () => {
      vi.mocked(apiClient.getVlmModels).mockRejectedValue(new APIError('Cannot reach the model server', 503));
      renderAsk();
      expect(await screen.findByText(/can't reach the model server/i)).toBeInTheDocument();
      expect(screen.getByText(/ollama.com\/download/)).toBeInTheDocument();
    });

    it('says to pull a model when none is there', async () => {
      vi.mocked(apiClient.getVlmModels).mockResolvedValue({ baseUrl: LOCAL.baseUrl, models: [], endpoint: LOCAL });
      renderAsk();
      expect(await screen.findByText('No model to ask yet')).toBeInTheDocument();
      expect(screen.getAllByText(/ollama pull/).length).toBeGreaterThan(0);
    });

    it('says it needs a server when there is none', async () => {
      server.capabilities = null;
      renderAsk();
      expect(await screen.findByTestId('needs-server')).toBeInTheDocument();
    });

    it('inserts an example, asking before replacing a question', async () => {
      renderAsk();
      const question = await screen.findByLabelText('Question');
      await userEvent.click(screen.getByRole('button', { name: /Examples/ }));
      await userEvent.click(await screen.findByText('Gaze'));
      expect((question as HTMLTextAreaElement).value).toContain('at the caregiver');
      await userEvent.click(screen.getByRole('button', { name: /Examples/ }));
      await userEvent.click(await screen.findByText('Touch'));
      await userEvent.click(await screen.findByRole('button', { name: 'Replace' }));
      expect((question as HTMLTextAreaElement).value).toContain('touching, not touching');
    });
  });

  describe('history (US2)', () => {
    it('restores an entry, keeping the unsaved question as a draft first', async () => {
      const old = entry({ id: 'e2', prompt: 'Where is the infant looking?', sampling: { interval: 2, start: 0, frames: 3, spacing: 0.5 } });
      vi.mocked(apiClient.listEntries).mockResolvedValue({ entries: [old], library: [], next_before: null });
      vi.mocked(apiClient.getEntry).mockResolvedValue(old);
      vi.mocked(apiClient.createDraft).mockResolvedValue(entry({ id: 'd1', kind: 'draft' }));
      renderAsk();
      await userEvent.type(await screen.findByLabelText('Question'), 'A question never tried');
      const [item] = await screen.findAllByTestId('history-entry');
      await userEvent.click(within(item).getByText('Where is the infant looking?'));
      expect(apiClient.createDraft).toHaveBeenCalledWith(expect.objectContaining({ prompt: 'A question never tried' }));
      expect((screen.getByLabelText('Question') as HTMLTextAreaElement).value).toBe('Where is the infant looking?');
      expect((screen.getByLabelText('Every (s)') as HTMLInputElement).value).toBe('2');
      expect((screen.getByLabelText('Seconds apart') as HTMLInputElement).value).toBe('0.5');
      expect(location).toContain('entry=e2');
    });

    it('shows two entries as two lanes and two answer columns', async () => {
      const a = entry({ id: 'a', name: 'touch v1', answers: [answer(0, 'TOUCH'), answer(5, 'TOUCH')] });
      const b = entry({ id: 'b', name: 'touch v2', answers: [answer(5, 'NO_TOUCH'), answer(10, 'TOUCH')] });
      vi.mocked(apiClient.listEntries).mockResolvedValue({ entries: [a, b], library: [], next_before: null });
      vi.mocked(apiClient.getEntry).mockImplementation(async (id) => (id === 'a' ? a : b));
      renderAsk();
      const items = await screen.findAllByTestId('history-entry');
      for (const item of items) await userEvent.click(within(item).getByRole('button', { name: 'Show its answers' }));
      await waitFor(() => expect(screen.getAllByTestId('answer-lane')).toHaveLength(2));
      const table = screen.getByTestId('answers-table');
      expect(within(table).getByText('touch v1')).toBeInTheDocument();
      expect(within(table).getByText('touch v2')).toBeInTheDocument();
      expect(screen.getAllByTestId('answer-row')).toHaveLength(3);
    });

    it('marks a moment tried twice with different answers', async () => {
      const a = entry({ answers: [answer(5, 'TOUCH'), answer(5, 'NO_TOUCH')] });
      vi.mocked(apiClient.listEntries).mockResolvedValue({ entries: [a], library: [], next_before: null });
      vi.mocked(apiClient.getEntry).mockResolvedValue(a);
      renderAsk();
      const [item] = await screen.findAllByTestId('history-entry');
      await userEvent.click(within(item).getByRole('button', { name: 'Show its answers' }));
      expect(await screen.findByTestId('repeat-differs')).toBeInTheDocument();
    });

    it('lands /workbench links on Ask AI with their question', async () => {
      renderAsk(`/workbench?prompt=hello&model=gemma4&video=${encodeURIComponent(VIDEO)}`);
      await waitFor(() => expect(location).toBe(`/ask?prompt=hello&model=gemma4&video=${encodeURIComponent(VIDEO)}`));
      expect(((await screen.findByLabelText('Question')) as HTMLTextAreaElement).value).toBe('hello');
    });

    it('lands /prompts on Ask AI with the history open', async () => {
      renderAsk('/prompts');
      await waitFor(() => expect(location).toBe('/ask?history=open'));
    });
  });

  describe('sampling (US3)', () => {
    it('redraws the ticks and count without asking the server', async () => {
      renderAsk();
      await waitFor(() => expect(screen.getAllByTestId('sample-tick')).toHaveLength(7));
      await waitFor(() => expect(screen.getByTestId('sample-count').textContent).toMatch(/^7 samples/));
      const before = callCount();
      fireEvent.change(screen.getByLabelText('Every (s)'), { target: { value: '2' } });
      expect(screen.getAllByTestId('sample-tick')).toHaveLength(17);
      expect(screen.getByTestId('sample-count').textContent).toMatch(/^17 samples/);
      expect(callCount()).toBe(before);
    });

    it('draws exactly the listed moments', async () => {
      renderAsk();
      await waitFor(() => expect(screen.getAllByTestId('sample-tick')).toHaveLength(7));
      await userEvent.click(screen.getByLabelText('Only at these moments'));
      fireEvent.change(screen.getByLabelText('Moments'), { target: { value: '0:12, 0:20, 0:31' } });
      expect(screen.getAllByTestId('sample-tick').map((t) => t.getAttribute('x1'))).toEqual(['37.5%', '62.5%', '96.875%']);
    });

    it('is a rough guess until the model has answered, then measured', async () => {
      renderAsk();
      expect(await screen.findByText(/rough guess until you try it/)).toBeInTheDocument();
    });

    it('uses the measured time once there are answers', async () => {
      vi.mocked(apiClient.listEntries).mockResolvedValue({
        entries: [entry({ answers: [answer(0, 'TOUCH', { total_time: 10 })] })],
        library: [],
        next_before: null,
      });
      renderAsk();
      await waitFor(() => expect(screen.getByTestId('sample-count').textContent).toBe('7 samples · about 1 min'));
      expect(screen.queryByText(/rough guess/)).not.toBeInTheDocument();
    });

    it('shows the frames of the sample nearest the playhead for a burst', async () => {
      vi.mocked(apiClient.getFrames).mockResolvedValue({
        t: 0,
        frames: [
          { frame_number: 0, timestamp_sec: 0, jpeg_base64: 'AA==' },
          { frame_number: 12, timestamp_sec: 0.5, jpeg_base64: 'AA==' },
        ],
        requested: 3,
      });
      renderAsk();
      await screen.findByTestId('ask-video');
      fireEvent.change(screen.getByLabelText('Frames per moment'), { target: { value: '3' } });
      expect(await screen.findByText(/2 of 3 frames/)).toBeInTheDocument();
      expect(apiClient.getFrames).toHaveBeenCalledWith(expect.objectContaining({ path: VIDEO, t: 0, frames: 3 }));
    });
  });

  describe('where the video goes (US4)', () => {
    it('asks before the first send to a remote server, and cancel sends nothing', async () => {
      vi.mocked(apiClient.getVlmModels).mockResolvedValue({ baseUrl: REMOTE.baseUrl, models: ['qwen3.5:9b'], endpoint: REMOTE });
      renderAsk();
      expect(await screen.findByTestId('endpoint-badge')).toHaveTextContent('Remote · frames are sent to api.example.com');
      await userEvent.type(screen.getByLabelText('Question'), 'Touch?');
      await userEvent.click(screen.getByRole('button', { name: /Try on this frame/ }));
      expect(await screen.findByText('Send frames to api.example.com?')).toBeInTheDocument();
      await userEvent.click(screen.getByRole('button', { name: "Don't send" }));
      expect(apiClient.createTry).not.toHaveBeenCalled();
      expect(apiClient.confirmEndpoint).not.toHaveBeenCalled();
    });

    it('records the agreement, then sends', async () => {
      vi.mocked(apiClient.getVlmModels).mockResolvedValue({ baseUrl: REMOTE.baseUrl, models: ['qwen3.5:9b'], endpoint: REMOTE });
      vi.mocked(apiClient.createTry).mockResolvedValue(entry({ status: 'running', answers: [] }));
      vi.mocked(apiClient.getEntry).mockResolvedValue(entry());
      vi.mocked(apiClient.confirmEndpoint).mockResolvedValue(undefined);
      renderAsk();
      await userEvent.type(await screen.findByLabelText('Question'), 'Touch?');
      await userEvent.click(screen.getByRole('button', { name: /Try on this frame/ }));
      await userEvent.click(await screen.findByRole('button', { name: 'Send to api.example.com' }));
      await waitFor(() => expect(apiClient.createTry).toHaveBeenCalled());
      expect(apiClient.confirmEndpoint).toHaveBeenCalledWith(REMOTE.baseUrl);
    });

    it("doesn't ask once the server is confirmed", async () => {
      vi.mocked(apiClient.getVlmModels).mockResolvedValue({
        baseUrl: REMOTE.baseUrl,
        models: ['qwen3.5:9b'],
        endpoint: { ...REMOTE, confirmed: true },
      });
      vi.mocked(apiClient.createTry).mockResolvedValue(entry({ status: 'running', answers: [] }));
      renderAsk();
      await userEvent.type(await screen.findByLabelText('Question'), 'Touch?');
      await userEvent.click(screen.getByRole('button', { name: /Try on this frame/ }));
      await waitFor(() => expect(apiClient.createTry).toHaveBeenCalled());
      expect(screen.queryByText('Send frames to api.example.com?')).not.toBeInTheDocument();
    });
  });

  describe('runs (US5)', () => {
    it('fills in answers as they come, and Stop keeps them', async () => {
      const runEntry = entry({ id: 'r1', kind: 'run', status: 'running', job_id: 'job-1', answers: null, moments: null });
      vi.mocked(apiClient.createRun).mockResolvedValue(runEntry);
      vi.mocked(apiClient.getEntry).mockResolvedValue(runEntry);
      vi.mocked(apiClient.cancelJob).mockResolvedValue({ job_id: 'job-1', status: 'cancelled', message: '' });
      const pages = [
        { status: 'running', planned: 7, done: 1, stopped_early: false, answers: [answer(0, 'TOUCH')], skipped: [] },
        { status: 'running', planned: 7, done: 2, stopped_early: false, answers: [answer(5, 'NO_TOUCH')], skipped: [] },
        { status: 'stopped', planned: 7, done: 2, stopped_early: true, answers: [], skipped: [] },
      ];
      vi.mocked(apiClient.getRunAnswers).mockImplementation(async () => pages.shift() ?? pages[0] ?? { status: 'stopped', planned: 7, done: 2, stopped_early: true, answers: [], skipped: [] });
      renderAsk();
      await userEvent.type(await screen.findByLabelText('Question'), 'Touch?');
      await userEvent.click(screen.getByRole('button', { name: /Run on whole video/ }));
      expect(await screen.findByText('1 of 7')).toBeInTheDocument();
      expect(await screen.findByText('2 of 7', {}, { timeout: 4000 })).toBeInTheDocument();
      expect(screen.getAllByTestId('answer-marker')).toHaveLength(2);
      expect(apiClient.getRunAnswers).toHaveBeenLastCalledWith('job-1', 1);
      await userEvent.click(screen.getByRole('button', { name: /Stop/ }));
      expect(apiClient.cancelJob).toHaveBeenCalledWith('job-1');
      expect(await screen.findByText('Stopped after 2 of 7', {}, { timeout: 4000 })).toBeInTheDocument();
      expect(screen.getAllByTestId('answer-marker')).toHaveLength(2);
    }, 15000);

    it('opens run setup with the question and sampling for more videos', async () => {
      renderAsk();
      await userEvent.type(await screen.findByLabelText('Question'), 'Touch?');
      fireEvent.change(screen.getByLabelText('Every (s)'), { target: { value: '2' } });
      await userEvent.click(screen.getByRole('button', { name: /Run on more videos/ }));
      expect(await screen.findByText('run setup')).toBeInTheDocument();
      const state = (window as unknown as { lastState: { runStart: { settings: { config: Record<string, Record<string, unknown>> } } } })
        .lastState;
      expect(state.runStart.settings.config.vlm_annotation).toMatchObject({ prompt: 'Touch?', model: 'qwen3.5:9b', frame_interval_sec: 2 });
    });
  });
});
