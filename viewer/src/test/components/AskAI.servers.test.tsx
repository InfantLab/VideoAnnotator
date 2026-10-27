/** Ask AI with more than one model server (spec 033 Story 1). */
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import AskAI from '@/pages/AskAI';
import { apiClient } from '@/api/client';
import { APIError } from '@/api/handleError';
import { DEFAULT_SERVER, LAB_SERVER, serversResponse, stubPointerCapture } from '@/test/fixtures/modelServers';
import type { AskEntry } from '@/types/ask';

vi.mock('@/api/client', () => ({
  apiClient: {
    baseURL: 'http://127.0.0.1:18011',
    getVlmModels: vi.fn(),
    getEndpoint: vi.fn(),
    streamToken: vi.fn(),
    streamUrl: vi.fn(() => 'http://server/stream'),
    listEntries: vi.fn(),
    getEntry: vi.fn(),
    createTry: vi.fn(),
    createDraft: vi.fn(),
    createRun: vi.fn(),
    getFrames: vi.fn(),
    getJobs: vi.fn(),
    getIngestAccess: vi.fn(),
    listModelServers: vi.fn(),
    getAskSettings: vi.fn(),
    setAskSettings: vi.fn(),
    suggestShape: vi.fn(),
    checkModelServer: vi.fn(),
  },
}));

vi.mock('@/contexts/ServerCapabilitiesContext', () => ({
  useServerCapabilitiesContext: () => ({ capabilities: { version: '1.7.0' }, isLoading: false, error: null }),
}));

const VIDEO = '/data/p01_t1.mp4';

const TRY = {
  id: 't1',
  kind: 'try',
  status: 'running',
  prompt: 'Touch?',
  prompt_sha256: 'a'.repeat(64),
  model: 'gemma-3-12b',
  base_url: LAB_SERVER.base_url,
  backend: 'openai_compatible',
  endpoint_class: 'local',
  sampling: { interval: 5, start: 0, end: null, moments: null, frames: 1, spacing: null },
  think: false,
  video_path: VIDEO,
  video_name: 'p01_t1.mp4',
  moments: [0],
  answers: [],
  starred: false,
  created_at: '2026-10-22T10:00:00Z',
} as unknown as AskEntry;

const endpointFor = (baseUrl?: string | null) =>
  baseUrl === LAB_SERVER.base_url
    ? { baseUrl, host: '10.0.4.12', class: 'local' as const, confirmed: true }
    : { baseUrl: DEFAULT_SERVER.base_url, host: '127.0.0.1', class: 'local' as const, confirmed: true };

const renderAsk = () =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <MemoryRouter initialEntries={[`/ask?video=${encodeURIComponent(VIDEO)}`]}>
        <Routes>
          <Route path="/ask" element={<AskAI />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );

describe('Ask AI model servers', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    stubPointerCapture();
    vi.mocked(apiClient.listModelServers).mockResolvedValue(serversResponse());
    vi.mocked(apiClient.getAskSettings).mockResolvedValue({ server: 'default' });
    vi.mocked(apiClient.setAskSettings).mockImplementation(async (body) => body);
    vi.mocked(apiClient.getVlmModels).mockImplementation(async (server) =>
      server === 'lab'
        ? { baseUrl: LAB_SERVER.base_url, models: ['gemma-3-12b'], endpoint: endpointFor(LAB_SERVER.base_url), server: 'lab', kind: 'openai_compatible' }
        : { baseUrl: DEFAULT_SERVER.base_url, models: ['qwen3.5:9b'], endpoint: endpointFor(), server: 'default', kind: 'ollama' },
    );
    vi.mocked(apiClient.getEndpoint).mockImplementation(async (baseUrl) => endpointFor(baseUrl));
    vi.mocked(apiClient.streamToken).mockResolvedValue({
      token: 'tok',
      expires_at: '2026-10-19T11:00:00Z',
      path: VIDEO,
      name: 'p01_t1.mp4',
      duration_sec: 32,
      playable_copy: 'none',
    });
    vi.mocked(apiClient.listEntries).mockResolvedValue({ entries: [], library: [], next_before: null });
    vi.mocked(apiClient.getFrames).mockResolvedValue({ t: 0, frames: [], requested: 1 });
    vi.mocked(apiClient.createTry).mockResolvedValue(TRY);
    vi.mocked(apiClient.getEntry).mockResolvedValue(TRY);
  });

  async function chooseLab() {
    await userEvent.click(await screen.findByRole('combobox', { name: 'Model server' }));
    await userEvent.click(await screen.findByRole('option', { name: /lab-gpu/ }));
  }

  it('starts on this computer, and the badge follows the chosen server', async () => {
    renderAsk();
    expect(await screen.findByTestId('server-picker')).toHaveTextContent(DEFAULT_SERVER.name);
    expect(await screen.findByTestId('endpoint-badge')).toHaveTextContent('on this computer');
    await chooseLab();
    await waitFor(() => expect(screen.getByTestId('endpoint-badge')).toHaveTextContent('on 10.0.4.12'));
    expect(apiClient.getVlmModels).toHaveBeenCalledWith('lab');
    expect(apiClient.setAskSettings).toHaveBeenCalledWith({ server: 'lab' });
  });

  it("sends the chosen server's kind and address with a try", async () => {
    renderAsk();
    await chooseLab();
    await waitFor(() => expect(screen.getByRole('combobox', { name: 'Model' })).toHaveTextContent('gemma-3-12b'));
    await userEvent.type(screen.getByLabelText('Question'), 'Touch?');
    await userEvent.click(screen.getByRole('button', { name: /Try on this frame/ }));
    await waitFor(() =>
      expect(apiClient.createTry).toHaveBeenCalledWith(
        expect.objectContaining({ backend: 'openai_compatible', base_url: LAB_SERVER.base_url, model: 'gemma-3-12b' }),
      ),
    );
  });

  it('opens on the server chosen last time', async () => {
    vi.mocked(apiClient.getAskSettings).mockResolvedValue({ server: 'lab' });
    renderAsk();
    await waitFor(() => expect(screen.getByTestId('server-picker')).toHaveTextContent('lab-gpu'));
    expect(apiClient.getVlmModels).toHaveBeenCalledWith('lab');
  });

  it("says what to do when the chosen server can't be reached", async () => {
    vi.mocked(apiClient.getAskSettings).mockResolvedValue({ server: 'lab' });
    vi.mocked(apiClient.getVlmModels).mockImplementation(async (server) => {
      if (server === 'lab') throw new APIError("Can't reach the model server", 503);
      return { baseUrl: DEFAULT_SERVER.base_url, models: ['qwen3.5:9b'], endpoint: endpointFor() };
    });
    renderAsk();
    expect(await screen.findByTestId('no-model')).toHaveTextContent("Ask AI can't reach lab-gpu");
    expect(screen.getByTestId('no-model')).not.toHaveTextContent('ollama.com');
  });

  describe('a short clip per sample', () => {
    const clipOption = () => screen.getByRole('option', { name: /A short clip/ }) as HTMLOptionElement;

    it("isn't offered on Ollama", async () => {
      renderAsk();
      await screen.findByTestId('server-picker');
      expect(clipOption().disabled).toBe(true);
      expect(clipOption()).toHaveTextContent("Ollama can't take video");
    });

    it('is offered when the check says the model takes video, and sent as a clip', async () => {
      vi.mocked(apiClient.listModelServers).mockResolvedValue(
        serversResponse([DEFAULT_SERVER, { ...LAB_SERVER, video_models: { 'gemma-3-12b': true } }]),
      );
      vi.mocked(apiClient.getAskSettings).mockResolvedValue({ server: 'lab' });
      renderAsk();
      await waitFor(() => expect(screen.getByRole('combobox', { name: 'Model' })).toHaveTextContent('gemma-3-12b'));
      await waitFor(() => expect(clipOption().disabled).toBe(false));
      await userEvent.selectOptions(screen.getByLabelText('Frames per moment'), 'clip');
      expect(screen.getByLabelText('Clip length (s)')).toHaveValue(2);
      await userEvent.type(screen.getByLabelText('Question'), 'What happens?');
      await userEvent.click(screen.getByRole('button', { name: /Try on this frame/ }));
      await waitFor(() =>
        expect(apiClient.createTry).toHaveBeenCalledWith(
          expect.objectContaining({ sampling: expect.objectContaining({ clip: { length: 2, fps: 4 } }) }),
        ),
      );
      expect(apiClient.checkModelServer).not.toHaveBeenCalled();
    });

    it("asks the server once when nobody has checked, and says no when it can't", async () => {
      vi.mocked(apiClient.getAskSettings).mockResolvedValue({ server: 'lab' });
      vi.mocked(apiClient.checkModelServer).mockResolvedValue({ video: false } as never);
      renderAsk();
      await waitFor(() =>
        expect(apiClient.checkModelServer).toHaveBeenCalledWith(
          expect.objectContaining({ id: 'lab', model: 'gemma-3-12b', kind: 'openai_compatible' }),
        ),
      );
      await waitFor(() => expect(clipOption()).toHaveTextContent("this model or server doesn't take video"));
      expect(clipOption().disabled).toBe(true);
    });
  });

  it('shows no picker while only the default server is listed', async () => {
    vi.mocked(apiClient.listModelServers).mockResolvedValue(serversResponse([DEFAULT_SERVER]));
    renderAsk();
    await screen.findByTestId('endpoint-badge');
    expect(screen.queryByTestId('server-picker')).not.toBeInTheDocument();
  });
});
