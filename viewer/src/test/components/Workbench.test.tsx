import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import Workbench from '@/pages/Workbench';
import { apiClient } from '@/api/client';
import { APIError } from '@/api/handleError';
import { evenlySpaced, parseMoments } from '@/lib/workbench';

vi.mock('@/api/client', () => ({
  apiClient: {
    getJobs: vi.fn(),
    getVlmModels: vi.fn(),
    listPrompts: vi.fn(),
    previewVlmPrompt: vi.fn(),
    createPreset: vi.fn(),
  },
}));

const job = { id: 'job-1', status: 'completed', video_path: '/srv/clip.mp4', video_filename: 'clip.mp4', video_duration_seconds: 10 };

const renderWorkbench = (state?: unknown) =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <MemoryRouter initialEntries={[{ pathname: '/workbench', state }]}>
        <Workbench />
      </MemoryRouter>
    </QueryClientProvider>,
  );

describe('workbench helpers', () => {
  it('reads moments and spreads them through a video', () => {
    expect(parseMoments('2, 5.5  9,x,-1')).toEqual([2, 5.5, 9]);
    expect(evenlySpaced(4, 10)).toEqual([2, 4, 6, 8]);
    expect(evenlySpaced(3, 0)).toEqual([]);
  });
});

describe('Workbench', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(apiClient.getJobs).mockResolvedValue({ jobs: [job], total: 1, page: 1, per_page: 50 } as never);
    vi.mocked(apiClient.getVlmModels).mockResolvedValue({ baseUrl: 'http://127.0.0.1:11434', models: ['gemma4:e4b', 'qwen3.5:9b'] });
    vi.mocked(apiClient.listPrompts).mockResolvedValue({ prompts: [], total: 0 });
  });

  it('runs every prompt × model × moment, shows each result, and keeps going past a failure', async () => {
    vi.mocked(apiClient.previewVlmPrompt).mockImplementation(async (req) => {
      if (req.timestampSec === 4 && req.prompt === 'B?') throw new APIError('model crashed', 502);
      return {
        label: `${req.prompt === 'A?' ? 'YES' : 'NO'}@${req.timestampSec}`,
        reasoning: 'because',
        rawResponse: '',
        totalTime: 1.5,
        loadTime: 0,
        promptTokens: 1,
        respTokens: 1,
        tokensPerSec: 1,
        frames: [{ frameNumber: 30, timestampSec: 1, jpegBase64: 'AA==' }],
      };
    });
    renderWorkbench({ workbench: { prompts: ['A?', 'B?'], model: 'gemma4:e4b' } });

    await userEvent.selectOptions(await screen.findByLabelText('Video'), 'job-1');
    await userEvent.type(screen.getByLabelText('Moments (seconds)'), '2, 4');
    await userEvent.click(screen.getByRole('button', { name: 'Run 4 combinations' }));

    await waitFor(() => expect(apiClient.previewVlmPrompt).toHaveBeenCalledTimes(4));
    expect(await screen.findByText('YES@2')).toBeInTheDocument();
    expect(screen.getByText('YES@4')).toBeInTheDocument();
    expect(screen.getByText('NO@2')).toBeInTheDocument();
    expect(screen.getByText('model crashed')).toBeInTheDocument();
    expect(screen.getAllByAltText('frame 30')).toHaveLength(3);
    expect(vi.mocked(apiClient.previewVlmPrompt).mock.calls[0][0]).toMatchObject({
      videoPath: '/srv/clip.mp4',
      model: 'gemma4:e4b',
      samplingMode: 'single_frame',
    });
    expect(screen.getByText(/Round 1: clip.mp4/)).toBeInTheDocument();
  });

  it('warns when the model server is not on this machine', async () => {
    vi.mocked(apiClient.getVlmModels).mockResolvedValue({ baseUrl: 'https://gpu.example.org', models: ['m'] });
    renderWorkbench();
    expect(await screen.findByText(/frames you test are sent there/)).toBeInTheDocument();
  });
});
