import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import Home from '@/pages/Home';
import { apiClient } from '@/api/client';
import type { JobResponse } from '@/api/client';

vi.mock('@/api/client', () => ({
  apiClient: {
    getJobs: vi.fn(),
    getPipelineCatalog: vi.fn(),
    listDatasets: vi.fn(),
    listPrompts: vi.fn(),
    rerunJob: vi.fn(),
  },
}));
const capabilities = { version: '1.6.0' };
vi.mock('@/contexts/ServerCapabilitiesContext', () => ({
  useServerCapabilitiesContext: () => ({ capabilities, isLoading: false }),
}));
const local = { folder: null as string | null, saved: {} as Record<string, unknown> };
vi.mock('@/lib/localLibrary/libraryStore', () => ({
  getRootDirHandle: async () => (local.folder ? { name: local.folder } : null),
  getJobDatasetIndex: async () => local.saved,
}));

const job = (over: Partial<JobResponse>): JobResponse =>
  ({ id: 'j1', status: 'completed', video_filename: 'baby.mp4', selected_pipelines: ['face_analysis'],
     progress_percentage: 100, created_at: new Date().toISOString(), ...over }) as JobResponse; // prettier-ignore

let wizardState: unknown;
const Wizard = () => {
  wizardState = useLocation().state;
  return <p>wizard</p>;
};
const renderHome = () =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <MemoryRouter>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/jobs/new" element={<Wizard />} />
          <Route path="/view/:id" element={<p>viewer</p>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );

describe('Home dashboard', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    local.folder = null;
    local.saved = {};
    vi.mocked(apiClient.getPipelineCatalog).mockResolvedValue({
      catalog: { pipelines: [{ id: 'a', name: 'A' }, { id: 'b', name: 'B', available: false }] },
    } as never);
    vi.mocked(apiClient.listDatasets).mockResolvedValue({ datasets: [], total: 0 });
    vi.mocked(apiClient.listPrompts).mockResolvedValue({ prompts: [], total: 0 } as never);
  });

  it('shows the setup checklist on a new install, with pipeline readiness', async () => {
    vi.mocked(apiClient.getJobs).mockResolvedValue({ jobs: [], total: 0 } as never);
    renderHome();
    expect(await screen.findByText('Getting set up')).toBeInTheDocument();
    expect(await screen.findByText(/1 of 2 pipelines ready/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Open latest results/ })).toBeDisabled();
  });

  it('opens the latest finished job and offers fix & rerun on a failed one', async () => {
    local.folder = 'results';
    local.saved = { j1: {} };
    vi.mocked(apiClient.getJobs).mockResolvedValue({
      jobs: [job({ id: 'j2', status: 'failed', video_filename: 'bad.mp4', error_message: 'CUDA out of memory' }), job({})],
      total: 2,
    } as never);
    renderHome();
    expect(await screen.findByText('CUDA out of memory')).toBeInTheDocument();
    expect(screen.queryByText('Getting set up')).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: /Fix & rerun/ }));
    expect(wizardState).toEqual({
      wizardStart: { mode: 'rerun', jobId: 'j2', label: 'bad.mp4', selectedPipelines: ['face_analysis'], config: null },
    });
  });

  it('Open latest results goes to the newest completed job', async () => {
    vi.mocked(apiClient.getJobs).mockResolvedValue({
      jobs: [job({ id: 'j3', status: 'running', progress_percentage: 40 }), job({})],
      total: 2,
    } as never);
    renderHome();
    expect(await screen.findByText('running 40%')).toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole('button', { name: /Open latest results/ })).toBeEnabled());
    await userEvent.click(screen.getByRole('button', { name: /Open latest results/ }));
    expect(await screen.findByText('viewer')).toBeInTheDocument();
  });
});
