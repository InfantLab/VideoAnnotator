import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { RunAgainActions, type RunAgainTarget } from '@/components/RunAgainActions';
import { apiClient } from '@/api/client';
import { APIError } from '@/api/handleError';

vi.mock('@/api/client', () => ({
  apiClient: { rerunJob: vi.fn(), rerunBatch: vi.fn(), createPreset: vi.fn() },
}));

const target: RunAgainTarget = {
  kind: 'job',
  id: 'j1',
  label: 'clip.mp4',
  settings: { selectedPipelines: ['scene_detection'], config: { scene_detection: { threshold: 20 } } },
};

let landed: { path: string; state: unknown } | null = null;
const Spy = () => {
  const location = useLocation();
  landed = { path: location.pathname, state: location.state };
  return null;
};

const renderActions = (t: RunAgainTarget = target) =>
  render(
    <QueryClientProvider client={new QueryClient()}>
      <MemoryRouter initialEntries={['/jobs/j1']}>
        <Routes>
          <Route path="/jobs/j1" element={<RunAgainActions target={t} />} />
          <Route path="*" element={<Spy />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );

describe('RunAgainActions', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    landed = null;
  });

  it('runs again only after saying the original is kept, then opens the new job', async () => {
    vi.mocked(apiClient.rerunJob).mockResolvedValue({ id: 'j2' } as never);
    renderActions();
    await userEvent.click(screen.getByRole('button', { name: /^Run again$/ }));
    expect(screen.getByText(/This one and its results are kept/)).toBeInTheDocument();
    expect(apiClient.rerunJob).not.toHaveBeenCalled();
    await userEvent.click(screen.getAllByRole('button', { name: 'Run again' }).at(-1)!);
    await waitFor(() => expect(landed?.path).toBe('/jobs/j2'));
    expect(apiClient.rerunJob).toHaveBeenCalledWith('j1');
  });

  it('opens the wizard to edit, or to reuse the settings on other videos', async () => {
    renderActions();
    await userEvent.click(screen.getByRole('button', { name: /Edit and run again/ }));
    expect(landed).toEqual({
      path: '/jobs/new',
      state: { wizardStart: { mode: 'rerun', jobId: 'j1', label: 'clip.mp4', ...target.settings } },
    });
  });

  it('says the video is gone and what to do instead', async () => {
    vi.mocked(apiClient.rerunJob).mockRejectedValue(
      new APIError('x', 409, undefined, { error: { code: 'RERUN_VIDEO_MISSING' } }),
    );
    renderActions();
    await userEvent.click(screen.getByRole('button', { name: /^Run again$/ }));
    await userEvent.click(screen.getAllByRole('button', { name: 'Run again' }).at(-1)!);
    expect(await screen.findByText(/no longer on the server/)).toBeInTheDocument();
  });

  it('saves the settings as a preset', async () => {
    vi.mocked(apiClient.createPreset).mockResolvedValue({ name: 'clip.mp4' } as never);
    renderActions();
    await userEvent.click(screen.getByRole('button', { name: /Save as preset/ }));
    await userEvent.click(screen.getByRole('button', { name: 'Save' }));
    expect(apiClient.createPreset).toHaveBeenCalledWith({
      name: 'clip.mp4',
      selected_pipelines: ['scene_detection'],
      config: { scene_detection: { threshold: 20 } },
    });
    expect(await screen.findByText(/Saved preset/)).toBeInTheDocument();
  });
});
