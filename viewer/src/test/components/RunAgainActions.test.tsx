import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { RunAgainActions, type RunAgainTarget } from '@/components/RunAgainActions';
import { apiClient } from '@/api/client';
import { APIError } from '@/api/handleError';

vi.mock('@/api/client', () => ({
  apiClient: {
    baseURL: 'http://127.0.0.1:18011',
    rerunJob: vi.fn(),
    rerunBatch: vi.fn(),
    createPreset: vi.fn(),
    getIngestAccess: vi.fn(),
    browseServerFolders: vi.fn(),
    scanServerFolder: vi.fn(),
  },
}));

const batchTarget: RunAgainTarget = {
  kind: 'batch',
  id: 'b1',
  label: 'Wave 2',
  videoCount: 3,
  settings: { selectedPipelines: ['scene_detection'] },
};

const result = (overrides: Partial<Awaited<ReturnType<typeof apiClient.rerunBatch>>> = {}) => ({
  batch_id: 'b2',
  rerun_of_batch: 'b1',
  created: [],
  skipped: [],
  relocated: [],
  ...overrides,
});

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
    expect(await screen.findByText(/choose the video where it is now/)).toBeInTheDocument();
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

  describe('a batch whose videos moved (spec 022)', () => {
    const missing = {
      job_id: 'j2',
      reason: 'Cannot run job j2 again: its video is no longer at /home/ada/Studies/site_a/child02.mp4',
    };

    beforeEach(() => {
      vi.mocked(apiClient.getIngestAccess).mockResolvedValue({
        same_machine: true,
        can_read_in_place: true,
        reason: null,
        allowed_folders: [{ path: '/home/ada', display_path: '/home/ada' }],
        results_root: { path: '/home/ada/VideoAnnotator', display_path: '/home/ada/VideoAnnotator' },
        can_open_folders: true,
        places: [{ label: 'Home', path: '/home/ada', display_path: '/home/ada', has_videos: true }],
      });
      vi.mocked(apiClient.browseServerFolders).mockImplementation(async (path?: string) => ({
        path: path ?? null, parent: null, roots: ['/home/ada'], directories: [], videos: [], video_count: 0, truncated: false,
      }));
      vi.mocked(apiClient.scanServerFolder).mockResolvedValue({ path: '/home/ada', recursive: false, videos: [] });
    });

    it('lists missing videos before anything starts, then runs the rest', async () => {
      vi.mocked(apiClient.rerunBatch)
        .mockResolvedValueOnce(result({ skipped: [missing] }))
        .mockResolvedValueOnce(result({ created: ['n1', 'n3'], skipped: [missing] }));
      renderActions(batchTarget);
      await userEvent.click(screen.getByRole('button', { name: /^Run again$/ }));

      expect(apiClient.rerunBatch).toHaveBeenCalledWith('b1', {}, { check: true });
      expect(await screen.findByTestId('missing-videos')).toHaveTextContent('/home/ada/Studies/site_a/child02.mp4');
      await userEvent.click(screen.getByRole('button', { name: 'Run the rest' }));
      await waitFor(() => expect(landed?.path).toBe('/batches/b2'));
      expect(apiClient.rerunBatch).toHaveBeenLastCalledWith('b1', {}, {});
    });

    it('looks for them in a chosen folder and runs them from there', async () => {
      vi.mocked(apiClient.rerunBatch)
        .mockResolvedValueOnce(result({ skipped: [missing] }))
        .mockResolvedValueOnce(
          result({ relocated: [{ job_id: 'j2', from: '/home/ada/Studies/site_a/child02.mp4', to: '/home/ada/child02.mp4' }] })
        )
        .mockResolvedValueOnce(result({ created: ['n1', 'n2', 'n3'] }));
      renderActions(batchTarget);
      await userEvent.click(screen.getByRole('button', { name: /^Run again$/ }));
      await userEvent.click(await screen.findByRole('button', { name: 'Locate…' }));
      await userEvent.click(await screen.findByRole('button', { name: /Home/ }));
      await waitFor(() => expect(apiClient.browseServerFolders).toHaveBeenCalledWith('/home/ada'));
      await userEvent.click(await screen.findByRole('button', { name: 'Look for them here' }));

      expect(apiClient.rerunBatch).toHaveBeenLastCalledWith('b1', {}, { check: true, relocateFolder: '/home/ada', recursive: false });
      expect(await screen.findByText(/Found 1 moved video/)).toBeInTheDocument();
      await userEvent.click(screen.getByRole('button', { name: 'Run again' }));
      await waitFor(() => expect(landed?.path).toBe('/batches/b2'));
      expect(apiClient.rerunBatch).toHaveBeenLastCalledWith('b1', {}, { relocateFolder: '/home/ada', recursive: false });
    });
  });
});
