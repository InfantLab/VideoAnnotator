// Integration tests for the Pipeline Extras Install UI (specs/002-pipeline-extras-install)
// US1: locked pipelines are visible, not hidden (T008)
// US2: admin can trigger + track an install; non-admin is blocked cleanly (T015)

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import React from 'react';
import { PipelineSelectionStep } from '@/pages/NewJob';
import { apiClient } from '@/api/client';
import type { PipelineDescriptor } from '@/types/pipelines';

vi.mock('@/api/client', async () => {
  const actual = await vi.importActual<typeof import('@/api/client')>('@/api/client');
  return {
    ...actual,
    hasConfiguredApiToken: vi.fn(() => true),
    apiClient: {
      installPipelineExtras: vi.fn(),
      getExtrasInstallJob: vi.fn()
    }
  };
});

const stubPipeline: PipelineDescriptor = {
  id: 'stub',
  name: 'Stub Forward-Compatibility Pipeline',
  group: 'Core',
  available: true
};

const facePipeline: PipelineDescriptor = {
  id: 'face_analysis',
  name: 'Face Analysis',
  group: 'Face',
  description: 'OpenFace3-based facial landmark and emotion analysis.',
  available: false,
  installHint: 'pip install videoannotator[face]'
};

const scenePipeline: PipelineDescriptor = {
  id: 'scene',
  name: 'Scene Detection',
  group: 'Scene',
  available: false,
  installHint: 'pip install videoannotator[scene]'
};

class MemoryStorage {
  private store = new Map<string, string>();
  getItem = vi.fn((key: string) => this.store.get(key) ?? null);
  setItem = vi.fn((key: string, value: string) => {
    this.store.set(key, value);
  });
  removeItem = vi.fn((key: string) => {
    this.store.delete(key);
  });
  clear = vi.fn(() => this.store.clear());
}

const renderWithProviders = (ui: React.ReactElement) => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
  });
  return render(<QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>);
};

describe('PipelineSelectionStep - Pipeline Extras Install UI', () => {
  beforeEach(() => {
    vi.stubGlobal('localStorage', new MemoryStorage());
    vi.clearAllMocks();
  });

  describe('locked pipeline visibility (US1)', () => {
    it('renders locked pipelines distinctly, with their install hint visible, alongside available ones', () => {
      renderWithProviders(
        <PipelineSelectionStep
          pipelines={[stubPipeline, facePipeline, scenePipeline]}
          selectedPipelines={['stub']}
          setSelectedPipelines={vi.fn()}
          isLoading={false}
          error={null}
          onRetry={vi.fn()}
        />
      );

      // Available pipeline: rendered as a normal selectable option
      expect(screen.getByText('Stub Forward-Compatibility Pipeline')).toBeInTheDocument();

      // Locked pipelines: visible, not omitted, with a "Not installed" marker and their hint text
      expect(screen.getByText('Face Analysis')).toBeInTheDocument();
      expect(screen.getByText('Scene Detection')).toBeInTheDocument();
      expect(screen.getAllByText('Not installed')).toHaveLength(2);
      expect(screen.getByText('pip install videoannotator[face]')).toBeInTheDocument();
      expect(screen.getByText('pip install videoannotator[scene]')).toBeInTheDocument();

      // Locked pipelines must not be selectable checkboxes
      const checkboxes = screen.getAllByRole('checkbox');
      expect(checkboxes).toHaveLength(1); // only the available "stub" pipeline
    });

    it('renders unchanged (no locked markers) when every pipeline is already available', () => {
      renderWithProviders(
        <PipelineSelectionStep
          pipelines={[stubPipeline, { ...facePipeline, available: true, installHint: undefined }]}
          selectedPipelines={['stub']}
          setSelectedPipelines={vi.fn()}
          isLoading={false}
          error={null}
          onRetry={vi.fn()}
        />
      );

      expect(screen.queryByText('Not installed')).not.toBeInTheDocument();
      expect(screen.getAllByRole('checkbox')).toHaveLength(2);
    });

    it('treats a pipeline with available left undefined (pre-v1.5.0 server shape) as available', () => {
      const legacyPipeline: PipelineDescriptor = { id: 'legacy', name: 'Legacy Pipeline', group: 'Core' };

      renderWithProviders(
        <PipelineSelectionStep
          pipelines={[legacyPipeline]}
          selectedPipelines={[]}
          setSelectedPipelines={vi.fn()}
          isLoading={false}
          error={null}
          onRetry={vi.fn()}
        />
      );

      expect(screen.getByRole('checkbox')).toBeInTheDocument();
      expect(screen.queryByText('Not installed')).not.toBeInTheDocument();
    });
  });

  describe('install trigger + tracking (US2)', () => {
    it('lets an authenticated user trigger an install and shows in-progress state scoped to that pipeline', async () => {
      const user = userEvent.setup();
      vi.mocked(apiClient.installPipelineExtras).mockResolvedValueOnce({
        jobId: 'job-1',
        extraName: 'face',
        status: 'pending'
      });
      vi.mocked(apiClient.getExtrasInstallJob).mockResolvedValue({
        jobId: 'job-1',
        extraName: 'face',
        status: 'running',
        createdAt: '2026-08-26T10:00:00Z',
        startedAt: '2026-08-26T10:00:01Z',
        finishedAt: null,
        commandOutput: null,
        restartRequired: false
      });

      renderWithProviders(
        <PipelineSelectionStep
          pipelines={[stubPipeline, facePipeline, scenePipeline]}
          selectedPipelines={['stub']}
          setSelectedPipelines={vi.fn()}
          isLoading={false}
          error={null}
          onRetry={vi.fn()}
        />
      );

      const installButtons = screen.getAllByRole('button', { name: 'Install' });
      // One Install button per locked pipeline's card - click the Face Analysis one.
      await user.click(installButtons[0]);

      await waitFor(() => {
        expect(apiClient.installPipelineExtras).toHaveBeenCalledWith('face');
      });

      await waitFor(() => {
        expect(screen.getByText(/Installing… this can take several minutes/)).toBeInTheDocument();
      });

      // Scene Detection's card (a different, untouched extras group) must not show progress.
      const sceneCard = screen.getByText('Scene Detection').closest('div[aria-disabled="true"]');
      expect(sceneCard?.textContent).not.toMatch(/Installing/);
    });

    it('shows an "administrator privileges required" message on a 403, not a silent failure', async () => {
      const user = userEvent.setup();
      const { APIError } = await import('@/api/handleError');
      vi.mocked(apiClient.installPipelineExtras).mockRejectedValueOnce(
        new APIError('Admin privileges required', 403)
      );

      renderWithProviders(
        <PipelineSelectionStep
          pipelines={[stubPipeline, facePipeline]}
          selectedPipelines={['stub']}
          setSelectedPipelines={vi.fn()}
          isLoading={false}
          error={null}
          onRetry={vi.fn()}
        />
      );

      await user.click(screen.getByRole('button', { name: 'Install' }));

      await waitFor(() => {
        expect(
          screen.getByText('Administrator privileges are required to install pipeline extras.')
        ).toBeInTheDocument();
      });
    });

    it('resumes showing in-progress state after a reload (pre-seeded localStorage job)', async () => {
      (localStorage as unknown as MemoryStorage).setItem(
        'videoannotator_extras_install_jobs',
        JSON.stringify({ face: { jobId: 'job-2', startedAt: '2026-08-26T10:00:00Z', pipelineIds: ['face_analysis'] } })
      );
      vi.mocked(apiClient.getExtrasInstallJob).mockResolvedValue({
        jobId: 'job-2',
        extraName: 'face',
        status: 'running',
        createdAt: '2026-08-26T10:00:00Z',
        startedAt: '2026-08-26T10:00:01Z',
        finishedAt: null,
        commandOutput: null,
        restartRequired: false
      });

      renderWithProviders(
        <PipelineSelectionStep
          pipelines={[stubPipeline, facePipeline]}
          selectedPipelines={['stub']}
          setSelectedPipelines={vi.fn()}
          isLoading={false}
          error={null}
          onRetry={vi.fn()}
        />
      );

      await waitFor(() => {
        expect(screen.getByText(/Installing… this can take several minutes/)).toBeInTheDocument();
      });
      expect(screen.queryByRole('button', { name: 'Install' })).not.toBeInTheDocument();
    });
  });

  describe('progress, failure, and restart messaging (US3)', () => {
    it('shows failure with expandable command output, and a Retry action, on a failed job', async () => {
      (localStorage as unknown as MemoryStorage).setItem(
        'videoannotator_extras_install_jobs',
        JSON.stringify({ face: { jobId: 'job-3', startedAt: '2026-08-26T10:00:00Z', pipelineIds: ['face_analysis'] } })
      );
      vi.mocked(apiClient.getExtrasInstallJob).mockResolvedValue({
        jobId: 'job-3',
        extraName: 'face',
        status: 'failed',
        createdAt: '2026-08-26T10:00:00Z',
        startedAt: '2026-08-26T10:00:01Z',
        finishedAt: '2026-08-26T10:02:00Z',
        commandOutput: 'ERROR: could not find a version that satisfies the requirement torch',
        restartRequired: false
      });

      renderWithProviders(
        <PipelineSelectionStep
          pipelines={[stubPipeline, facePipeline]}
          selectedPipelines={['stub']}
          setSelectedPipelines={vi.fn()}
          isLoading={false}
          error={null}
          onRetry={vi.fn()}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Install failed.')).toBeInTheDocument();
      });
      expect(
        screen.getByText('ERROR: could not find a version that satisfies the requirement torch')
      ).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Retry install' })).toBeInTheDocument();
    });

    it('shows the restart-required banner (expected step, not an error) once a job completes, and keeps the pipeline locked', async () => {
      (localStorage as unknown as MemoryStorage).setItem(
        'videoannotator_extras_install_jobs',
        JSON.stringify({ face: { jobId: 'job-4', startedAt: '2026-08-26T10:00:00Z', pipelineIds: ['face_analysis'] } })
      );
      vi.mocked(apiClient.getExtrasInstallJob).mockResolvedValue({
        jobId: 'job-4',
        extraName: 'face',
        status: 'completed',
        createdAt: '2026-08-26T10:00:00Z',
        startedAt: '2026-08-26T10:00:01Z',
        finishedAt: '2026-08-26T10:04:00Z',
        commandOutput: null,
        restartRequired: true
      });

      renderWithProviders(
        <PipelineSelectionStep
          pipelines={[stubPipeline, facePipeline]}
          selectedPipelines={['stub']}
          setSelectedPipelines={vi.fn()}
          isLoading={false}
          error={null}
          onRetry={vi.fn()}
          restartRequired
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Server restart needed')).toBeInTheDocument();
      });
      // Wording is an expected next step, not error language.
      expect(screen.queryByText(/error/i)).not.toBeInTheDocument();
      // The pipeline itself is still shown locked - completion alone doesn't mean "ready".
      expect(screen.getByText('Face Analysis').closest('div[aria-disabled="true"]')).not.toBeNull();
    });

    it('renders a pipeline identically to an always-available one once the catalog reports it available (post-restart)', () => {
      renderWithProviders(
        <PipelineSelectionStep
          pipelines={[stubPipeline, { ...facePipeline, available: true, installHint: undefined }]}
          selectedPipelines={['stub']}
          setSelectedPipelines={vi.fn()}
          isLoading={false}
          error={null}
          onRetry={vi.fn()}
          restartRequired={false}
        />
      );

      expect(screen.queryByText('Server restart needed')).not.toBeInTheDocument();
      expect(screen.queryByText('Not installed')).not.toBeInTheDocument();
      expect(screen.getAllByRole('checkbox')).toHaveLength(2);
    });
  });
});
