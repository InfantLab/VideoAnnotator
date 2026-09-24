// Readiness-driven pipeline cards (VideoAnnotator spec 011, viewer-handoff.md item 1):
// every state renders, only `ready` is selectable, sizes and shared groups show,
// setup blockers are display-only, and older servers fall back to `available`.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import React from 'react';
import { PipelineSelectionStep } from '@/pages/NewJob';
import { apiClient } from '@/api/client';
import { pipelineCardMode } from '@/lib/pipelineExtras';
import type { PipelineDescriptor, PipelineReadiness } from '@/types/pipelines';

vi.mock('@/api/client', async () => {
  const actual = await vi.importActual<typeof import('@/api/client')>('@/api/client');
  return {
    ...actual,
    hasConfiguredApiToken: vi.fn(() => true),
    apiClient: {
      installPipelineExtras: vi.fn(),
      getExtrasInstallJob: vi.fn(),
      getCurrentUser: vi.fn(),
      getExtrasGroups: vi.fn(),
      getPipelineCatalog: vi.fn().mockResolvedValue({ pipelines: [], restartRequired: false }),
      clearPipelineCache: vi.fn(),
      clearServerInfoCache: vi.fn()
    }
  };
});

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

const readiness = (overrides: Partial<PipelineReadiness>): PipelineReadiness => ({
  state: 'ready',
  nextAction: 'none',
  extrasGroup: 'audio',
  installJobId: null,
  blockers: [],
  notes: [],
  ...overrides
});

const pipeline = (id: string, name: string, r?: Partial<PipelineReadiness>, extra: Partial<PipelineDescriptor> = {}): PipelineDescriptor => ({
  id,
  name,
  group: 'audio',
  available: r?.state === 'not_installed' ? false : true,
  readiness: r ? readiness(r) : undefined,
  ...extra
});

const renderStep = (pipelines: PipelineDescriptor[]) => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <PipelineSelectionStep
          pipelines={pipelines}
          selectedPipelines={[]}
          setSelectedPipelines={() => {}}
          isLoading={false}
          error={null}
          onRetry={() => {}}
        />
      </MemoryRouter>
    </QueryClientProvider>
  );
};

const card = (name: string) => screen.getByText(name, { selector: 'span' }).closest('div[data-locked="true"], label') as HTMLElement;

describe('Pipeline readiness cards', () => {
  beforeEach(() => {
    vi.stubGlobal('localStorage', new MemoryStorage());
    vi.clearAllMocks();
    vi.mocked(apiClient.getCurrentUser).mockResolvedValue({ id: 'u', username: 'a', email: 'a@b.c', isAdmin: true });
    vi.mocked(apiClient.getExtrasGroups).mockResolvedValue([
      {
        name: 'audio',
        pipelines: ['speech_recognition', 'speaker_diarization'],
        installed: false,
        approxDownloadMb: 550,
        includesGpuTorch: false,
        installJobId: null
      }
    ]);
  });

  it('shows the group, its size and what else it enables on a not-installed card', async () => {
    renderStep([
      pipeline('speech_recognition', 'Speech Recognition', { state: 'not_installed', nextAction: 'install' }),
      pipeline('speaker_diarization', 'Speaker Diarization', { state: 'not_installed', nextAction: 'install' })
    ]);

    const speech = card('Speech Recognition');
    expect(within(speech).getByText('Not installed')).toBeInTheDocument();
    expect(await within(speech).findByText(/approx\. 550 MB/)).toBeInTheDocument();
    expect(within(speech).getByText(/which also enables Speaker Diarization/)).toBeInTheDocument();
    expect(within(speech).getByRole('button', { name: 'Install' })).toBeInTheDocument();
    expect(within(speech).queryByRole('checkbox')).not.toBeInTheDocument();
  });

  it('makes only ready pipelines selectable, with their notes shown quietly', () => {
    renderStep([
      pipeline('speaker_diarization', 'Speaker Diarization', {
        notes: [
          {
            kind: 'licence',
            name: 'pyannote/speaker-diarization-3.1',
            message: 'Accept the pyannote licence on Hugging Face.',
            helpUrl: 'https://huggingface.co/pyannote/speaker-diarization-3.1'
          }
        ]
      })
    ]);

    const c = card('Speaker Diarization');
    expect(within(c).getByRole('checkbox')).toBeInTheDocument();
    expect(within(c).getByText('Accept the pyannote licence on Hugging Face.')).toBeInTheDocument();
    expect(within(c).getByRole('link', { name: 'Open the model page' })).toHaveAttribute(
      'href',
      'https://huggingface.co/pyannote/speaker-diarization-3.1'
    );
  });

  it('explains a missing secret without offering to take its value', () => {
    renderStep([
      pipeline('speaker_diarization', 'Speaker Diarization', {
        state: 'needs_setup',
        nextAction: 'setup',
        blockers: [
          {
            kind: 'secret',
            name: 'HF_AUTH_TOKEN',
            message: "A Hugging Face access token isn't set. Set HF_AUTH_TOKEN in the server's environment.",
            helpUrl: 'https://huggingface.co/settings/tokens'
          }
        ]
      })
    ]);

    const c = card('Speaker Diarization');
    expect(within(c).getByText('Needs setup')).toBeInTheDocument();
    expect(within(c).getByText(/Set HF_AUTH_TOKEN in the server's environment/)).toBeInTheDocument();
    expect(within(c).getByRole('link', { name: 'Get a token' })).toBeInTheDocument();
    expect(within(c).getByRole('button', { name: 'Check again' })).toBeInTheDocument();
    expect(within(c).queryByRole('textbox')).not.toBeInTheDocument();
    expect(within(c).queryByRole('checkbox')).not.toBeInTheDocument();
  });

  it('says an import failure is for the administrator', () => {
    renderStep([
      pipeline('face_analysis', 'Face Analysis', {
        state: 'needs_setup',
        extrasGroup: 'face',
        blockers: [{ kind: 'import_error', name: 'face_analysis', message: 'The pipeline failed to load: ImportError: libGL' }]
      })
    ]);
    expect(screen.getByText(/libGL/)).toHaveTextContent('The server administrator needs to look at this.');
  });

  it('shows restart-needed pipelines as installed but locked', () => {
    renderStep([pipeline('person_tracking', 'Person Tracking', { state: 'restart_required', nextAction: 'restart', extrasGroup: 'person' })]);
    const c = card('Person Tracking');
    expect(within(c).getByText('Restart needed')).toBeInTheDocument();
    expect(within(c).getByText('Installed. Restart the server to activate it.')).toBeInTheDocument();
  });

  it('picks up an install the server reports, even one this browser did not start', async () => {
    vi.mocked(apiClient.getExtrasInstallJob).mockResolvedValue({
      jobId: 'job-42',
      extraName: 'audio',
      status: 'running',
      createdAt: '2026-09-24T10:00:00Z',
      startedAt: '2026-09-24T10:00:01Z',
      finishedAt: null,
      commandOutput: null,
      restartRequired: false
    });

    renderStep([pipeline('speech_recognition', 'Speech Recognition', { state: 'installing', nextAction: 'wait', installJobId: 'job-42' })]);

    const c = card('Speech Recognition');
    expect(within(c).getByText('Installing')).toBeInTheDocument();
    expect(within(c).getByText(/Installing…|Installing&hellip;|this can take several minutes/)).toBeInTheDocument();
    await waitFor(() => expect(apiClient.getExtrasInstallJob).toHaveBeenCalledWith('job-42'));
  });

  it('renders a state it does not know as not available, with the server message', () => {
    renderStep([
      pipeline('x', 'Future Pipeline', {
        state: 'quarantined',
        nextAction: 'contact_admin',
        blockers: [{ kind: 'policy', name: 'x', message: 'Disabled by policy.' }]
      })
    ]);
    const c = card('Future Pipeline');
    expect(within(c).getByText('Not available')).toBeInTheDocument();
    expect(within(c).getByText('Disabled by policy.')).toBeInTheDocument();
  });

  it('falls back to available/install_hint on servers without readiness', () => {
    renderStep([
      pipeline('face_analysis', 'Face Analysis', undefined, {
        available: false,
        installHint: 'pip install videoannotator[face]'
      })
    ]);
    const c = card('Face Analysis');
    expect(within(c).getByText('Not installed')).toBeInTheDocument();
    expect(within(c).getByText('pip install videoannotator[face]')).toBeInTheDocument();
  });
});

describe('pipelineCardMode', () => {
  it.each([
    ['ready', 'selectable'],
    ['not_installed', 'install'],
    ['installing', 'installing'],
    ['restart_required', 'restart'],
    ['needs_setup', 'setup'],
    ['something_new', 'unavailable']
  ])('%s -> %s', (state, mode) => {
    expect(pipelineCardMode(pipeline('p', 'P', { state }))).toBe(mode);
  });

  it('uses available when there is no readiness', () => {
    expect(pipelineCardMode({ id: 'p', name: 'P', available: false })).toBe('install');
    expect(pipelineCardMode({ id: 'p', name: 'P' })).toBe('selectable');
  });
});
