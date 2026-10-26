/** New run (spec 029 R9, FR-012, FR-017). */
import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { getApiClient } from '@/api/client';
import NewRun from '@/pages/NewRun';
import { runState, type RunStart } from '@/lib/runTarget';
import { wizardState } from '@/lib/wizardStart';
import { PK, PROJECT_ID, RUN_SETTINGS, mockVideosApi, videosResponse } from '@/test/fixtures/videos';
import { landed, renderAt } from '@/test/fixtures/renderPage';

const toast = vi.hoisted(() => Object.assign(vi.fn(), { success: vi.fn(), error: vi.fn() }));
vi.mock('sonner', () => ({ toast }));

const client = () => Object.getPrototypeOf(getApiClient());

function mockRunApi(settings = RUN_SETTINGS.defaults) {
  const api = mockVideosApi({ settings });
  return {
    ...api,
    getPipelineSchema: vi.spyOn(client(), 'getPipelineSchema').mockImplementation(async (id: string) => ({
      pipeline: { id, name: id },
      parameters:
        id === 'face_analysis'
          ? [{ name: 'min_face_size', type: 'number', default: 30, description: 'Smallest face to look for, in pixels' }]
          : [],
    })),
    validateConfig: vi.spyOn(client(), 'validateConfig').mockResolvedValue({ valid: true, errors: [], warnings: [] }),
    getCurrentUser: vi.spyOn(client(), 'getCurrentUser').mockResolvedValue({ id: 'u1', is_admin: false }),
    getExtrasGroups: vi.spyOn(client(), 'getExtrasGroups').mockResolvedValue([]),
    getIngestAccess: vi.spyOn(client(), 'getIngestAccess').mockResolvedValue({ can_read_in_place: false, same_machine: false, reason: 'Not shared.' }),
    rerunJob: vi.spyOn(client(), 'rerunJob').mockResolvedValue({ id: 'j-new' }),
    rerunBatch: vi.spyOn(client(), 'rerunBatch').mockResolvedValue({ batch_id: 'b-new', created: ['a', 'b'], skipped: [] }),
    getJob: vi.spyOn(client(), 'getJob').mockResolvedValue({ id: 'j1', selected_pipelines: ['scene_detection'], config: {} }),
    getJobs: vi.spyOn(client(), 'getJobs').mockResolvedValue({ jobs: [{ id: 'j1', selected_pipelines: ['speech_recognition'], config: {} }], total: 1 }),
    submitJob: vi.spyOn(client(), 'submitJob').mockImplementation(async (file: File) => ({ id: `job-${file.name}` })),
  };
}

const open = (start?: RunStart | object) =>
  renderAt('/run', [{ path: '/run', element: <NewRun /> }], start ? ('runStart' in start || 'wizardStart' in start ? start : runState(start as RunStart)) : undefined);
const startButton = () => screen.findByRole('button', { name: /Start run on/ });

beforeEach(() => vi.clearAllMocks());
afterEach(() => vi.restoreAllMocks());

describe('with a target, Start is the only confirmation', () => {
  it('a project: its videos, its settings, one press', async () => {
    const api = mockRunApi(RUN_SETTINGS.project);
    const view = open({ target: { kind: 'project', id: PROJECT_ID, name: 'Peekaboo study' } });

    const button = await startButton();
    expect(button).toHaveTextContent('Start run on 3 videos');
    expect(api.getRunSettings).toHaveBeenCalledWith({ project: PROJECT_ID, workflow: null });
    expect(screen.getByTestId('settings-source')).toHaveTextContent("Settings from: this project's settings, last changed");
    await waitFor(() => expect(button).toBeEnabled());
    fireEvent.click(button);

    await waitFor(() => expect(api.runVideos).toHaveBeenCalledTimes(1));
    expect(api.runVideos).toHaveBeenCalledWith({
      paths: [`${PK}/p01_t1.mp4`, `${PK}/p02_t1.mp4`, `${PK}/p04_t1.mp4`],
      selected_pipelines: ['face_analysis'],
      config: { face_analysis: { min_face_size: 40 } },
      name: undefined,
      project_id: PROJECT_ID,
    });
    await waitFor(() => expect(landed(view.container)?.address).toBe('/videos?run=b9'));
    expect(toast.success).toHaveBeenCalledWith(
      'Run started on 3 videos',
      expect.objectContaining({ description: expect.stringContaining('You can close this tab'), action: expect.objectContaining({ label: 'Show run' }) }),
    );
  });

  it('a selection: exactly those videos', async () => {
    const api = mockRunApi(RUN_SETTINGS.last_used);
    open({ target: { kind: 'videos', paths: ['/data/pk/a.mp4', '/elsewhere/b.mp4'], label: '2 videos' } });

    const button = await startButton();
    expect(button).toHaveTextContent('Start run on 2 videos');
    expect(screen.getByTestId('settings-source')).toHaveTextContent('Settings from: your last run (Yesterday)');
    await waitFor(() => expect(button).toBeEnabled());
    fireEvent.click(button);
    await waitFor(() => expect(api.runVideos).toHaveBeenCalled());
    expect(api.runVideos.mock.calls[0][0]).toMatchObject({ paths: ['/data/pk/a.mp4', '/elsewhere/b.mp4'], selected_pipelines: ['scene_detection'] });
  });

  it('a folder: listed by the server', async () => {
    const api = mockRunApi();
    open({ target: { kind: 'folder', path: PK, recursive: true } });

    await startButton();
    expect(api.listVideos).toHaveBeenCalledWith({ folder: PK, recursive: true });
    expect(screen.getByTestId('settings-source')).toHaveTextContent('Settings from: the defaults');
  });

  it('the defaults tick nothing (UAT #42)', async () => {
    mockRunApi();
    open({ target: { kind: 'videos', paths: ['/a.mp4'], label: 'a.mp4' } });

    expect(await startButton()).toBeDisabled();
    expect(screen.getByTestId('start-run')).toHaveTextContent('Nothing chosen yet');
    expect(screen.getByTestId('settings-source')).toHaveTextContent('nothing ticked until you choose');
  });

  it('asks before a big run, saying its size (UAT #42)', async () => {
    const api = mockRunApi({ ...RUN_SETTINGS.project, selected_pipelines: ['face_analysis', 'scene_detection'] });
    const paths = Array.from({ length: 21 }, (_, i) => `/v${i}.mp4`);
    open({ target: { kind: 'videos', paths, label: '21 videos' } });

    const button = await startButton();
    await waitFor(() => expect(button).toBeEnabled());
    fireEvent.click(button);
    const dialog = await screen.findByRole('alertdialog');
    expect(dialog).toHaveTextContent('Run 2 pipelines on 21 videos?');
    expect(dialog).toHaveTextContent("That's 42 runs");
    expect(api.runVideos).not.toHaveBeenCalled();

    fireEvent.click(within(dialog).getByRole('button', { name: 'Start run on 21 videos' }));
    await waitFor(() => expect(api.runVideos).toHaveBeenCalledTimes(1));
  });

  it('a run again: its own settings, and the rerun route', async () => {
    const api = mockRunApi();
    open({ target: { kind: 'rerun', runId: 'b1', label: 'Pilot', single: false, videoCount: 2 } });

    const button = await startButton();
    expect(button).toHaveTextContent('Start run on 2 videos');
    await waitFor(() => expect(screen.getByTestId('settings-source')).toHaveTextContent('the run “Pilot”'));
    expect(api.getRunSettings).not.toHaveBeenCalled();
    await waitFor(() => expect(button).toBeEnabled());
    fireEvent.click(button);
    await waitFor(() => expect(api.rerunBatch).toHaveBeenCalledWith('b1', { selected_pipelines: ['speech_recognition'], config: {} }));
  });

  it('an old wizard address still lands: Run again on one video', async () => {
    const api = mockRunApi();
    open(wizardState({ mode: 'rerun', jobId: 'j1', label: 'clip.mp4', selectedPipelines: ['scene_detection'], config: {} }));

    const button = await startButton();
    expect(button).toHaveTextContent('Start run on 1 video');
    await waitFor(() => expect(button).toBeEnabled());
    fireEvent.click(button);
    await waitFor(() => expect(api.rerunJob).toHaveBeenCalledWith('j1', { selected_pipelines: ['scene_detection'], config: {} }));
  });

  it('a workflow chosen at the top', async () => {
    const api = mockRunApi();
    open({ target: { kind: 'videos', paths: ['/a.mp4'], label: 'a.mp4' } });

    const picker = await screen.findByLabelText('Workflow');
    await userEvent.selectOptions(picker, 'w-faces40');

    await waitFor(() => expect(api.getRunSettings).toHaveBeenCalledWith({ project: null, workflow: 'w-faces40' }));
  });

  it('nothing chosen to find: Start says so', async () => {
    mockRunApi({ ...RUN_SETTINGS.defaults, selected_pipelines: ['no_such_pipeline'] });
    open({ target: { kind: 'videos', paths: ['/a.mp4'], label: 'a.mp4' } });

    await waitFor(() => expect(screen.getByText(/no_such_pipeline: not on this server/)).toBeInTheDocument());
    expect(screen.getByTestId('start-run')).toHaveTextContent('Choose at least one thing to find.');
    expect(await startButton()).toBeDisabled();
  });
});

describe('without a target', () => {
  it('asks for the videos first; a project chooses its videos', async () => {
    mockRunApi();
    open();
    await userEvent.click(await screen.findByRole('tab', { name: /A project/ }));
    await userEvent.click(await screen.findByRole('button', { name: /Peekaboo study · 48 videos/ }));

    expect(await startButton()).toHaveTextContent('Start run on 3 videos');
  });

  it('uploads videos from this computer as one run', async () => {
    const api = mockRunApi(RUN_SETTINGS.last_used);
    const view = open();
    await userEvent.click(await screen.findByRole('tab', { name: /This computer/ }));
    const input = screen.getByTestId('upload-input') as HTMLInputElement;
    const files = [new File(['a'], 'a.mp4'), new File(['b'], 'b.mp4')];
    fireEvent.change(input, { target: { files } });
    await userEvent.click(screen.getByRole('button', { name: 'Use 2 videos' }));

    const button = await startButton();
    await waitFor(() => expect(button).toBeEnabled());
    fireEvent.click(button);

    await waitFor(() => expect(api.submitJob).toHaveBeenCalledTimes(2));
    const runIds = api.submitJob.mock.calls.map((call) => (call[3] as { id: string }).id);
    expect(new Set(runIds).size).toBe(1);
    await waitFor(() => expect(landed(view.container)?.address).toBe(`/videos?run=${runIds[0]}`));
  });

  it('a video ran before', async () => {
    const api = mockRunApi();
    open();
    await userEvent.click(await screen.findByRole('tab', { name: /A video you ran before/ }));
    await userEvent.click(await screen.findByRole('button', { name: /p01_t1.mp4/ }));

    expect(await startButton()).toHaveTextContent('Start run on 1 video');
    expect(api.listVideos).toHaveBeenCalledWith({});
  });
});

it('a server error is said in the viewer\'s words', async () => {
  const api = mockRunApi(RUN_SETTINGS.last_used);
  const { APIError } = await import('@/api/handleError');
  api.runVideos.mockRejectedValue(new APIError('x', 422, undefined, { error: { code: 'NOTHING_TO_RUN' } }));
  open({ target: { kind: 'videos', paths: ['/a.mp4'], label: 'a.mp4' } });
  const button = await startButton();
  await waitFor(() => expect(button).toBeEnabled());
  fireEvent.click(button);
  expect(await screen.findByRole('alert')).toHaveTextContent('None of these videos can be run');
  expect(within(screen.getByTestId('start-run')).getByRole('button', { name: /Start run on/ })).toBeEnabled();
  expect(videosResponse().videos.length).toBe(4);
});
