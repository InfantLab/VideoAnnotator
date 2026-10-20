/** Projects remember their settings (spec 029 US3, T039-T041). */
import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { getApiClient } from '@/api/client';
import Videos from '@/pages/Videos';
import { FINISHED_RUN, PROJECT_ID, RUN_SETTINGS, mockVideosApi, run, videosResponse } from '@/test/fixtures/videos';
import { renderAt } from '@/test/fixtures/renderPage';

const toast = vi.hoisted(() => Object.assign(vi.fn(), { success: vi.fn(), error: vi.fn() }));
vi.mock('sonner', () => ({ toast }));

const client = () => Object.getPrototypeOf(getApiClient());
const show = (address: string) => renderAt(address, [{ path: '/videos', element: <Videos /> }]);

beforeEach(() => vi.clearAllMocks());
afterEach(() => vi.restoreAllMocks());

function settingsApi() {
  const api = mockVideosApi({ settings: RUN_SETTINGS.project });
  vi.spyOn(client(), 'getPipelineSchema').mockResolvedValue({ pipeline: { id: 'face_analysis', name: 'f' }, parameters: [] });
  vi.spyOn(client(), 'validateConfig').mockResolvedValue({ valid: true, errors: [], warnings: [] });
  vi.spyOn(client(), 'getCurrentUser').mockResolvedValue({ id: 'u1', is_admin: false });
  vi.spyOn(client(), 'getExtrasGroups').mockResolvedValue([]);
  return api;
}

describe("a project's settings", () => {
  it('shows them and where they came from', async () => {
    settingsApi();
    show(`/videos?project=${PROJECT_ID}`);
    const line = await screen.findByTestId('project-settings');
    expect(line).toHaveTextContent('Settings: Faces');
    expect(line).toHaveTextContent('from the run First faces');
  });

  it('shows Ask AI\'s question when it is in them', async () => {
    const api = settingsApi();
    api.getRunSettings.mockResolvedValue({
      ...RUN_SETTINGS.project,
      selected_pipelines: ['vlm_annotation'],
      config: { vlm_annotation: { prompt: 'Is the baby smiling?' } },
    });
    show(`/videos?project=${PROJECT_ID}`);
    expect(await screen.findByTestId('project-settings')).toHaveTextContent('“Is the baby smiling?”');
  });

  it('Edit saves them without running', async () => {
    const api = settingsApi();
    show(`/videos?project=${PROJECT_ID}`);
    await userEvent.click(within(await screen.findByTestId('project-settings')).getByRole('button', { name: 'Edit' }));
    await userEvent.click(await screen.findByRole('button', { name: 'Save settings' }));

    await waitFor(() =>
      expect(api.updateDataset).toHaveBeenCalledWith(PROJECT_ID, {
        current_settings: { selected_pipelines: ['face_analysis'], config: { face_analysis: { min_face_size: 40 } }, source: { kind: 'edit' } },
      }),
    );
    expect(api.runVideos).not.toHaveBeenCalled();
    expect(toast.success).toHaveBeenCalledWith('Settings saved for Peekaboo study', expect.objectContaining({ description: 'The next run starts from them.' }));
  });
});

describe('What changed?', () => {
  it('lists the settings that differ between two runs', async () => {
    settingsApi();
    vi.spyOn(client(), 'getJob').mockImplementation(async (id: string) =>
      id === 'j-old'
        ? { id, selected_pipelines: ['face_analysis'], config: { face_analysis: { min_face_size: 30 } } }
        : { id, selected_pipelines: ['face_analysis', 'speech_recognition'], config: { face_analysis: { min_face_size: 40 } } },
    );
    const older = { ...FINISHED_RUN, job_ids: ['j-old'] };
    const newer = run({ job_ids: ['j-new'] });
    mockVideosApi({ settings: RUN_SETTINGS.project, videos: (q) => ({ ...videosResponse(q), runs: [newer, older] }) });
    show(`/videos?project=${PROJECT_ID}`);
    await userEvent.click(await screen.findByRole('button', { name: /What changed/ }));
    const changes = await screen.findByTestId('settings-changes');
    await waitFor(() => expect(changes).toHaveTextContent('Faces · min_face_size: 30 → 40'));
    expect(changes).toHaveTextContent('Speech: added');
  });
});
