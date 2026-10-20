/** Workflows saved and applied (spec 029 US4, T048). */
import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { getApiClient } from '@/api/client';
import Videos from '@/pages/Videos';
import { FINISHED_RUN, PROJECT_ID, RUN_SETTINGS, mockVideosApi, videosResponse } from '@/test/fixtures/videos';
import { renderAt } from '@/test/fixtures/renderPage';

const toast = vi.hoisted(() => Object.assign(vi.fn(), { success: vi.fn(), error: vi.fn() }));
vi.mock('sonner', () => ({ toast }));

const client = () => Object.getPrototypeOf(getApiClient());
const show = (address: string) => renderAt(address, [{ path: '/videos', element: <Videos /> }]);

beforeEach(() => vi.clearAllMocks());
afterEach(() => vi.restoreAllMocks());

describe('Save as workflow', () => {
  it("from a project: its settings, a name, and how results are offered", async () => {
    const api = mockVideosApi({ settings: RUN_SETTINGS.project });
    show(`/videos?project=${PROJECT_ID}`);
    await userEvent.click(await screen.findByRole('button', { name: /Save as workflow/ }));
    const name = screen.getByLabelText('Name');
    await userEvent.clear(name);
    await userEvent.type(name, 'Faces 40');
    await userEvent.selectOptions(screen.getByLabelText('Table format'), 'parquet');
    await userEvent.click(screen.getByRole('checkbox', { name: 'ELAN files' }));
    await userEvent.click(screen.getByRole('button', { name: 'Save workflow' }));

    await waitFor(() =>
      expect(api.createPreset).toHaveBeenCalledWith({
        name: 'Faces 40',
        selected_pipelines: ['face_analysis'],
        config: { face_analysis: { min_face_size: 40 } },
        export_choices: { table: 'parquet', elan: false, methods: true },
        source: { kind: 'project', id: PROJECT_ID, name: 'Peekaboo study' },
      }),
    );
    expect(toast.success).toHaveBeenCalledWith('Workflow Faces 40 saved', expect.anything());
  });

  it('from a run: the settings it ran with', async () => {
    const api = mockVideosApi({ videos: (q) => videosResponse(q, { runs: [{ ...FINISHED_RUN, job_ids: ['j1'] }] }) });
    vi.spyOn(client(), 'getJob').mockResolvedValue({ id: 'j1', selected_pipelines: ['scene_detection'], config: { scene_detection: { threshold: 0.4 } } });
    show('/videos?run=b1');
    await userEvent.click(await screen.findByRole('button', { name: /Save as workflow/ }));
    await userEvent.click(screen.getByRole('button', { name: 'Save workflow' }));
    await waitFor(() =>
      expect(api.createPreset).toHaveBeenCalledWith(
        expect.objectContaining({ name: 'First faces', selected_pipelines: ['scene_detection'], source: { kind: 'run', id: 'b1', name: 'First faces' } }),
      ),
    );
  });
});

describe('Apply workflow', () => {
  it("copies a workflow's settings into the project", async () => {
    const api = mockVideosApi({ settings: RUN_SETTINGS.project });
    show(`/videos?project=${PROJECT_ID}`);
    await userEvent.click(await screen.findByRole('button', { name: /Apply workflow/ }));
    await userEvent.click(await screen.findByRole('menuitem', { name: 'Faces 40' }));
    await waitFor(() =>
      expect(api.updateDataset).toHaveBeenCalledWith(PROJECT_ID, {
        current_settings: {
          selected_pipelines: ['face_analysis'],
          config: { face_analysis: { min_face_size: 40 } },
          source: { kind: 'workflow', id: 'w-faces40', name: 'Faces 40' },
        },
      }),
    );
    expect(toast.success).toHaveBeenCalledWith("Peekaboo study now uses Faces 40's settings", expect.anything());
  });
});

describe("a run whose project uses a workflow (T050)", () => {
  it('offers its results as the workflow says', async () => {
    mockVideosApi({ videos: (q) => videosResponse(q, { runs: [{ ...FINISHED_RUN }] }) });
    const { dataset, workflow } = await import('@/test/fixtures/videos');
    vi.spyOn(client(), 'getDataset').mockResolvedValue(
      dataset({ current_settings: { selected_pipelines: [], config: {}, source: { kind: 'workflow', id: 'w-faces40', name: 'Faces 40' } } }),
    );
    vi.spyOn(client(), 'listPresets').mockResolvedValue({
      presets: [workflow({ export_choices: { table: 'parquet', elan: false, methods: true } })],
      total: 1,
    });
    show('/videos?run=b1');
    const headline = await screen.findByTestId('run-headline');
    await waitFor(() => expect(within(headline).getByRole('button', { name: /Download \(Parquet\)/ })).toBeInTheDocument());
    expect(within(headline).queryByRole('button', { name: /ELAN/ })).toBeNull();
  });
});
