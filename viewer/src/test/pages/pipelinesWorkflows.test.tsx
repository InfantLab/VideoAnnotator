/** Pipelines → Workflows (spec 029 FR-016, T049). */
import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { getApiClient } from '@/api/client';
import Pipelines from '@/pages/Pipelines';
import { PROJECT_ID, mockVideosApi, workflow } from '@/test/fixtures/videos';
import { landed, renderAt } from '@/test/fixtures/renderPage';
import { workflowFile } from '@/lib/workflows';

const toast = vi.hoisted(() => Object.assign(vi.fn(), { success: vi.fn(), error: vi.fn() }));
vi.mock('sonner', () => ({ toast }));
const saved = vi.hoisted(() => vi.fn());
vi.mock('@/lib/exports', async (original) => ({ ...(await original<typeof import('@/lib/exports')>()), saveBlob: saved }));

const client = () => Object.getPrototypeOf(getApiClient());
const show = () => renderAt('/pipelines', [{ path: '/pipelines', element: <Pipelines /> }]);

beforeEach(() => {
  vi.clearAllMocks();
  vi.spyOn(client(), 'getCurrentUser').mockResolvedValue({ id: 'u1', is_admin: false });
  vi.spyOn(client(), 'getExtrasGroups').mockResolvedValue([]);
});
afterEach(() => vi.restoreAllMocks());

const card = async () => (await screen.findAllByTestId('workflow'))[0];

describe('Workflows', () => {
  it('lists each with its pipelines and where it came from', async () => {
    mockVideosApi();
    show();
    const one = await card();
    expect(one).toHaveTextContent('Faces 40');
    expect(one).toHaveTextContent('Faces');
    expect(one).toHaveTextContent('from the project Peekaboo study');
  });

  it('applies to a project, and is used in a new run', async () => {
    const api = mockVideosApi();
    const view = show();
    const one = await card();
    await userEvent.click(within(one).getByRole('button', { name: /Apply to project/ }));
    await userEvent.click(await screen.findByRole('menuitem', { name: 'Peekaboo study' }));
    await waitFor(() => expect(api.updateDataset).toHaveBeenCalledWith(PROJECT_ID, expect.objectContaining({ current_settings: expect.objectContaining({ source: { kind: 'workflow', id: 'w-faces40', name: 'Faces 40' } }) })));
    await userEvent.click(within(await card()).getByRole('button', { name: /Use in a new run/ }));
    expect(landed(view.container)).toEqual({ address: '/run', state: { runStart: { target: null, workflowId: 'w-faces40' } } });
  });

  it('exports a file that imports as it was', async () => {
    const api = mockVideosApi();
    show();
    await userEvent.click(within(await card()).getByRole('button', { name: /Export/ }));
    const [, name] = saved.mock.calls[0] as [Blob, string];
    expect(name).toBe('Faces 40.workflow.json');
    const text = JSON.stringify(workflowFile(workflow()));
    fireEvent.change(screen.getByTestId('workflow-import'), { target: { files: [{ name: 'f.json', text: async () => text }] } });
    await waitFor(() => expect(api.createPreset).toHaveBeenCalledWith({
      name: 'Faces 40',
      selected_pipelines: ['face_analysis'],
      config: { face_analysis: { min_face_size: 40 } },
      export_choices: { table: 'csv', elan: true, methods: true },
      source: { kind: 'project', id: PROJECT_ID, name: 'Peekaboo study' },
    }));
    expect(toast.success).toHaveBeenCalledWith('Workflow Faces 40 imported', expect.anything());
  });

  it('renames and deletes', async () => {
    const api = mockVideosApi();
    show();
    await userEvent.click(within(await card()).getByRole('button', { name: /Rename/ }));
    const field = screen.getByLabelText('Workflow name');
    await userEvent.clear(field);
    await userEvent.type(field, 'Faces 50');
    await userEvent.click(screen.getByRole('button', { name: 'Rename' }));
    await waitFor(() => expect(api.updatePreset).toHaveBeenCalledWith('w-faces40', { name: 'Faces 50' }));
    await userEvent.click(within(await card()).getByRole('button', { name: /Delete/ }));
    await userEvent.click(within(screen.getByRole('alertdialog')).getByRole('button', { name: 'Delete' }));
    await waitFor(() => expect(api.deletePreset).toHaveBeenCalled());
  });

  it('a preset saved before is listed as a workflow', async () => {
    mockVideosApi({ presets: [workflow({ name: 'Old preset', source: null, export_choices: null })] });
    show();
    expect(await card()).toHaveTextContent('Old preset');
  });
});
