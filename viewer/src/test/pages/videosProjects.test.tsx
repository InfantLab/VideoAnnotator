/** Projects after the fact (spec 029 US2, T035-T037). */
import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import Videos from '@/pages/Videos';
import { OTHER_PROJECT_ID, PK, PROJECT_ID, dataset, mockVideosApi } from '@/test/fixtures/videos';
import { landed, renderAt } from '@/test/fixtures/renderPage';
import { demoVideoRuns, lanesLoaded, mockVideoApi, renderVideoPage, wideLayout } from '@/test/videoPageHarness';

const toast = vi.hoisted(() => Object.assign(vi.fn(), { success: vi.fn(), error: vi.fn() }));
vi.mock('sonner', () => ({ toast }));

const show = (address = '/videos') => renderAt(address, [{ path: '/videos', element: <Videos /> }]);
const pick = async (...names: string[]) => {
  await screen.findAllByTestId('video-row');
  for (const name of names) await userEvent.click(screen.getByRole('checkbox', { name: `Select ${name}` }));
};

beforeEach(() => vi.clearAllMocks());
afterEach(() => vi.restoreAllMocks());

describe('Add to project', () => {
  it('adds the selection to a project and says so', async () => {
    const api = mockVideosApi();
    show();
    await pick('p01_t1.mp4', 'p04_t1.mp4');
    await userEvent.click(within(screen.getByTestId('selection-bar')).getByRole('button', { name: /Add to project/ }));
    await userEvent.click(await screen.findByRole('menuitem', { name: 'QS2' }));

    await waitFor(() => expect(api.addProjectVideos).toHaveBeenCalledWith(OTHER_PROJECT_ID, [`${PK}/p01_t1.mp4`, `${PK}/p04_t1.mp4`]));
    expect(toast).toHaveBeenCalledWith('2 videos added to QS2', expect.anything());
    await waitFor(() => expect(screen.queryByTestId('selection-bar')).toBeNull());
  });

  it('makes a new project holding the selection', async () => {
    const api = mockVideosApi();
    show();
    await pick('p04_t1.mp4');
    await userEvent.click(within(screen.getByTestId('selection-bar')).getByRole('button', { name: /Add to project/ }));
    await userEvent.click(await screen.findByRole('menuitem', { name: /New project/ }));
    await userEvent.type(screen.getByLabelText('Project name'), 'QS');
    await userEvent.click(screen.getByRole('button', { name: 'Make project' }));

    await waitFor(() => expect(api.createDataset).toHaveBeenCalledWith({ name: 'QS', video_manifest: [], server_selection: true }));
    expect(api.addProjectVideos).toHaveBeenCalledWith('d-new', [`${PK}/p04_t1.mp4`]);
    expect(toast.success).toHaveBeenCalledWith('Project QS made', expect.objectContaining({ description: '1 video in it.' }));
  });

  it('makes an empty project from the left', async () => {
    const api = mockVideosApi();
    show();
    await userEvent.click(await screen.findByRole('button', { name: /New project/ }));
    await userEvent.type(screen.getByLabelText('Project name'), 'Pilot');
    await userEvent.click(screen.getByRole('button', { name: 'Make project' }));
    await waitFor(() => expect(api.createDataset).toHaveBeenCalled());
    expect(api.addProjectVideos).not.toHaveBeenCalled();
  });
});

describe('a project', () => {
  it('takes videos out, leaving them be', async () => {
    const api = mockVideosApi();
    show(`/videos?project=${PROJECT_ID}`);
    await pick('p02_t1.mp4');
    await userEvent.click(screen.getByRole('button', { name: 'Remove from project' }));
    await waitFor(() => expect(api.removeProjectVideos).toHaveBeenCalledWith(PROJECT_ID, [`${PK}/p02_t1.mp4`]));
    expect(toast).toHaveBeenCalledWith('1 video taken out of Peekaboo study', expect.objectContaining({ description: expect.stringContaining('stay') }));
  });

  it('shows its owner, runs, and new videos in its folder', async () => {
    const api = mockVideosApi();
    show(`/videos?project=${PROJECT_ID}`);
    expect(await screen.findByText(/made by caspar/)).toBeInTheDocument();
    expect(screen.getByTestId('project-runs')).toHaveTextContent('running 31/48');
    const fresh = screen.getByTestId('new-in-folder');
    expect(fresh).toHaveTextContent(`1 new video in ${PK}`);
    await userEvent.click(within(fresh).getByRole('button', { name: 'Add it' }));
    await waitFor(() => expect(api.addProjectVideos).toHaveBeenCalledWith(PROJECT_ID, [`${PK}/p49_t1.mp4`]));
  });

  it('says when it follows its folder', async () => {
    mockVideosApi({ datasets: [dataset({ server_selection: false })] });
    show(`/videos?project=${PROJECT_ID}`);
    expect(await screen.findByText(`All videos in ${PK}, including new ones`, { exact: false })).toBeInTheDocument();
  });

  it('renames', async () => {
    const api = mockVideosApi();
    show(`/videos?project=${PROJECT_ID}`);
    await userEvent.click(await screen.findByRole('button', { name: 'Rename' }));
    const field = screen.getByLabelText('Project name');
    await userEvent.clear(field);
    await userEvent.type(field, 'Peekaboo 2');
    await userEvent.click(screen.getByRole('button', { name: 'Rename' }));
    await waitFor(() => expect(api.updateDataset).toHaveBeenCalledWith(PROJECT_ID, { name: 'Peekaboo 2' }));
  });

  it('is removed with a confirmation that the videos and results stay', async () => {
    const api = mockVideosApi();
    const view = show(`/videos?project=${PROJECT_ID}`);
    await userEvent.click(await screen.findByRole('button', { name: /Remove project/ }));
    expect(screen.getByRole('alertdialog')).toHaveTextContent('Its videos and all their results stay');
    await userEvent.click(within(screen.getByRole('alertdialog')).getByRole('button', { name: 'Remove project' }));
    await waitFor(() => expect(api.deleteDataset).toHaveBeenCalledWith(PROJECT_ID));
    await waitFor(() => expect(landed(view.container)?.address ?? '/videos').toBe('/videos'));
  });
});

describe("the Video page's projects (T037)", () => {
  it('links each project holding the video to its list', async () => {
    const api = mockVideoApi();
    wideLayout();
    api.getVideoRuns.mockResolvedValue(demoVideoRuns({ projects: [{ id: PROJECT_ID, name: 'Peekaboo study' }, { id: OTHER_PROJECT_ID, name: 'QS2' }] }));
    renderVideoPage();
    await lanesLoaded();
    const projects = screen.getByTestId('video-projects');
    expect(within(projects).getByRole('link', { name: 'QS2' })).toHaveAttribute('href', `/videos?project=${OTHER_PROJECT_ID}`);
  });
});
