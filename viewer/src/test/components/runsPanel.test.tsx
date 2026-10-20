/** The Video page's Runs tab: what Job detail showed, per run (spec 029 R12, T042). */
import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { getApiClient } from '@/api/client';
import { demoRun, demoVideoRuns, lanesLoaded, mockVideoApi, renderVideoPage, wideLayout } from '@/test/videoPageHarness';

const toast = vi.hoisted(() => Object.assign(vi.fn(), { success: vi.fn(), error: vi.fn() }));
vi.mock('sonner', () => ({ toast }));
vi.mock('@/lib/exports', async (original) => ({ ...(await original<typeof import('@/lib/exports')>()), saveBlob: vi.fn() }));

const client = () => Object.getPrototypeOf(getApiClient());

let api: ReturnType<typeof mockVideoApi>;
beforeEach(() => {
  api = mockVideoApi();
  wideLayout();
  vi.spyOn(client(), 'getIngestAccess').mockResolvedValue({ can_read_in_place: false, same_machine: false, can_open_folders: false });
  vi.spyOn(client(), 'getPipelineCatalog').mockResolvedValue({ catalog: { pipelines: [] }, server: { version: 'x' }, restartRequired: false });
});
afterEach(() => vi.restoreAllMocks());

async function openRuns() {
  renderVideoPage();
  await lanesLoaded();
  fireEvent.mouseDown(screen.getByRole('tab', { name: /Runs/ }));
  await userEvent.click(screen.getByRole('tab', { name: /Runs/ }));
  return screen.findByTestId('runs-panel');
}

describe('the Runs tab', () => {
  it('lists every run, newest first, with each pipeline and its error, settings and results folder', async () => {
    api.getVideoRuns.mockResolvedValue(
      demoVideoRuns({
        runs: [
          demoRun('job-2', {
            run_name: 'Second (2026-10-20)',
            batch_id: 'b2',
            batch_name: 'Second',
            config: { face_analysis: { min_face_size: 40 } },
            results_folder: { path: '/home/a/VideoAnnotator/Second', display_path: '~/VideoAnnotator/Second', exists: true },
            pipelines: [
              { name: 'face_analysis', status: 'completed', error: null, files: [] },
              { name: 'speech_recognition', status: 'failed', error: 'No audio track', files: [] },
            ],
          }),
          demoRun('job-1', { run_name: 'First', created_at: '2026-10-01T09:00:00Z' }),
        ],
      }),
    );
    const panel = await openRuns();
    const cards = within(panel).getAllByTestId('run-card');
    expect(cards[0]).toHaveTextContent('Second (2026-10-20)');
    expect(cards[0]).toHaveTextContent('No audio track');
    expect(cards[0]).toHaveTextContent('Face analysis · min_face_size: 40');
    expect(within(cards[0]).getByRole('link', { name: 'Second' })).toHaveAttribute('href', '/videos?run=b2');
    expect(cards[0]).toHaveTextContent('~/VideoAnnotator/Second');
    expect(within(cards[0]).getByRole('button', { name: /Download all files/ })).toBeInTheDocument();
    expect(within(cards[0]).getByRole('button', { name: /Delete run/ })).toBeInTheDocument();
    expect(within(cards[0]).getByRole('button', { name: 'Run again…' })).toBeInTheDocument();
    expect(cards[1]).toHaveTextContent('First');
  });

  it('a run waiting its turn: its place in the queue, and Cancel', async () => {
    const cancel = vi.spyOn(client(), 'cancelJob').mockResolvedValue({ job_id: 'job-3', status: 'cancelled', message: '' });
    api.getVideoRuns.mockResolvedValue(
      demoVideoRuns({ runs: [demoRun('job-3', { status: 'pending', progress: 0, queue_position: 2, pipelines: [], selected_pipelines: ['face_analysis'] }), demoRun()] }),
    );
    const panel = await openRuns();
    const [card] = within(panel).getAllByTestId('run-card');
    expect(card).toHaveTextContent('2nd in line');
    expect(card).toHaveTextContent('waiting');
    await userEvent.click(within(card).getByRole('button', { name: /Cancel/ }));
    await waitFor(() => expect(cancel).toHaveBeenCalledWith('job-3'));
  });

  it('downloads all of a run\'s files', async () => {
    const artifacts = vi.spyOn(client(), 'getJobArtifacts').mockResolvedValue(new Response('zip'));
    const panel = await openRuns();
    await userEvent.click(within(panel).getByRole('button', { name: /Download all files/ }));
    await waitFor(() => expect(artifacts).toHaveBeenCalledWith('job-1'));
  });
});
