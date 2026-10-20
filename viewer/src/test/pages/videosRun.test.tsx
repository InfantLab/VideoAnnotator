/** A run as a filter of Videos (spec 029 R12, T030). */
import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { getApiClient } from '@/api/client';
import Videos from '@/pages/Videos';
import { FINISHED_RUN, mockVideosApi, run, videosResponse } from '@/test/fixtures/videos';
import { landed, renderAt } from '@/test/fixtures/renderPage';
import { demoRun, demoVideoRuns } from '@/test/videoPageHarness';
import type { RunSummary } from '@/types/videos';

const client = () => Object.getPrototypeOf(getApiClient());
const show = (id: string, summary: RunSummary) => {
  const api = mockVideosApi({ videos: (q) => videosResponse(q, { runs: [summary] }) });
  const view = renderAt(`/videos?run=${id}`, [{ path: '/videos', element: <Videos /> }]);
  return { api, view };
};

vi.mock('@/lib/exports', async (original) => ({ ...(await original<typeof import('@/lib/exports')>()), saveBlob: vi.fn() }));

afterEach(() => vi.restoreAllMocks());

describe('a run going on', () => {
  it('shows its progress, and Cancel stops it', async () => {
    const cancel = vi.spyOn(client(), 'cancelBatch').mockResolvedValue({ batch_id: 'b2', cancelled: [], skipped: [] });
    show('b2', run());
    const header = await screen.findByTestId('run-header');
    expect(within(header).getByRole('heading')).toHaveTextContent('Peekaboo faces');
    expect(within(header).getByTestId('run-state')).toHaveTextContent('running 31/48');
    expect(within(header).getByRole('progressbar', { name: 'Run progress' })).toBeInTheDocument();
    expect(within(header).queryByTestId('run-headline')).toBeNull();
    await userEvent.click(within(header).getByRole('button', { name: /Cancel/ }));
    expect(cancel).toHaveBeenCalledWith('b2');
  });

  it("a run of one video is cancelled as that video's", async () => {
    const cancel = vi.spyOn(client(), 'cancelJob').mockResolvedValue({ job_id: 'j1', status: 'cancelled', message: '' });
    show('j1', run({ id: 'j1', job_ids: ['j1'], total: 1, done: 0, running: 1, waiting: 0 }));
    await userEvent.click(within(await screen.findByTestId('run-header')).getByRole('button', { name: /Cancel/ }));
    expect(cancel).toHaveBeenCalledWith('j1');
  });
});

describe('a finished run', () => {
  it('leads with its results table, ELAN and methods', async () => {
    const { api } = show('b1', FINISHED_RUN);
    const headline = await screen.findByTestId('run-headline');
    expect(headline).toHaveTextContent('48 videos → results table');
    await userEvent.click(within(headline).getByRole('button', { name: /Download/ }));
    expect(api.exportResults).toHaveBeenCalledWith({ kind: 'run', id: 'b1' }, 'table', 'csv');
    await userEvent.click(within(headline).getByRole('button', { name: /ELAN/ }));
    expect(api.exportResults).toHaveBeenCalledWith({ kind: 'run', id: 'b1' }, 'elan', 'csv');
    expect(within(headline).getByRole('button', { name: /Methods/ })).toBeInTheDocument();
  });

  it('retries failed videos, runs again and deletes', async () => {
    const retry = vi.spyOn(client(), 'retryBatch').mockResolvedValue({ batch_id: 'b1', retried: [], skipped: [] });
    const { view } = show('b1', FINISHED_RUN);
    const header = await screen.findByTestId('run-header');
    await userEvent.click(within(header).getByRole('button', { name: /Retry failed videos/ }));
    expect(retry).toHaveBeenCalledWith('b1');
    expect(within(header).getByRole('button', { name: /Delete run/ })).toBeInTheDocument();
    await userEvent.click(within(header).getByRole('button', { name: /Run again…/ }));
    expect(landed(view.container)).toEqual({
      address: '/run',
      state: { runStart: { target: { kind: 'rerun', runId: 'b1', label: 'First faces', single: false, videoCount: 48 } } },
    });
  });

  it('compares Ask AI answers with an earlier run of the same video', async () => {
    const vlm = { ...FINISHED_RUN, pipelines: ['vlm_annotation'], job_ids: ['job-1'] };
    vi.spyOn(client(), 'getVideoRuns').mockResolvedValue(
      demoVideoRuns({ runs: [demoRun('job-1'), demoRun('job-0', { created_at: '2026-10-01T09:00:00Z' })] }),
    );
    const { view } = show('b1', vlm);
    await userEvent.click(within(await screen.findByTestId('run-headline')).getByRole('button', { name: /Compare answers/ }));
    await waitFor(() => expect(landed(view.container)?.address).toBe('/compare?a=job-0&b=job-1'));
  });
});
