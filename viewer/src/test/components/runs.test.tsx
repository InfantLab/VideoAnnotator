/** Spec 028 US5: which run, and comparing runs. */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { FIXTURES, demoRun, demoVideoRuns, lanesLoaded, mockVideoApi, renderVideoPage, VIDEO_PATH, wideLayout } from '@/test/videoPageHarness';

const OLD = demoRun('job-1', { run_name: 'Pilot (old)', created_at: '2026-10-01T09:00:00Z', completed_at: '2026-10-01T09:30:00Z' });
const NEW = demoRun('job-2', { run_name: 'Pilot (new)', created_at: '2026-10-12T09:00:00Z', completed_at: '2026-10-12T09:30:00Z' });

const lanes = (kind: string) => screen.getAllByRole('group').filter((g) => g.dataset.kind === kind);
const runOf = (g: HTMLElement) => within(g).getByTestId('lane-run').textContent;

let api: ReturnType<typeof mockVideoApi>;

beforeEach(() => {
  api = mockVideoApi(demoVideoRuns({ runs: [NEW, OLD] }));
  // The older run diarized with a different minimum number of speakers.
  api.downloadResultFile.mockImplementation(async (url: string, name: string) => {
    let text = readFileSync(path.join(FIXTURES, name), 'utf8');
    if (url.includes('/jobs/job-1/') && name.endsWith('.provenance.json')) {
      const record = JSON.parse(text);
      record.settings.min_speakers = 2;
      text = JSON.stringify(record);
    }
    return new File([text], name);
  });
  wideLayout();
});
afterEach(() => vi.restoreAllMocks());

describe('runs on a video', () => {
  it('shows the newest completed run by default, and names it', async () => {
    renderVideoPage();
    await lanesLoaded();
    expect(new Set(screen.getAllByTestId('lane-run').map((r) => r.textContent))).toEqual(new Set(['Pilot (new)']));
  });

  it('starts on the run named in the address where it has results', async () => {
    renderVideoPage(`?path=${encodeURIComponent(VIDEO_PATH)}&run=job-1`);
    await lanesLoaded();
    expect(new Set(screen.getAllByTestId('lane-run').map((r) => r.textContent))).toEqual(new Set(['Pilot (old)']));
  });

  it('switches one lane to another run from its menu', async () => {
    renderVideoPage();
    await lanesLoaded();
    fireEvent.click(within(lanes('speech')[0]).getByTestId('lane-run'));
    const menu = await screen.findByTestId('run-menu');
    fireEvent.click(within(menu).getAllByText('Pilot (old)')[0]);
    await waitFor(() => expect(runOf(lanes('speech')[0])).toBe('Pilot (old)'));
    expect(runOf(lanes('speakers')[0])).toBe('Pilot (new)');
    expect(api.downloadResultFile.mock.calls.map(([u]) => String(u))).toContainEqual(
      expect.stringContaining('/jobs/job-1/results/files/speech_recognition'),
    );
  });

  it('offers a failed run disabled, with its reason', async () => {
    const failed = demoRun('job-3', { run_name: 'Broken', created_at: '2026-10-15T09:00:00Z' });
    failed.pipelines = failed.pipelines.map((p) => ({ ...p, status: 'failed', error: 'out of memory', files: [] }));
    api.getVideoRuns.mockResolvedValue(demoVideoRuns({ runs: [failed, NEW, OLD] }));
    renderVideoPage();
    await lanesLoaded();
    fireEvent.click(within(lanes('speech')[0]).getByTestId('lane-run'));
    const menu = await screen.findByTestId('run-menu');
    const broken = within(menu).getByText('Broken').closest('button')!;
    expect(broken).toBeDisabled();
    expect(broken).toHaveAttribute('title', 'out of memory');
  });

  it('compares two runs as adjacent lanes, and About lists the settings that differ', async () => {
    renderVideoPage();
    await lanesLoaded();
    fireEvent.click(within(lanes('speakers')[0]).getByTestId('lane-run'));
    fireEvent.click(within(await screen.findByTestId('run-menu')).getByText(/^Pilot \(old\)/, { selector: 'button' }));
    await waitFor(() => expect(lanes('speakers')).toHaveLength(2));
    const groups = screen.getAllByRole('group');
    const [first, second] = lanes('speakers');
    expect(groups.indexOf(second)).toBe(groups.indexOf(first) + 1);
    expect(runOf(first)).toBe('Pilot (new)');
    expect(runOf(second)).toBe('compared: Pilot (old)');

    fireEvent.mouseDown(screen.getByRole('tab', { name: 'About' }));
    const diff = await screen.findByTestId('settings-diff');
    expect(diff).toHaveTextContent('min_speakers');
    expect(within(diff).getAllByRole('row')).toHaveLength(2);
  });

  it('removes a comparison from its own menu', async () => {
    renderVideoPage();
    await lanesLoaded();
    fireEvent.click(within(lanes('scenes')[0]).getByTestId('lane-run'));
    fireEvent.click(within(await screen.findByTestId('run-menu')).getByText(/^Pilot \(old\)/, { selector: 'button' }));
    await waitFor(() => expect(lanes('scenes')).toHaveLength(2));
    fireEvent.click(within(lanes('scenes')[1]).getByTestId('lane-run'));
    fireEvent.click(await screen.findByText('Remove this comparison'));
    await waitFor(() => expect(lanes('scenes')).toHaveLength(1));
  });
});
