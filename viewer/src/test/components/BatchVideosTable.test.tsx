// The batch page's video list: what a run shares is said once, rows can be
// narrowed to the ones that need attention, and a large run stays searchable.

import { describe, it, expect, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import React from 'react';
import { BatchVideosTable } from '@/components/BatchVideosTable';
import { filterJobs, outcomeOf, sharedPipelines } from '@/lib/jobRow';
import type { JobResponse } from '@/api/client';

vi.mock('@/hooks/use-toast', () => ({
  useToast: () => ({ toast: vi.fn() }),
}));

const PIPELINES = ['face_analysis', 'person_tracking', 'scene_detection'];

function makeJob(i: number, overrides: Partial<JobResponse> = {}): JobResponse {
  return {
    id: `job-${i}`,
    status: 'completed',
    video_filename: `clip_${String(i).padStart(2, '0')}.mp4`,
    selected_pipelines: PIPELINES,
    error_message: null,
    created_at: '2026-10-06T10:00:00Z',
    ...overrides,
  } as JobResponse;
}

function renderTable(jobs: JobResponse[], active = false) {
  const client = new QueryClient();
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <BatchVideosTable jobs={jobs} active={active} />
      </MemoryRouter>
    </QueryClientProvider>
  );
}

const bodyRows = () => screen.getAllByRole('row').slice(1);

describe('jobRow helpers', () => {
  it('reports pipelines shared by every video, in any order', () => {
    const jobs = [makeJob(1), makeJob(2, { selected_pipelines: [...PIPELINES].reverse() })];
    expect(sharedPipelines(jobs)).toEqual(PIPELINES);
  });

  it('reports null when a video ran different pipelines', () => {
    expect(sharedPipelines([makeJob(1), makeJob(2, { selected_pipelines: ['face_analysis'] })])).toBeNull();
  });

  it('separates completed-with-errors from done', () => {
    expect(outcomeOf(makeJob(1))).toBe('done');
    expect(outcomeOf(makeJob(1, { error_message: 'scene_detection failed' }))).toBe('errors');
    expect(outcomeOf(makeJob(1, { status: 'pending' }))).toBe('queued');
  });

  it('filters by outcome and filename', () => {
    const jobs = [makeJob(1), makeJob(2, { status: 'failed' }), makeJob(12, { status: 'failed' })];
    expect(filterJobs(jobs, 'failed', '').map((j) => j.id)).toEqual(['job-2', 'job-12']);
    expect(filterJobs(jobs, 'all', 'CLIP_1').map((j) => j.id)).toEqual(['job-12']);
  });
});

describe('BatchVideosTable', () => {
  it('names the shared pipelines once, not on every row', () => {
    renderTable([makeJob(1), makeJob(2), makeJob(3)]);

    expect(screen.getAllByText('face_analysis')).toHaveLength(1);
    expect(screen.queryByRole('columnheader', { name: 'Pipelines' })).not.toBeInTheDocument();
  });

  it('shows pipelines per row when they differ', () => {
    renderTable([makeJob(1), makeJob(2, { selected_pipelines: ['face_analysis'] })]);

    expect(screen.getByRole('columnheader', { name: 'Pipelines' })).toBeInTheDocument();
  });

  it('has no delete on rows; one action per finished video', () => {
    renderTable([makeJob(1)]);

    expect(screen.queryByRole('button', { name: /delete/i })).not.toBeInTheDocument();
    expect(within(bodyRows()[0]).getAllByRole('button')).toHaveLength(1);
  });

  it('narrows to failed videos', async () => {
    const jobs = [
      ...Array.from({ length: 20 }, (_, i) => makeJob(i)),
      makeJob(20, { status: 'failed', error_message: 'out of memory' }),
    ];
    renderTable(jobs);
    expect(bodyRows()).toHaveLength(21);

    await userEvent.click(screen.getByRole('button', { name: '1 failed' }));

    expect(bodyRows()).toHaveLength(1);
    expect(screen.getByText('clip_20.mp4')).toBeInTheDocument();
  });

  it('offers search only for larger runs', async () => {
    renderTable(Array.from({ length: 4 }, (_, i) => makeJob(i)));
    expect(screen.queryByLabelText('Find a video')).not.toBeInTheDocument();
  });

  it('finds a video by name in a large run', async () => {
    renderTable(Array.from({ length: 25 }, (_, i) => makeJob(i)));

    await userEvent.type(screen.getByLabelText('Find a video'), 'clip_17');

    expect(bodyRows()).toHaveLength(1);
  });

  it('shows progress only while the run is active', () => {
    const { unmount } = renderTable([makeJob(1)]);
    expect(screen.queryByRole('columnheader', { name: 'Progress' })).not.toBeInTheDocument();
    unmount();

    renderTable([makeJob(1, { status: 'running' })], true);
    expect(screen.getByRole('columnheader', { name: 'Progress' })).toBeInTheDocument();
  });
});
