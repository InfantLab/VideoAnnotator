import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import Compare from '@/pages/Compare';
import { apiClient } from '@/api/client';

vi.mock('@/api/client', () => ({
  apiClient: { getJob: vi.fn(), getResultFileText: vi.fn(), getJobVideoUrl: vi.fn() },
}));

const fixture = JSON.parse(
  readFileSync(path.resolve(__dirname, '../../../../tests/fixtures/viewer_contract/demo_clip_vlm_annotation.json'), 'utf8'),
);

const runFile = (labels: string[], sha: string) =>
  JSON.stringify({
    ...fixture,
    annotations: fixture.annotations.slice(0, labels.length).map((a: Record<string, unknown>, i: number) => ({
      ...a,
      timestamp_sec: i * 5,
      label: labels[i],
    })),
    provenance: {
      schema_version: 1,
      pipeline: { name: 'vlm_annotation' },
      videoannotator_version: '1.6.0',
      created_at: '2026-10-04T00:00:00+00:00',
      input: { name: 'clip.mp4', sha256: sha },
    },
  });

const job = (id: string) => ({ id, status: 'completed', video_filename: 'clip.mp4', video_size_bytes: 10, selected_pipelines: ['vlm_annotation'] });

const renderCompare = () =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <MemoryRouter initialEntries={['/compare?a=job-a&b=job-b']}>
        <Routes>
          <Route path="/compare" element={<Compare />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );

describe('Compare', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(apiClient.getJob).mockImplementation(async (id: string) => job(id) as never);
    vi.mocked(apiClient.getJobVideoUrl).mockResolvedValue('blob:video');
  });

  it('shows where two runs of the same video disagree, with a summary', async () => {
    vi.mocked(apiClient.getResultFileText).mockImplementation(async (id: string) =>
      id === 'job-a' ? runFile(['TOUCH', 'TOUCH', 'NO_TOUCH'], 'same') : runFile(['TOUCH', 'NO_TOUCH', 'NO_TOUCH'], 'same'),
    );
    renderCompare();
    expect(await screen.findByText(/3 moments compared/)).toBeInTheDocument();
    expect(screen.getByText(/67% agreement/)).toBeInTheDocument();
    expect(screen.getByText('Where they disagree (1)')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /5s: A TOUCH · B NO_TOUCH/ }));
    expect(screen.getByText('At 5s')).toBeInTheDocument();
  });

  it('refuses runs of different videos', async () => {
    vi.mocked(apiClient.getResultFileText).mockImplementation(async (id: string) =>
      runFile(['TOUCH'], id === 'job-a' ? 'one' : 'two'),
    );
    renderCompare();
    expect(await screen.findByText(/different videos/)).toBeInTheDocument();
  });
});
