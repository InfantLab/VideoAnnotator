// A test helper, never hot-reloaded.
/* eslint-disable react-refresh/only-export-components */
/**
 * The Video page against the demo-clip fixtures, through a mocked API (spec
 * 028 tests). The files are the real outputs in tests/fixtures/viewer_contract.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { vi } from 'vitest';

import { getApiClient } from '@/api/client';
import VideoPage from '@/pages/VideoPage';
import type { VideoNote, VideoNoteCreate, VideoNoteUpdate, VideoRuns } from '@/types/video';

export const FIXTURES = path.resolve(__dirname, '../../../tests/fixtures/viewer_contract');
export const VIDEO_PATH = '/data/pk/demo_clip.mp4';

const PIPELINE_FILES: Record<string, string[]> = {
  face_analysis: ['demo_clip_face_detections.json'],
  face_openface3_embedding: ['demo_clip_openface3_analysis.json', 'demo_clip_openface3_detailed.json'],
  person_tracking: ['demo_clip_person_tracking.json', 'demo_clip_person_tracks.json'],
  scene_detection: ['demo_clip_scene_detection.json'],
  speaker_diarization: ['demo_clip_speaker_diarization.rttm', 'demo_clip_speaker_diarization.rttm.provenance.json'],
  speech_recognition: ['demo_clip_speech_recognition.vtt'],
  vlm_annotation: ['demo_clip_vlm_annotation.json'],
};

const kindOf = (name: string) =>
  name.endsWith('.provenance.json') ? 'provenance' : /_(person_tracks|openface3_detailed)\.json$/.test(name) ? 'other' : 'result';

export function demoRun(jobId = 'job-1', over: Partial<VideoRuns['runs'][number]> = {}): VideoRuns['runs'][number] {
  return {
    job_id: jobId,
    status: 'completed',
    created_at: '2026-10-12T09:14:00Z',
    completed_at: '2026-10-12T09:31:00Z',
    batch_id: null,
    dataset_id: null,
    run_name: null,
    progress: 100,
    pipelines: Object.entries(PIPELINE_FILES).map(([name, files]) => ({
      name,
      status: 'completed',
      error: null,
      files: files.map((file) => ({
        name: file,
        kind: kindOf(file),
        download_url: `/api/v1/jobs/${jobId}/results/files/${name}?name=${file}`,
      })),
    })),
    ...over,
  };
}

export function demoVideoRuns(over: Partial<VideoRuns> = {}): VideoRuns {
  return {
    path: VIDEO_PATH,
    name: 'demo_clip.mp4',
    available: true,
    unavailable_reason: null,
    duration_sec: 20,
    sha256: null,
    paths: [VIDEO_PATH],
    runs: [demoRun()],
    projects: [],
    ...over,
  };
}

/** Mocks the API the page uses; returns the spies. */
export function mockVideoApi(runs: VideoRuns = demoVideoRuns(), notes: VideoNote[] = []) {
  vi.spyOn(console, 'log').mockImplementation(() => {});
  // `apiClient` is a Proxy onto this client; its methods live on the prototype.
  const client = Object.getPrototypeOf(getApiClient());
  return {
    getVideoRuns: vi.spyOn(client, 'getVideoRuns').mockResolvedValue(runs),
    streamToken: vi.spyOn(client, 'streamToken').mockResolvedValue({
      token: 'tok',
      expires_at: new Date(Date.now() + 3600_000).toISOString(),
      path: runs.path,
      name: runs.name,
      duration_sec: runs.duration_sec,
    }),
    notes,
    listNotes: vi.spyOn(client, 'listNotes').mockImplementation(async () => [...notes]),
    createNote: vi.spyOn(client, 'createNote').mockImplementation(async (body: VideoNoteCreate) => {
      const end = body.end_sec ?? null;
      const created: VideoNote = {
        id: `note-${notes.length + 1}`,
        video_path: body.path,
        start_sec: end !== null ? Math.min(body.start_sec, end) : body.start_sec,
        end_sec: end !== null ? Math.max(body.start_sec, end) : null,
        text: body.text.trim(),
        author_id: 'u1',
        author_name: 'alice',
        created_at: new Date(Date.UTC(2026, 9, 19, 10, notes.length)).toISOString(),
        updated_at: null,
        updated_by: null,
      };
      notes.push(created);
      return created;
    }),
    updateNote: vi.spyOn(client, 'updateNote').mockImplementation(async (id: string, body: VideoNoteUpdate) => {
      const found = notes.find((n) => n.id === id)!;
      Object.assign(found, body.text !== undefined && body.text !== null ? { text: body.text } : {});
      return found;
    }),
    deleteNote: vi.spyOn(client, 'deleteNote').mockImplementation(async (id: string) => {
      notes.splice(notes.findIndex((n) => n.id === id), 1);
    }),
    downloadResultFile: vi
      .spyOn(client, 'downloadResultFile')
      .mockImplementation(async (_url: string, name: string) => new File([readFileSync(path.join(FIXTURES, name), 'utf8')], name)),
  };
}

export function renderVideoPage(search = `?path=${encodeURIComponent(VIDEO_PATH)}`) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[`/video${search}`]}>
        <Routes>
          <Route path="/video" element={<VideoPage />} />
          <Route path="/ask" element={<div>Ask AI page</div>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

/** Waits until the lanes are drawn from the fetched files (the Notes lane is there from the start). */
export async function lanesLoaded() {
  await screen.findAllByTestId('row-label', {}, { timeout: 5000 });
}

/** Every element 1000 px wide from x = 0, so clicks map to times. */
export function wideLayout() {
  return vi.spyOn(Element.prototype, 'getBoundingClientRect').mockReturnValue({
    x: 0,
    y: 0,
    left: 0,
    top: 0,
    right: 1000,
    bottom: 200,
    width: 1000,
    height: 200,
    toJSON: () => ({}),
  } as DOMRect);
}
