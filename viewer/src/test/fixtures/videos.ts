// A test helper, never hot-reloaded.
/**
 * Sample responses for spec 029's routes, matching contracts/api.md, and a
 * `mockVideosApi()` that stubs the client with them (beside
 * `videoPageHarness.tsx`, which does the same for the Video page).
 */
import { vi } from 'vitest';

import { getApiClient } from '@/api/client';
import type { SavedDataset } from '@/types/datasets';
import type { PipelineCatalogResponse, PipelineDescriptor } from '@/types/pipelines';
import type { Preset } from '@/types/presets';
import type {
  Activity,
  ActivityRun,
  RunListItem,
  RunSettings,
  RunSummary,
  VideoCounts,
  VideoItem,
  VideosQuery,
  VideosResponse,
} from '@/types/videos';

export const PK = '/data/pk';
export const PROJECT_ID = 'd-peekaboo';
export const OTHER_PROJECT_ID = 'd-qs2';

const done = (run_id: string, finished_at = '2026-10-12T10:00:00Z') => ({
  state: 'done',
  run_id,
  finished_at,
  error: null,
});

export function video(name: string, over: Partial<VideoItem> = {}): VideoItem {
  const path = `${PK}/${name}`;
  return {
    path,
    paths: [path],
    name,
    folder: PK,
    available: true,
    size_bytes: 148_000_000,
    projects: [{ id: PROJECT_ID, name: 'Peekaboo study' }],
    results: { face_analysis: done('b1') },
    last_run_at: '2026-10-12T09:00:00Z',
    ...over,
  };
}

export const VIDEOS: VideoItem[] = [
  video('p01_t1.mp4', {
    results: {
      face_analysis: done('b1'),
      speech_recognition: { state: 'failed', run_id: 'b1', finished_at: null, error: 'No audio track' },
    },
    projects: [
      { id: PROJECT_ID, name: 'Peekaboo study' },
      { id: OTHER_PROJECT_ID, name: 'QS2' },
    ],
  }),
  video('p02_t1.mp4', {
    results: { face_analysis: { state: 'running', run_id: 'b2', finished_at: null, error: null } },
  }),
  video('p03_t1.mp4', { available: false, results: { face_analysis: done('b1') } }),
  video('p04_t1.mp4', { results: {}, projects: [], last_run_at: null }),
];

export const COUNTS: VideoCounts = {
  all: 212,
  not_in_project: 37,
  projects: [
    {
      id: PROJECT_ID,
      name: 'Peekaboo study',
      owner_name: 'caspar',
      count: 48,
      follows_folder: false,
      server_folder: PK,
    },
    { id: OTHER_PROJECT_ID, name: 'QS2', owner_name: 'ana', count: 1, follows_folder: false, server_folder: null },
  ],
  folders: [
    { path: PK, label: PK, count: 48, uploads: false, recursive: false },
    { path: 'uploads', label: 'Uploaded videos', count: 3, uploads: true, recursive: false },
  ],
};

export function run(over: Partial<RunSummary> = {}): RunSummary {
  return {
    id: 'b2',
    name: 'Peekaboo faces',
    total: 48,
    done: 31,
    failed: 0,
    running: 2,
    waiting: 15,
    cancelled: 0,
    started_at: '2026-10-20T09:00:00Z',
    finished_at: null,
    pipelines: ['face_analysis'],
    project: { id: PROJECT_ID, name: 'Peekaboo study' },
    active: true,
    job_ids: ['j1', 'j2'],
    ...over,
  };
}

/** A run as the Runs page lists it. */
export function listedRun(over: Partial<RunListItem> = {}): RunListItem {
  return { ...run(), videos: ['p01_t1.mp4', 'p02_t1.mp4'], failure: null, ...over };
}

export const FINISHED_RUN = run({
  id: 'b1',
  name: 'First faces',
  done: 48,
  running: 0,
  waiting: 0,
  failed: 1,
  finished_at: '2026-10-12T10:03:00Z',
  active: false,
});

export function videosResponse(query: VideosQuery = {}, over: Partial<VideosResponse> = {}): VideosResponse {
  const scope = query.project
    ? { kind: 'project', id: query.project, name: query.project === PROJECT_ID ? 'Peekaboo study' : 'QS2', path: null, recursive: false }
    : query.run
      ? { kind: 'run', id: query.run, name: query.run === 'b1' ? 'First faces' : 'Peekaboo faces', path: null, recursive: false }
      : query.folder
        ? { kind: 'folder', id: null, name: query.folder === 'uploads' ? 'Uploaded videos' : query.folder, path: query.folder, recursive: !!query.recursive }
        : query.notInProject
          ? { kind: 'not_in_project', id: null, name: null, path: null, recursive: false }
          : { kind: 'all', id: null, name: null, path: null, recursive: false };
  const videos = query.notInProject ? VIDEOS.filter((v) => v.projects.length === 0) : VIDEOS;
  const filtered = query.q ? videos.filter((v) => v.name.toLowerCase().includes(query.q!.toLowerCase())) : videos;
  return {
    scope,
    videos: filtered,
    total: filtered.length,
    counts: COUNTS,
    new_in_folder: query.project === PROJECT_ID ? [{ path: `${PK}/p49_t1.mp4`, name: 'p49_t1.mp4' }] : [],
    runs: query.project ? [run(), FINISHED_RUN] : query.run ? [query.run === 'b1' ? FINISHED_RUN : run({ id: query.run })] : [],
    subfolders: query.folder && query.folder !== 'uploads' ? [{ path: `${query.folder}/site_a`, name: 'site_a' }] : [],
    ...over,
  };
}

export const EMPTY: VideosResponse = {
  scope: { kind: 'all', id: null, name: null, path: null, recursive: false },
  videos: [],
  total: 0,
  counts: { all: 0, not_in_project: 0, projects: [], folders: [] },
  new_in_folder: [],
  runs: [],
  subfolders: [],
};

export function activityRun(over: Partial<ActivityRun> = {}): ActivityRun {
  return {
    id: 'b2',
    name: 'Peekaboo faces',
    total: 48,
    done: 31,
    failed: 0,
    running: 2,
    waiting: 15,
    cancelled: 0,
    project: { id: PROJECT_ID, name: 'Peekaboo study' },
    started_at: '2026-10-20T09:00:00Z',
    finished_at: null,
    stopped: false,
    ...over,
  };
}

export const ACTIVITY: Activity = {
  active: [activityRun()],
  recent: [activityRun({ id: 'b1', name: 'First faces', done: 48, running: 0, waiting: 0, failed: 1, finished_at: '2026-10-20T08:00:00Z' })],
};

export const RUN_SETTINGS: Record<string, RunSettings> = {
  project: {
    selected_pipelines: ['face_analysis'],
    config: { face_analysis: { min_face_size: 40 } },
    source: { kind: 'project', id: PROJECT_ID, name: 'Peekaboo study', changed_at: '2026-10-12T10:03:00Z', from: { kind: 'run', id: 'b1', name: 'First faces' } },
  },
  last_used: {
    selected_pipelines: ['scene_detection'],
    config: {},
    source: { kind: 'last_used', id: null, name: 'Yesterday', changed_at: '2026-10-19T10:00:00Z', from: null },
  },
  defaults: { selected_pipelines: [], config: {}, source: { kind: 'defaults', id: null, name: null, changed_at: null, from: null } },
};

export function dataset(over: Partial<SavedDataset> = {}): SavedDataset {
  return {
    id: PROJECT_ID,
    name: 'Peekaboo study',
    owner_user_id: 'u1',
    owner_name: 'caspar',
    video_manifest: [{ filename: 'p01_t1.mp4', size_bytes: 1, path: `${PK}/p01_t1.mp4` }],
    server_folder: PK,
    server_selection: true,
    current_settings: {
      selected_pipelines: ['face_analysis'],
      config: { face_analysis: { min_face_size: 40 } },
      source: { kind: 'run', id: 'b1', name: 'First faces' },
      updated_at: '2026-10-12T10:03:00Z',
      updated_by: 'u1',
    },
    created_at: '2026-10-01T09:00:00Z',
    ...over,
  };
}

export function workflow(over: Partial<Preset> = {}): Preset {
  return {
    id: 'w-faces40',
    name: 'Faces 40',
    owner_user_id: 'u1',
    selected_pipelines: ['face_analysis'],
    config: { face_analysis: { min_face_size: 40 } },
    created_at: '2026-10-12T11:00:00Z',
    export_choices: { table: 'csv', elan: true, methods: true },
    source: { kind: 'project', id: PROJECT_ID, name: 'Peekaboo study' },
    ...over,
  };
}

export const PIPELINES: PipelineDescriptor[] = [
  { id: 'face_analysis', name: 'DeepFace', title: 'Faces: where they are and their expression', available: true },
  { id: 'speech_recognition', name: 'Whisper large-v3-turbo', title: 'Speech: what was said', available: true },
  { id: 'scene_detection', name: 'PySceneDetect + CLIP', title: 'Scenes: cuts and settings', available: true },
  { id: 'vlm_annotation', name: 'Ask AI', title: 'Ask AI: your question about each moment', available: true },
];

export const CATALOG: PipelineCatalogResponse = {
  catalog: { pipelines: PIPELINES },
  server: { version: '1.7.0.dev0' },
  restartRequired: false,
};

/** Stubs the client's spec 029 methods; returns the spies. */
export function mockVideosApi(
  options: {
    videos?: (query: VideosQuery) => VideosResponse;
    activity?: Activity;
    runs?: RunListItem[];
    settings?: RunSettings;
    datasets?: SavedDataset[];
    presets?: Preset[];
  } = {},
) {
  vi.spyOn(console, 'log').mockImplementation(() => {});
  const client = Object.getPrototypeOf(getApiClient());
  const datasets = options.datasets ?? [dataset(), dataset({ id: OTHER_PROJECT_ID, name: 'QS2', owner_name: 'ana', server_folder: null })];
  const presets = options.presets ?? [workflow()];
  return {
    listVideos: vi.spyOn(client, 'listVideos').mockImplementation(async (query: VideosQuery = {}) =>
      (options.videos ?? videosResponse)(query),
    ),
    getActivity: vi.spyOn(client, 'getActivity').mockResolvedValue(options.activity ?? { active: [], recent: [] }),
    listRuns: vi
      .spyOn(client, 'listRuns')
      .mockResolvedValue({ runs: options.runs ?? [], total: options.runs?.length ?? 0 }),
    getRunSettings: vi.spyOn(client, 'getRunSettings').mockResolvedValue(options.settings ?? RUN_SETTINGS.defaults),
    runVideos: vi.spyOn(client, 'runVideos').mockResolvedValue({
      run_id: 'b9',
      name: 'Peekaboo faces',
      created: ['j1', 'j2', 'j3'],
      skipped: [],
      results_folder: null,
    }),
    listDatasets: vi.spyOn(client, 'listDatasets').mockResolvedValue({ datasets, total: datasets.length }),
    getDataset: vi
      .spyOn(client, 'getDataset')
      .mockImplementation(async (id: string) => datasets.find((d) => d.id === id) ?? datasets[0]),
    createDataset: vi
      .spyOn(client, 'createDataset')
      .mockImplementation(async (body: { name: string }) => dataset({ id: 'd-new', name: body.name, video_manifest: [] })),
    updateDataset: vi
      .spyOn(client, 'updateDataset')
      .mockImplementation(async (id: string, body: object) => ({ ...(datasets.find((d) => d.id === id) ?? datasets[0]), ...body })),
    deleteDataset: vi.spyOn(client, 'deleteDataset').mockResolvedValue(undefined),
    addProjectVideos: vi.spyOn(client, 'addProjectVideos').mockImplementation(async (id: string) => datasets.find((d) => d.id === id) ?? dataset({ id })),
    removeProjectVideos: vi.spyOn(client, 'removeProjectVideos').mockImplementation(async (id: string) => datasets.find((d) => d.id === id) ?? dataset({ id })),
    listPresets: vi.spyOn(client, 'listPresets').mockResolvedValue({ presets, total: presets.length }),
    createPreset: vi.spyOn(client, 'createPreset').mockImplementation(async (body: Partial<Preset>) => workflow({ ...body, id: 'w-new' })),
    updatePreset: vi.spyOn(client, 'updatePreset').mockImplementation(async (id: string, body: Partial<Preset>) => workflow({ id, ...body })),
    deletePreset: vi.spyOn(client, 'deletePreset').mockResolvedValue(undefined),
    getPipelineCatalog: vi.spyOn(client, 'getPipelineCatalog').mockResolvedValue(CATALOG),
    exportResults: vi.spyOn(client, 'exportResults').mockResolvedValue({ blob: new Blob(['run,video\n']), filename: 'Videos.csv', summary: null }),
  };
}
