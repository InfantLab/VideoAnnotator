/** The client's spec 029 methods: the requests they make. */
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { APIClient } from '@/api/client';

const BASE = 'http://127.0.0.1:18011';

function json(body: unknown, headers: Record<string, string> = {}) {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json', ...headers },
  });
}

describe('videos client', () => {
  let client: APIClient;
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.spyOn(globalThis.localStorage, 'getItem').mockImplementation((key: string) =>
      key === 'videoannotator_api_url' ? BASE : key === 'videoannotator_api_token' ? 'va_test12345678' : null,
    );
    fetchMock = vi.fn(async () => json({}));
    global.fetch = fetchMock as unknown as typeof fetch;
    client = new APIClient(BASE, 'va_test12345678');
  });

  const call = (n = 0) => fetchMock.mock.calls[n] as [string, RequestInit];

  it('lists videos for a scope', async () => {
    await client.listVideos({ project: 'd1', q: 'p0' });
    await client.listVideos({ folder: '/data/pk', recursive: true });
    await client.listVideos({ notInProject: true });
    await client.listVideos({ run: 'b1' });

    expect(call(0)[0]).toBe(`${BASE}/api/v1/videos?project=d1&q=p0&limit=2000`);
    expect(call(1)[0]).toBe(`${BASE}/api/v1/videos?folder=%2Fdata%2Fpk&recursive=true&limit=2000`);
    expect(call(2)[0]).toBe(`${BASE}/api/v1/videos?not_in_project=true&limit=2000`);
    expect(call(3)[0]).toBe(`${BASE}/api/v1/videos?run=b1&limit=2000`);
  });

  it('starts a run on videos', async () => {
    await client.runVideos({ paths: ['/a.mp4'], selected_pipelines: ['face_analysis'], project_id: 'd1' });

    const [url, init] = call();
    expect(url).toBe(`${BASE}/api/v1/videos/run`);
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body as string)).toEqual({ paths: ['/a.mp4'], selected_pipelines: ['face_analysis'], project_id: 'd1' });
  });

  it('asks for run settings and activity', async () => {
    await client.getRunSettings();
    await client.getRunSettings({ project: 'd1', workflow: 'w1' });
    await client.getActivity();

    expect(call(0)[0]).toBe(`${BASE}/api/v1/videos/run-settings`);
    expect(call(1)[0]).toBe(`${BASE}/api/v1/videos/run-settings?project=d1&workflow=w1`);
    expect(call(2)[0]).toBe(`${BASE}/api/v1/videos/activity`);
  });

  it('adds and removes project videos', async () => {
    await client.addProjectVideos('d1', ['/a.mp4']);
    await client.removeProjectVideos('d1', ['/a.mp4']);

    expect(call(0)[0]).toBe(`${BASE}/api/v1/datasets/d1/videos`);
    expect(call(0)[1].method).toBe('POST');
    expect(call(1)[1].method).toBe('DELETE');
    expect(JSON.parse(call(1)[1].body as string)).toEqual({ paths: ['/a.mp4'] });
  });

  it('exports a set of videos by POST, and a run as every result', async () => {
    fetchMock.mockImplementation(async () =>
      new Response('run,video\n', {
        status: 200,
        headers: { 'Content-Disposition': 'attachment; filename="Videos.csv"', 'X-VideoAnnotator-Export-Summary': '{"rows":1}' },
      }),
    );

    const download = await client.exportResults({ kind: 'videos', paths: ['/a.mp4'] }, 'table', 'parquet');
    await client.exportResults({ kind: 'project', id: 'd1', results: 'all' }, 'elan');
    await client.exportResults({ kind: 'run', id: 'b1' }, 'table');
    await client.exportResults({ kind: 'batch', id: 'b1' }, 'table');

    expect(download.filename).toBe('Videos.csv');
    expect(call(0)[0]).toBe(`${BASE}/api/v1/videos/export/table`);
    expect(JSON.parse(call(0)[1].body as string)).toEqual({ paths: ['/a.mp4'], results: 'latest', format: 'parquet' });
    expect(JSON.parse(call(1)[1].body as string)).toEqual({ project_id: 'd1', results: 'all' });
    expect(JSON.parse(call(2)[1].body as string)).toEqual({ run_id: 'b1', results: 'all', format: 'csv' });
    expect(call(3)[0]).toBe(`${BASE}/api/v1/batches/b1/export/table?format=csv`);
  });

  it('asks for methods for a set', async () => {
    fetchMock.mockImplementation(async () => json({ prose: 'x' }));

    await client.getMethods({ kind: 'project', id: 'd1' });

    expect(call()[0]).toBe(`${BASE}/api/v1/videos/export/methods`);
    expect(JSON.parse(call()[1].body as string)).toEqual({ project_id: 'd1', results: 'latest', format: 'json' });
  });

  it('changes and removes a workflow', async () => {
    await client.updatePreset('w1', { name: 'Faces 50' });
    await client.deletePreset('w1');

    expect(call(0)[1].method).toBe('PUT');
    expect(call(1)[0]).toBe(`${BASE}/api/v1/presets/w1`);
    expect(call(1)[1].method).toBe('DELETE');
  });
});
