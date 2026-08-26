// Unit tests for the Pipeline Extras Install UI feature (specs/002-pipeline-extras-install)
// Covers: getPipelineCatalog({ includeUnavailable }) field mapping (T005),
// installPipelineExtras / getExtrasInstallJob (T011)

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { APIClient } from '@/api/client';

const TEST_BASE_URL = 'http://127.0.0.1:18011';
const TEST_TOKEN = 'va_test12345678';

describe('APIClient pipeline extras install', () => {
  let client: APIClient;
  let mockFetch: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.spyOn(globalThis.localStorage, 'getItem').mockImplementation((key: string) => {
      if (key === 'videoannotator_api_url') return TEST_BASE_URL;
      if (key === 'videoannotator_api_token') return TEST_TOKEN;
      return null;
    });

    mockFetch = vi.fn();
    global.fetch = mockFetch as unknown as typeof fetch;

    client = new APIClient(TEST_BASE_URL, TEST_TOKEN);
  });

  describe('getPipelineCatalog({ includeUnavailable: true })', () => {
    it('maps available/install_hint per pipeline and top-level restart_required', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => ({
          pipelines: [
            { slug: 'stub', name: 'Stub Pipeline', available: true, install_hint: null },
            {
              slug: 'face',
              name: 'Face Analysis',
              available: false,
              install_hint: 'pip install videoannotator[face]'
            }
          ],
          restart_required: true
        })
      });

      const response = await client.getPipelineCatalog({ includeUnavailable: true });

      expect(mockFetch).toHaveBeenCalledWith(
        `${TEST_BASE_URL}/api/v1/pipelines?include_unavailable=true`,
        expect.anything()
      );

      expect(response.restartRequired).toBe(true);

      const stub = response.catalog.pipelines.find((p) => p.id === 'stub');
      const face = response.catalog.pipelines.find((p) => p.id === 'face');

      expect(stub?.available).toBe(true);
      expect(stub?.installHint).toBeUndefined();

      expect(face?.available).toBe(false);
      expect(face?.installHint).toBe('pip install videoannotator[face]');
    });

    it('defaults available and restartRequired for a pre-v1.5.0 server response missing the new fields', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => ({
          pipelines: [{ slug: 'stub', name: 'Stub Pipeline' }]
          // no restart_required field at all - legacy server shape
        })
      });

      const response = await client.getPipelineCatalog({ includeUnavailable: true });

      expect(response.restartRequired).toBe(false);
      expect(response.catalog.pipelines[0].available).toBeUndefined();
    });

    it('omits the include_unavailable query param when not requested', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => ({ pipelines: [] })
      });

      await client.getPipelineCatalog();

      expect(mockFetch).toHaveBeenCalledWith(`${TEST_BASE_URL}/api/v1/pipelines`, expect.anything());
    });
  });

  describe('installPipelineExtras', () => {
    it('POSTs to the extras install endpoint and returns the trigger response', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 202,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => ({ job_id: 'job-1', extra_name: 'face', status: 'pending' })
      });

      const result = await client.installPipelineExtras('face');

      expect(result).toEqual({ jobId: 'job-1', extraName: 'face', status: 'pending' });
      expect(mockFetch).toHaveBeenCalledWith(
        `${TEST_BASE_URL}/api/v1/pipelines/extras/face/install`,
        expect.objectContaining({ method: 'POST' })
      );
    });

    it('throws with status 401 when unauthenticated', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        statusText: 'Unauthorized',
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => ({ detail: 'Not authenticated' })
      });

      await expect(client.installPipelineExtras('face')).rejects.toMatchObject({ status: 401 });
    });

    it('throws with status 403 when authenticated but not admin', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 403,
        statusText: 'Forbidden',
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => ({ detail: 'Admin privileges required' })
      });

      await expect(client.installPipelineExtras('face')).rejects.toMatchObject({ status: 403 });
    });

    it('throws with status 422 for an unrecognized extras name', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 422,
        statusText: 'Unprocessable Entity',
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => ({ detail: 'Unknown extras group: bogus. Valid: face, audio, scene, person' })
      });

      await expect(client.installPipelineExtras('bogus')).rejects.toMatchObject({ status: 422 });
    });
  });

  describe('getExtrasInstallJob', () => {
    it('GETs the job status endpoint and maps snake_case fields to camelCase', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => ({
          job_id: 'job-1',
          extra_name: 'face',
          status: 'running',
          created_at: '2026-08-26T10:00:00Z',
          started_at: '2026-08-26T10:00:01Z',
          finished_at: null,
          command_output: null,
          restart_required: false
        })
      });

      const job = await client.getExtrasInstallJob('job-1');

      expect(job).toEqual({
        jobId: 'job-1',
        extraName: 'face',
        status: 'running',
        createdAt: '2026-08-26T10:00:00Z',
        startedAt: '2026-08-26T10:00:01Z',
        finishedAt: null,
        commandOutput: null,
        restartRequired: false
      });
      expect(mockFetch).toHaveBeenCalledWith(
        `${TEST_BASE_URL}/api/v1/pipelines/extras/install-jobs/job-1`,
        expect.anything()
      );
    });

    it('maps a failed job with command_output', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => ({
          job_id: 'job-2',
          extra_name: 'scene',
          status: 'failed',
          created_at: '2026-08-26T10:00:00Z',
          started_at: '2026-08-26T10:00:01Z',
          finished_at: '2026-08-26T10:05:00Z',
          command_output: 'ERROR: CUDA wheel download failed',
          restart_required: false
        })
      });

      const job = await client.getExtrasInstallJob('job-2');

      expect(job.status).toBe('failed');
      expect(job.commandOutput).toBe('ERROR: CUDA wheel download failed');
    });

    it('throws with status 404 when the job is unknown/expired', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        statusText: 'Not Found',
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => ({ detail: 'Job not found' })
      });

      await expect(client.getExtrasInstallJob('stale-job')).rejects.toMatchObject({ status: 404 });
    });
  });
});
