// Unit tests for the Pipeline Extras Install UI feature (specs/002-pipeline-extras-install)
// Covers: getPipelineCatalog({ includeUnavailable }) field mapping (T005),
// installPipelineExtras / getExtrasInstallJob (T011), and VideoAnnotator spec 011's
// activation fields, restartServer and getBootIdentity

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { APIClient } from '@/api/client';
import { apiErrorEnvelope } from '@/api/handleError';

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
        restartRequired: false,
        activation: null,
        conflictingDistributions: []
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

  describe('getCurrentUser', () => {
    it('GETs /api/v1/auth/me and maps is_admin to isAdmin', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => ({ id: 1, username: 'alice', email: 'alice@example.com', is_admin: true })
      });

      const user = await client.getCurrentUser();

      expect(user).toEqual({ id: 1, username: 'alice', email: 'alice@example.com', isAdmin: true });
      expect(mockFetch).toHaveBeenCalledWith(`${TEST_BASE_URL}/api/v1/auth/me`, expect.anything());
    });

    it('maps a non-admin identity to isAdmin: false', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => ({ id: 2, username: 'bob', email: 'bob@example.com', is_admin: false })
      });

      const user = await client.getCurrentUser();

      expect(user.isAdmin).toBe(false);
    });

    it('throws with status 401 when unauthenticated', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        statusText: 'Unauthorized',
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => ({ detail: 'Not authenticated' })
      });

      await expect(client.getCurrentUser()).rejects.toMatchObject({ status: 401 });
    });

    it('throws with status 404 on a server that predates this endpoint', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        statusText: 'Not Found',
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => ({ detail: 'Not found' })
      });

      await expect(client.getCurrentUser()).rejects.toMatchObject({ status: 404 });
    });
  });

  describe('spec 011: activation and restart', () => {
    const json = (status: number, body: unknown) => ({
      ok: status < 400,
      status,
      statusText: String(status),
      headers: new Headers({ 'content-type': 'application/json' }),
      json: async () => body
    });

    it('maps activation and conflicting distributions on a completed install', async () => {
      mockFetch.mockResolvedValueOnce(
        json(200, {
          job_id: 'job-2',
          extra_name: 'person',
          status: 'completed',
          created_at: '2026-09-23T10:00:00Z',
          started_at: '2026-09-23T10:00:01Z',
          finished_at: '2026-09-23T10:01:00Z',
          command_output: 'ok',
          restart_required: true,
          activation: 'restart_required',
          conflicting_distributions: [{ name: 'pyyaml', old_version: '6.0.2', new_version: '6.0.3' }]
        })
      );

      const job = await client.getExtrasInstallJob('job-2');

      expect(job.activation).toBe('restart_required');
      expect(job.conflictingDistributions).toEqual([
        { name: 'pyyaml', oldVersion: '6.0.2', newVersion: '6.0.3' }
      ]);
    });

    it('POSTs a restart, with force only when asked, and returns the old boot id', async () => {
      mockFetch.mockResolvedValue(json(202, { restarting: true, boot_id: 'boot-1' }));

      await expect(client.restartServer()).resolves.toEqual({ bootId: 'boot-1' });
      expect(mockFetch).toHaveBeenLastCalledWith(
        `${TEST_BASE_URL}/api/v1/system/restart`,
        expect.objectContaining({ method: 'POST' })
      );

      await client.restartServer(true);
      expect(mockFetch).toHaveBeenLastCalledWith(
        `${TEST_BASE_URL}/api/v1/system/restart?force=true`,
        expect.objectContaining({ method: 'POST' })
      );
    });

    it("keeps a refusal's code, hint and details readable", async () => {
      mockFetch.mockResolvedValueOnce(
        json(409, {
          error: {
            code: 'JOBS_RUNNING',
            message: '2 annotation job(s) are running and would be interrupted.',
            details: { job_ids: ['a', 'b'] }
          },
          detail: '2 annotation job(s) are running and would be interrupted.'
        })
      );

      const error = await client.restartServer().catch((e: unknown) => e);

      expect(error).toMatchObject({ status: 409 });
      expect(apiErrorEnvelope(error)).toEqual({
        code: 'JOBS_RUNNING',
        hint: undefined,
        details: { job_ids: ['a', 'b'] }
      });
    });

    it('reads boot identity from /health, undefined on older servers', async () => {
      mockFetch.mockResolvedValueOnce(json(200, { status: 'healthy', boot_id: 'b2', restart_mode: 'execv' }));
      await expect(client.getBootIdentity()).resolves.toEqual({ bootId: 'b2', restartMode: 'execv' });

      mockFetch.mockResolvedValueOnce(json(200, { status: 'healthy' }));
      await expect(client.getBootIdentity()).resolves.toEqual({ bootId: undefined, restartMode: undefined });
    });
  });
});
