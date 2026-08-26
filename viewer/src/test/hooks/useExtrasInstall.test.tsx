// Unit tests for useExtrasInstall (specs/002-pipeline-extras-install, T013)
// Covers: trigger persists to localStorage, mount rehydrates + resumes polling,
// polling stops on terminal status, 404 drops the tracked job.

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { renderHook, waitFor, act } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import React from 'react';
import { useExtrasInstall } from '@/hooks/useExtrasInstall';
import { apiClient } from '@/api/client';
import { APIError } from '@/api/handleError';
import type { PipelineDescriptor } from '@/types/pipelines';

const STORAGE_KEY = 'videoannotator_extras_install_jobs';

vi.mock('@/api/client', () => ({
  apiClient: {
    installPipelineExtras: vi.fn(),
    getExtrasInstallJob: vi.fn()
  }
}));

// Stateful localStorage mock so writes made by the hook are observable via reads.
class MemoryStorage {
  private store = new Map<string, string>();
  getItem = vi.fn((key: string) => this.store.get(key) ?? null);
  setItem = vi.fn((key: string, value: string) => {
    this.store.set(key, value);
  });
  removeItem = vi.fn((key: string) => {
    this.store.delete(key);
  });
  clear = vi.fn(() => this.store.clear());
}

describe('useExtrasInstall', () => {
  let queryClient: QueryClient;
  let memoryStorage: MemoryStorage;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
    });
    memoryStorage = new MemoryStorage();
    vi.stubGlobal('localStorage', memoryStorage);
    vi.clearAllMocks();
  });

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );

  it('persists the job id to localStorage on a successful trigger and begins polling', async () => {
    vi.mocked(apiClient.installPipelineExtras).mockResolvedValueOnce({
      jobId: 'job-1',
      extraName: 'face',
      status: 'pending'
    });
    vi.mocked(apiClient.getExtrasInstallJob).mockResolvedValue({
      jobId: 'job-1',
      extraName: 'face',
      status: 'pending',
      createdAt: '2026-08-26T10:00:00Z',
      startedAt: null,
      finishedAt: null,
      commandOutput: null,
      restartRequired: false
    });

    const { result } = renderHook(() => useExtrasInstall(), { wrapper });

    await act(async () => {
      await result.current.install('face', ['face_analysis']);
    });

    const stored = JSON.parse(memoryStorage.getItem(STORAGE_KEY) as string);
    expect(stored.face.jobId).toBe('job-1');
    expect(stored.face.pipelineIds).toEqual(['face_analysis']);

    await waitFor(() => {
      expect(apiClient.getExtrasInstallJob).toHaveBeenCalledWith('job-1');
    });
  });

  it('rehydrates a stored job on mount and resumes polling', async () => {
    memoryStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        face: { jobId: 'job-2', startedAt: '2026-08-26T10:00:00Z', pipelineIds: ['face_analysis'] }
      })
    );
    vi.mocked(apiClient.getExtrasInstallJob).mockResolvedValue({
      jobId: 'job-2',
      extraName: 'face',
      status: 'running',
      createdAt: '2026-08-26T10:00:00Z',
      startedAt: '2026-08-26T10:00:01Z',
      finishedAt: null,
      commandOutput: null,
      restartRequired: false
    });

    const { result } = renderHook(() => useExtrasInstall(), { wrapper });

    await waitFor(() => {
      expect(apiClient.getExtrasInstallJob).toHaveBeenCalledWith('job-2');
    });
    await waitFor(() => {
      expect(result.current.jobsByExtra.face?.status).toBe('running');
    });
  });

  it('stops polling once the job reaches a terminal status', async () => {
    memoryStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        scene: { jobId: 'job-3', startedAt: '2026-08-26T10:00:00Z', pipelineIds: ['scene_detection'] }
      })
    );
    vi.mocked(apiClient.getExtrasInstallJob).mockResolvedValue({
      jobId: 'job-3',
      extraName: 'scene',
      status: 'completed',
      createdAt: '2026-08-26T10:00:00Z',
      startedAt: '2026-08-26T10:00:01Z',
      finishedAt: '2026-08-26T10:08:00Z',
      commandOutput: null,
      restartRequired: true
    });

    const { result } = renderHook(() => useExtrasInstall(), { wrapper });

    await waitFor(() => {
      expect(result.current.jobsByExtra.scene?.status).toBe('completed');
    });

    const callCountAtCompletion = vi.mocked(apiClient.getExtrasInstallJob).mock.calls.length;

    // Give React Query's refetch scheduler a real chance to fire again if it were
    // (incorrectly) still polling a terminal job.
    await new Promise((resolve) => setTimeout(resolve, 150));

    expect(vi.mocked(apiClient.getExtrasInstallJob).mock.calls.length).toBe(callCountAtCompletion);
  });

  it('drops a completed/terminal job once every pipeline it unlocked is observed available again', async () => {
    memoryStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        scene: { jobId: 'job-4', startedAt: '2026-08-26T10:00:00Z', pipelineIds: ['scene_detection'] }
      })
    );
    vi.mocked(apiClient.getExtrasInstallJob).mockResolvedValue({
      jobId: 'job-4',
      extraName: 'scene',
      status: 'completed',
      createdAt: '2026-08-26T10:00:00Z',
      startedAt: '2026-08-26T10:00:01Z',
      finishedAt: '2026-08-26T10:08:00Z',
      commandOutput: null,
      restartRequired: true
    });

    const lockedPipeline: PipelineDescriptor = { id: 'scene_detection', name: 'Scene Detection', available: false };
    const unlockedPipeline: PipelineDescriptor = { id: 'scene_detection', name: 'Scene Detection', available: true };

    const { result, rerender } = renderHook(
      (pipelines: PipelineDescriptor[]) => useExtrasInstall(pipelines),
      { wrapper, initialProps: [lockedPipeline] }
    );

    await waitFor(() => {
      expect(result.current.jobsByExtra.scene?.status).toBe('completed');
    });

    // Still locked in the catalog snapshot - tracked job must persist across a reload.
    let stored = JSON.parse(memoryStorage.getItem(STORAGE_KEY) as string);
    expect(stored.scene).toBeDefined();

    // Simulate a catalog refresh after the server restarted.
    rerender([unlockedPipeline]);

    await waitFor(() => {
      stored = JSON.parse(memoryStorage.getItem(STORAGE_KEY) as string);
      expect(stored.scene).toBeUndefined();
    });
  });

  it('drops the tracked job when the server returns 404 (stale/expired job)', async () => {
    memoryStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        audio: { jobId: 'stale-job', startedAt: '2026-08-26T10:00:00Z', pipelineIds: ['whisper'] }
      })
    );
    vi.mocked(apiClient.getExtrasInstallJob).mockRejectedValue(new APIError('Not found', 404));

    const { result } = renderHook(() => useExtrasInstall(), { wrapper });

    await waitFor(() => {
      expect(apiClient.getExtrasInstallJob).toHaveBeenCalledWith('stale-job');
    });

    await waitFor(() => {
      const stored = JSON.parse(memoryStorage.getItem(STORAGE_KEY) as string);
      expect(stored.audio).toBeUndefined();
    });

    expect(result.current.jobsByExtra.audio).toBeUndefined();
  });
});
