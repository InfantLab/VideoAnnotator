// Unit tests for useJobCancellation hook
// Tests confirmation dialog flow, optimistic updates, and error rollback

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useJobCancellation, canCancelJob } from '@/hooks/useJobCancellation';
import { apiClient } from '@/api/client';
import type { JobStatus } from '@/types/api';
import React from 'react';

type CancelJobResponse = Awaited<ReturnType<(typeof apiClient)['cancelJob']>>;
type CachedJob = {
  id: string;
  status: JobStatus;
  created_at?: string;
  updated_at?: string;
};

// Mock the API client
vi.mock('@/api/client', () => ({
  apiClient: {
    cancelJob: vi.fn(),
  },
}));

// Mock the toast hook
vi.mock('@/hooks/use-toast', () => ({
  useToast: () => ({
    toast: vi.fn(),
  }),
}));

describe('useJobCancellation', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
    vi.clearAllMocks();
  });

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );

  describe('cancelJob mutation', () => {
    it('should successfully cancel a job', async () => {
      const mockResponse = {
        job_id: 'job123',
        status: 'cancelled' as const,
        message: 'Job cancelled successfully',
        cancelled_at: '2025-10-28T10:00:00Z',
      };

      vi.mocked(apiClient.cancelJob).mockResolvedValueOnce(mockResponse);

      const { result } = renderHook(
        () => useJobCancellation('job123', 'running'),
        { wrapper }
      );

      expect(result.current.isLoading).toBe(false);

      result.current.cancelJob(undefined);

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(apiClient.cancelJob).toHaveBeenCalledWith('job123', undefined);
    });

    it('should cancel job with reason', async () => {
      const mockResponse = {
        job_id: 'job123',
        status: 'cancelling' as const,
        message: 'Cancellation in progress',
      };

      vi.mocked(apiClient.cancelJob).mockResolvedValueOnce(mockResponse);

      const { result } = renderHook(
        () => useJobCancellation('job123', 'running'),
        { wrapper }
      );

      result.current.cancelJob('User requested cancellation');

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(apiClient.cancelJob).toHaveBeenCalledWith(
        'job123',
        'User requested cancellation'
      );
    });

    it('should handle cancellation errors', async () => {
      const error = new Error('Job already cancelled');
      vi.mocked(apiClient.cancelJob).mockRejectedValueOnce(error);

      const { result } = renderHook(
        () => useJobCancellation('job123', 'running'),
        { wrapper }
      );

      result.current.cancelJob(undefined);

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });

      expect(result.current.error).toEqual(error);
    });

    it('should set loading state during cancellation', async () => {
      let resolveCancel!: (value: CancelJobResponse) => void;
      const cancelPromise = new Promise<CancelJobResponse>((resolve) => {
        resolveCancel = resolve;
      });

      vi.mocked(apiClient.cancelJob).mockReturnValueOnce(cancelPromise);

      const { result } = renderHook(
        () => useJobCancellation('job123', 'running'),
        { wrapper }
      );

      expect(result.current.isLoading).toBe(false);

      result.current.cancelJob(undefined);

      await waitFor(() => {
        expect(result.current.isLoading).toBe(true);
      });

      resolveCancel!({
        job_id: 'job123',
        status: 'cancelled',
        message: 'Done',
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });
    });
  });

  describe('optimistic updates', () => {
    it('should optimistically update job status to cancelling', async () => {
      // Use a delayed response to catch the optimistic update
      let resolveCancel!: (value: CancelJobResponse) => void;
      const cancelPromise = new Promise<CancelJobResponse>((resolve) => {
        resolveCancel = resolve;
      });

      vi.mocked(apiClient.cancelJob).mockReturnValueOnce(cancelPromise);

      // Pre-populate cache with job data
      queryClient.setQueryData(['jobs', 'job123'], {
        id: 'job123',
        status: 'running',
        updated_at: '2025-10-28T09:00:00Z',
      });

      const { result } = renderHook(
        () => useJobCancellation('job123', 'running'),
        { wrapper }
      );

      result.current.cancelJob(undefined);

      // Check optimistic update happened immediately
      await waitFor(() => {
        const jobData = queryClient.getQueryData<CachedJob>(['jobs', 'job123']);
        expect(jobData?.status).toBe('cancelling');
      });

      // Now resolve the mutation
      resolveCancel!({
        job_id: 'job123',
        status: 'cancelled' as const,
        message: 'Cancelled',
      });

      // Wait for mutation to complete
      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      // Check final status is 'cancelled'
      const finalData = queryClient.getQueryData<CachedJob>(['jobs', 'job123']);
      expect(finalData?.status).toBe('cancelled');
    });

    it('should rollback optimistic update on error', async () => {
      const error = new Error('Network error');
      vi.mocked(apiClient.cancelJob).mockRejectedValueOnce(error);

      // Pre-populate cache with job data
      const originalJob = {
        id: 'job123',
        status: 'running',
        updated_at: '2025-10-28T09:00:00Z',
      };
      queryClient.setQueryData(['jobs', 'job123'], originalJob);

      const { result } = renderHook(
        () => useJobCancellation('job123', 'running'),
        { wrapper }
      );

      result.current.cancelJob(undefined);

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });

      // Check rollback happened
      const rollbackData = queryClient.getQueryData(['jobs', 'job123']);
      expect(rollbackData).toEqual(originalJob);
    });
  });

  describe('cache invalidation', () => {
    it('should invalidate job list on successful cancellation', async () => {
      const mockResponse = {
        job_id: 'job123',
        status: 'cancelled' as const,
        message: 'Cancelled',
      };

      vi.mocked(apiClient.cancelJob).mockResolvedValueOnce(mockResponse);

      const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

      const { result } = renderHook(
        () => useJobCancellation('job123', 'running'),
        { wrapper }
      );

      result.current.cancelJob(undefined);

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(invalidateSpy).toHaveBeenCalledWith({
        queryKey: ['jobs'],
      });
    });
  });
});

describe('canCancelJob', () => {
  it('should return true for pending jobs', () => {
    expect(canCancelJob('pending')).toBe(true);
  });

  it('should return true for queued jobs', () => {
    expect(canCancelJob('queued')).toBe(true);
  });

  it('should return true for running jobs', () => {
    expect(canCancelJob('running')).toBe(true);
  });

  it('should return false for completed jobs', () => {
    expect(canCancelJob('completed')).toBe(false);
  });

  it('should return false for failed jobs', () => {
    expect(canCancelJob('failed')).toBe(false);
  });

  it('should return false for cancelled jobs', () => {
    expect(canCancelJob('cancelled')).toBe(false);
  });

  it('should return false for cancelling jobs', () => {
    expect(canCancelJob('cancelling')).toBe(false);
  });
});
