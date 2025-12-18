// Integration test for job cancellation flow
// Tests the complete flow from user interaction to API call

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { JobCancelButton } from '@/components/JobCancelButton';
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
const mockToast = vi.fn();
vi.mock('@/hooks/use-toast', () => ({
  useToast: () => ({
    toast: mockToast,
  }),
}));

describe('Job Cancellation Integration', () => {
  let queryClient: QueryClient;
  let user: ReturnType<typeof userEvent.setup>;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
    user = userEvent.setup();
    vi.clearAllMocks();
  });

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );

  it('should complete full cancellation flow successfully', async () => {
    const mockResponse = {
      job_id: 'job123',
      status: 'cancelling' as const,
      message: 'Job cancellation initiated',
    };

    vi.mocked(apiClient.cancelJob).mockResolvedValueOnce(mockResponse);

    // Pre-populate cache with running job
    queryClient.setQueryData(['jobs', 'job123'], {
      id: 'job123',
      status: 'running',
      created_at: '2025-10-28T10:00:00Z',
      updated_at: '2025-10-28T10:00:00Z',
    });

    render(<JobCancelButton jobId="job123" jobStatus="running" />, { wrapper });

    // 1. Click cancel button
    const cancelButton = screen.getByRole('button', { name: /cancel/i });
    expect(cancelButton).toBeInTheDocument();
    await user.click(cancelButton);

    // 2. Dialog appears
    await waitFor(() => {
      expect(screen.getByRole('alertdialog')).toBeInTheDocument();
    });
    expect(screen.getByRole('heading', { name: /cancel job/i })).toBeInTheDocument();

    // 3. Confirm cancellation
    const confirmButton = screen.getByRole('button', { name: /yes.*cancel/i });
    await user.click(confirmButton);

    // 4. API called
    await waitFor(() => {
      expect(apiClient.cancelJob).toHaveBeenCalledWith('job123', undefined);
      expect(apiClient.cancelJob).toHaveBeenCalledTimes(1);
    });

    // 5. Optimistic update applied
    const jobData = queryClient.getQueryData<CachedJob>(['jobs', 'job123']);
    expect(jobData?.status).toBe('cancelling');

    // 6. Toast notification shown
    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith({
        title: 'Job cancelled',
        description: 'Job cancellation initiated',
      });
    });
  });

  it('should handle cancellation error with rollback', async () => {
    const error = new Error('Job already completed');
    vi.mocked(apiClient.cancelJob).mockRejectedValueOnce(error);

    // Pre-populate cache with running job
    const originalJob = {
      id: 'job123',
      status: 'running',
      created_at: '2025-10-28T10:00:00Z',
      updated_at: '2025-10-28T10:00:00Z',
    };
    queryClient.setQueryData(['jobs', 'job123'], originalJob);

    render(<JobCancelButton jobId="job123" jobStatus="running" />, { wrapper });

    // 1. Click cancel button
    const cancelButton = screen.getByRole('button', { name: /cancel/i });
    await user.click(cancelButton);

    // 2. Confirm cancellation
    const confirmButton = screen.getByRole('button', { name: /yes.*cancel/i });
    await user.click(confirmButton);

    // 3. API called and failed
    await waitFor(() => {
      expect(apiClient.cancelJob).toHaveBeenCalledWith('job123', undefined);
    });

    // 4. Rollback applied (job back to original state)
    await waitFor(() => {
      const jobData = queryClient.getQueryData(['jobs', 'job123']);
      expect(jobData).toEqual(originalJob);
    });

    // 5. Error toast shown
    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith({
        title: 'Cancellation failed',
        description: 'Job already completed',
        variant: 'destructive',
      });
    });

    // 6. Button re-enabled
    await waitFor(() => {
      const cancelBtn = screen.getByRole('button', { name: /cancel/i });
      expect(cancelBtn).not.toBeDisabled();
    });
  });

  it('should support cancellation with reason', async () => {
    const mockResponse = {
      job_id: 'job123',
      status: 'cancelled' as const,
      message: 'Cancelled by user',
      cancelled_at: '2025-10-28T10:30:00Z',
    };

    vi.mocked(apiClient.cancelJob).mockResolvedValueOnce(mockResponse);

    render(<JobCancelButton jobId="job123" jobStatus="running" />, { wrapper });

    const cancelButton = screen.getByRole('button', { name: /cancel/i });
    await user.click(cancelButton);

    const confirmButton = screen.getByRole('button', { name: /yes.*cancel/i });
    await user.click(confirmButton);

    await waitFor(() => {
      expect(apiClient.cancelJob).toHaveBeenCalledWith('job123', undefined);
    });
  });

  it('should not render for non-cancellable job states', () => {
    const nonCancellableStates: JobStatus[] = ['completed', 'failed', 'cancelled', 'cancelling'];

    nonCancellableStates.forEach((status) => {
      const { container } = render(
        <JobCancelButton
          jobId="job123"
          jobStatus={status}
        />,
        { wrapper }
      );

      expect(container.firstChild).toBeNull();
    });
  });

  it('should show loading state during cancellation', async () => {
    let resolveCancel!: (value: CancelJobResponse) => void;
    const cancelPromise = new Promise<CancelJobResponse>((resolve) => {
      resolveCancel = resolve;
    });

    vi.mocked(apiClient.cancelJob).mockReturnValueOnce(cancelPromise);

    render(<JobCancelButton jobId="job123" jobStatus="running" />, { wrapper });

    const cancelButton = screen.getByRole('button', { name: /cancel/i });
    await user.click(cancelButton);

    const confirmButton = screen.getByRole('button', { name: /yes.*cancel/i });
    await user.click(confirmButton);

    // Button shows loading state
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /cancelling/i })).toBeDisabled();
    });

    // Resolve the promise
    resolveCancel!({
      job_id: 'job123',
      status: 'cancelled',
      message: 'Cancelled',
    });

    // Button returns to normal state (or disappears if cancelled)
    await waitFor(() => {
      expect(screen.queryByRole('button', { name: /cancelling/i })).not.toBeInTheDocument();
    });
  });
});
