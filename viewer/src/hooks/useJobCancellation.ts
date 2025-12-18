// React hook for job cancellation with confirmation and optimistic updates
// Provides user-friendly cancellation flow with error handling

import { useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/api/client';
import { useToast } from '@/hooks/use-toast';
import { QueryKeys } from '@/types/api';
import type { JobStatus } from '@/types/api';
import type { Job } from '@/types/api';
import { showErrorToast } from '@/lib/toastHelpers';
import { parseApiError } from '@/lib/errorHandling';

/**
 * Hook to cancel a job with confirmation dialog and optimistic updates
 * 
 * Features:
 * - Optimistic UI updates (immediate status change)
 * - Automatic rollback on error
 * - Toast notifications (success/error)
 * - Cache invalidation to refresh job list
 * 
 * @param jobId - ID of the job to cancel
 * @param currentStatus - Current job status (for optimistic update rollback)
 * @returns Mutation object with cancelJob function and loading state
 */
export function useJobCancellation(jobId: string, currentStatus: JobStatus) {
    const queryClient = useQueryClient();
    const { toast } = useToast();

    const mutation = useMutation({
        mutationFn: async (reason?: string) => {
            return await apiClient.cancelJob(jobId, reason);
        },

        // Optimistic update: immediately set status to 'cancelling'
        onMutate: async (reason?: string) => {
            // Cancel any outgoing refetches for this job
            await queryClient.cancelQueries({ queryKey: QueryKeys.job(jobId) });

            // Snapshot the previous value for rollback
            const previousJob = queryClient.getQueryData<Job>(QueryKeys.job(jobId));

            // Optimistically update the job status
            queryClient.setQueryData<Job>(QueryKeys.job(jobId), (old) => {
                if (!old) return old;
                return {
                    ...old,
                    status: 'cancelling' as JobStatus,
                    updated_at: new Date().toISOString(),
                };
            });

            return { previousJob };
        },

        // On success: update to 'cancelled' and show success toast
        onSuccess: (data) => {
            queryClient.setQueryData<Job>(QueryKeys.job(jobId), (old) => {
                if (!old) return old;
                return {
                    ...old,
                    status: data.status,
                    cancelled_at: data.cancelled_at || new Date().toISOString(),
                    updated_at: new Date().toISOString(),
                };
            });

            // Invalidate job list to refresh
            queryClient.invalidateQueries({ queryKey: QueryKeys.jobs });

            toast({
                title: 'Job cancelled',
                description: data.message || `Job ${jobId} has been cancelled successfully.`,
            });
        },

        // On error: rollback to previous status and show error toast WITH COPY BUTTON
        onError: (error, _variables, context) => {
            // Rollback optimistic update
            if (context?.previousJob) {
                queryClient.setQueryData(QueryKeys.job(jobId), context.previousJob);
            }

            // Parse the error and show toast with copy button
            const parsedError = parseApiError(error);
            // @ts-expect-error - toast type mismatch with ToastFunction, but runtime works fine
            showErrorToast(toast, parsedError);
        },

        // Always refetch job data after mutation settles
        onSettled: () => {
            queryClient.invalidateQueries({ queryKey: QueryKeys.job(jobId) });
        },
    });

    return {
        cancelJob: mutation.mutate,
        cancelJobAsync: mutation.mutateAsync,
        isLoading: mutation.isPending,
        isSuccess: mutation.isSuccess,
        isError: mutation.isError,
        error: mutation.error,
    };
}

/**
 * Check if a job can be cancelled based on its status
 * 
 * @param status - Current job status
 * @returns true if job can be cancelled
 */
export function canCancelJob(status: JobStatus): boolean {
    return status === 'pending' || status === 'queued' || status === 'running';
}
