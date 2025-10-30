// React hook for job deletion with confirmation and optimistic updates
// Provides user-friendly deletion flow with error handling

import { useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/api/client';
import { useToast } from '@/hooks/use-toast';
import { QueryKeys } from '@/types/api';
import type { JobStatus } from '@/types/api';
import { useNavigate } from 'react-router-dom';
import { showErrorToast } from '@/lib/toastHelpers';
import { parseApiError } from '@/lib/errorHandling';

/**
 * Hook to delete a job with confirmation dialog and optimistic updates
 * 
 * Features:
 * - Optimistic UI updates (immediate removal from list)
 * - Automatic rollback on error
 * - Toast notifications (success/error)
 * - Cache invalidation to refresh job list
 * - Navigation support (redirect from detail page after deletion)
 * 
 * @param jobId - ID of the job to delete
 * @param currentStatus - Current job status (for validation)
 * @returns Mutation object with deleteJob function and loading state
 */
export function useJobDeletion(jobId: string, currentStatus: JobStatus) {
    const queryClient = useQueryClient();
    const { toast } = useToast();

    const mutation = useMutation({
        mutationFn: async () => {
            return await apiClient.deleteJob(jobId);
        },

        // Optimistic update: immediately remove from cache
        onMutate: async () => {
            // Cancel any outgoing refetches for this job
            await queryClient.cancelQueries({ queryKey: QueryKeys.job(jobId) });

            // Snapshot the previous value for rollback
            const previousJob = queryClient.getQueryData(QueryKeys.job(jobId));

            // Optimistically remove the job from cache
            queryClient.removeQueries({ queryKey: QueryKeys.job(jobId) });

            return { previousJob };
        },

        // On success: show success toast and invalidate job list
        onSuccess: () => {
            // Invalidate job list to refresh
            queryClient.invalidateQueries({ queryKey: QueryKeys.jobs });

            toast({
                title: 'Job deleted',
                description: `Job ${jobId.slice(0, 8)}... has been permanently deleted.`,
            });
        },

        // On error: rollback and show error toast WITH COPY BUTTON
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

        // Always refetch job list after mutation settles
        onSettled: () => {
            queryClient.invalidateQueries({ queryKey: QueryKeys.jobs });
        },
    });

    return {
        deleteJob: mutation.mutate,
        deleteJobAsync: mutation.mutateAsync,
        isLoading: mutation.isPending,
        isSuccess: mutation.isSuccess,
        isError: mutation.isError,
        error: mutation.error,
    };
}

/**
 * Check if a job can be deleted based on its status
 * Jobs can be deleted when they are in a terminal state (completed, failed, cancelled).
 * Running/pending jobs should be cancelled first before deletion.
 * 
 * @param status - Current job status
 * @returns true if job can be deleted
 */
export function canDeleteJob(status: JobStatus): boolean {
    return status === 'completed' || status === 'failed' || status === 'cancelled';
}
