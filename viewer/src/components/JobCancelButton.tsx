// Job Cancel Button with confirmation dialog
// Displays cancel button conditionally based on job status with loading states

import { useState } from 'react';
import { XCircle, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { useJobCancellation, canCancelJob } from '@/hooks/useJobCancellation';
import type { JobStatus } from '@/types/api';

interface JobCancelButtonProps {
  jobId: string;
  jobStatus: JobStatus;
  variant?: 'default' | 'destructive' | 'outline' | 'secondary' | 'ghost' | 'link';
  size?: 'default' | 'sm' | 'lg' | 'icon';
  className?: string;
}

/**
 * Button component to cancel a job with confirmation dialog
 * 
 * Features:
 * - Only renders for cancellable jobs (pending, queued, running)
 * - Shows confirmation dialog before cancelling
 * - Displays loading state during cancellation
 * - Automatically disabled for non-cancellable statuses
 * 
 * @param jobId - ID of the job to cancel
 * @param jobStatus - Current status of the job
 * @param variant - Button variant (default: destructive)
 * @param size - Button size (default: default)
 * @param className - Additional CSS classes
 */
export function JobCancelButton({
  jobId,
  jobStatus,
  variant = 'destructive',
  size = 'default',
  className,
}: JobCancelButtonProps) {
  const [showDialog, setShowDialog] = useState(false);
  const { cancelJob, isLoading } = useJobCancellation(jobId, jobStatus);

  // Don't render button if job cannot be cancelled
  if (!canCancelJob(jobStatus)) {
    return null;
  }

  const handleCancel = () => {
    cancelJob(undefined, {
      onSuccess: () => {
        setShowDialog(false);
      },
    });
  };

  return (
    <>
      <Button
        variant={variant}
        size={size}
        className={className}
        onClick={() => setShowDialog(true)}
        disabled={isLoading || jobStatus === 'cancelling'}
      >
        {isLoading || jobStatus === 'cancelling' ? (
          <>
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            Cancelling...
          </>
        ) : (
          <>
            <XCircle className="mr-2 h-4 w-4" />
            Cancel Job
          </>
        )}
      </Button>

      <AlertDialog open={showDialog} onOpenChange={setShowDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Cancel Job?</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to cancel this job? This action cannot be undone.
              The job will stop processing and any partial results may be discarded.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={isLoading}>
              No, keep running
            </AlertDialogCancel>
            <AlertDialogAction
              onClick={handleCancel}
              disabled={isLoading}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Cancelling...
                </>
              ) : (
                'Yes, cancel job'
              )}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
