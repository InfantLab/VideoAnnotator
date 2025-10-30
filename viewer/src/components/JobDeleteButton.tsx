// Job Delete Button with confirmation dialog
// Displays delete button for finished jobs with loading states

import { useState } from 'react';
import { Trash2, Loader2 } from 'lucide-react';
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
import { useJobDeletion, canDeleteJob } from '@/hooks/useJobDeletion';
import type { JobStatus } from '@/types/api';

interface JobDeleteButtonProps {
    jobId: string;
    jobStatus: JobStatus;
    variant?: 'default' | 'destructive' | 'outline' | 'secondary' | 'ghost' | 'link';
    size?: 'default' | 'sm' | 'lg' | 'icon';
    className?: string;
    onDeleted?: () => void;
}

/**
 * Button component to delete a job with confirmation dialog
 * 
 * Features:
 * - Only renders for deletable jobs (completed, failed, cancelled)
 * - Shows confirmation dialog before deleting
 * - Displays loading state during deletion
 * - Warns user that deletion is permanent
 * 
 * @param jobId - ID of the job to delete
 * @param jobStatus - Current status of the job
 * @param variant - Button variant (default: outline)
 * @param size - Button size (default: default)
 * @param className - Additional CSS classes
 * @param onDeleted - Optional callback when job is successfully deleted
 */
export function JobDeleteButton({
    jobId,
    jobStatus,
    variant = 'outline',
    size = 'default',
    className,
    onDeleted,
}: JobDeleteButtonProps) {
    const [showDialog, setShowDialog] = useState(false);
    const { deleteJob, isLoading } = useJobDeletion(jobId, jobStatus);

    // Don't render button if job cannot be deleted
    if (!canDeleteJob(jobStatus)) {
        return null;
    }

    const handleDelete = () => {
        deleteJob(undefined, {
            onSuccess: () => {
                setShowDialog(false);
                onDeleted?.();
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
                disabled={isLoading}
            >
                {isLoading ? (
                    <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Deleting...
                    </>
                ) : (
                    <>
                        <Trash2 className="mr-2 h-4 w-4" />
                        Delete
                    </>
                )}
            </Button>

            <AlertDialog open={showDialog} onOpenChange={setShowDialog}>
                <AlertDialogContent>
                    <AlertDialogHeader>
                        <AlertDialogTitle>Delete Job?</AlertDialogTitle>
                        <AlertDialogDescription>
                            Are you sure you want to delete this job? This action cannot be undone.
                            All job data and results will be permanently removed from the server.
                        </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                        <AlertDialogCancel disabled={isLoading}>
                            Cancel
                        </AlertDialogCancel>
                        <AlertDialogAction
                            onClick={handleDelete}
                            disabled={isLoading}
                            className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                        >
                            {isLoading ? (
                                <>
                                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                    Deleting...
                                </>
                            ) : (
                                'Yes, delete permanently'
                            )}
                        </AlertDialogAction>
                    </AlertDialogFooter>
                </AlertDialogContent>
            </AlertDialog>
        </>
    );
}
