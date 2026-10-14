// Delete a whole run (spec 022): its jobs and its results folder, never the
// videos. Without this, a 100-video run could only be removed job by job.

import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { Loader2, Trash2 } from 'lucide-react';
import { apiClient } from '@/api/client';
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
import { BatchQueryKeys } from '@/hooks/useBatches';

interface RunDeleteButtonProps {
  batchId: string;
  videoCount: number;
  /** The run's results folder as the researcher sees it. */
  resultsFolder?: string | null;
}

export function RunDeleteButton({ batchId, videoCount, resultsFolder }: RunDeleteButtonProps) {
  const [open, setOpen] = useState(false);
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const remove = useMutation({
    mutationFn: () => apiClient.deleteBatch(batchId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: BatchQueryKeys.all });
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
      setOpen(false);
      navigate('/jobs');
    },
  });

  return (
    <>
      <Button variant="outline" size="sm" onClick={() => setOpen(true)} disabled={remove.isPending}>
        <Trash2 className="h-4 w-4 mr-1" />
        Delete run
      </Button>
      <AlertDialog open={open} onOpenChange={setOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete this run?</AlertDialogTitle>
            <AlertDialogDescription asChild>
              <div className="space-y-2">
                <p>
                  All {videoCount} video{videoCount === 1 ? '' : 's'}&apos; jobs and results are
                  permanently removed. Jobs still running are cancelled first.
                </p>
                {resultsFolder && (
                  <p>
                    Its results folder will be deleted:{' '}
                    <span className="font-mono text-xs break-all">{resultsFolder}</span>
                  </p>
                )}
                <p>The original videos are not touched.</p>
                {remove.error && (
                  <p className="text-destructive">
                    {remove.error instanceof Error ? remove.error.message : String(remove.error)}
                  </p>
                )}
              </div>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={remove.isPending}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={(event) => {
                event.preventDefault();
                remove.mutate();
              }}
              disabled={remove.isPending}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {remove.isPending ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
              Yes, delete the run
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
