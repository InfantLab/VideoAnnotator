// Delete a whole run (spec 022): its results and results folder, never the
// videos. Without this, a 100-video run could only be removed video by video.

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
import { notify } from '@/lib/notify';
import { userMessage } from '@/lib/userMessage';

interface RunDeleteButtonProps {
  runId: string;
  /** A run of one video without a batch: deleted as that video's run. */
  single?: boolean;
  videoCount: number;
  name?: string;
  /** The run's results folder as the researcher sees it. */
  resultsFolder?: string | null;
  /** Where to go once it's gone; default the Videos page. */
  after?: string | null;
}

export function RunDeleteButton({ runId, single = false, videoCount, name, resultsFolder, after = '/videos' }: RunDeleteButtonProps) {
  const [open, setOpen] = useState(false);
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const remove = useMutation({
    mutationFn: () => (single ? apiClient.deleteJob(runId) : apiClient.deleteBatch(runId)),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['videos'] });
      queryClient.invalidateQueries({ queryKey: ['videoRuns'] });
      setOpen(false);
      notify({
        title: name ? `Deleted ${name}` : 'Run deleted',
        body: 'Its results are gone; the videos are where they were.',
        action: { label: 'Videos', to: '/videos' },
      });
      if (after) navigate(after);
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
                  The results of its {videoCount === 1 ? 'video' : `${videoCount} videos`} are permanently removed.
                  If it is still going, it is stopped first.
                </p>
                {resultsFolder && (
                  <p>
                    Its results folder will be deleted:{' '}
                    <span className="font-mono text-xs break-all">{resultsFolder}</span>
                  </p>
                )}
                <p>The videos themselves are not touched.</p>
                {remove.error && <p className="text-destructive">{userMessage(remove.error)}</p>}
              </div>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={remove.isPending}>Keep it</AlertDialogCancel>
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
