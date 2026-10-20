import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { Pencil, RotateCcw } from 'lucide-react';
import { apiClient, type BatchRerunResult } from '@/api/client';
import { errorCode, userMessage } from '@/lib/userMessage';
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
import { Button } from '@/components/ui/button';
import { notify } from '@/lib/notify';
import { runState, type RunTarget } from '@/lib/runTarget';
import { videosWord } from '@/lib/runs';
import { hrefOf } from '@/lib/videosScope';
import { ServerFolderPicker, type ServerFolderSelection } from '@/components/ServerFolderPicker';
import { useIngestAccess } from '@/hooks/useIngestAccess';

type Rerun = Extract<RunTarget, { kind: 'rerun' }>;

/**
 * The next step after a result (spec 019, spec 029): run it again with its
 * settings to change in New run (the main choice), or unchanged, on the same
 * videos. A run of several videos first checks which are no longer where they
 * were, and offers to look for them.
 */
export const RunAgainActions = ({ target, size = 'sm' }: { target: Rerun; size?: 'sm' | 'default' }) => {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [checked, setChecked] = useState<BatchRerunResult | null>(null);
  const [locating, setLocating] = useState(false);
  const [folder, setFolder] = useState<ServerFolderSelection | null>(null);
  const { canReadInPlace } = useIngestAccess();
  const videoCount = target.single ? 1 : target.videoCount ?? 0;

  const relocation = () => (folder ? { relocateFolder: folder.path, recursive: folder.recursive } : {});

  const startRunAgain = async () => {
    setMessage(null);
    if (target.single) {
      setConfirming(true);
      return;
    }
    setBusy(true);
    setFolder(null);
    setLocating(false);
    try {
      setChecked(await apiClient.rerunBatch(target.runId, {}, { check: true }));
    } catch {
      setChecked(null); // an older server: just run it, as before
    } finally {
      setBusy(false);
      setConfirming(true);
    }
  };

  const lookInFolder = async () => {
    if (!folder) return;
    setBusy(true);
    try {
      setChecked(await apiClient.rerunBatch(target.runId, {}, { check: true, ...relocation() }));
      setLocating(false);
    } catch (error) {
      setMessage(userMessage(error));
    } finally {
      setBusy(false);
    }
  };

  const runAgain = async () => {
    setBusy(true);
    setMessage(null);
    try {
      const runId = target.single
        ? (await apiClient.rerunJob(target.runId)).id
        : (await apiClient.rerunBatch(target.runId, {}, relocation())).batch_id;
      queryClient.invalidateQueries({ queryKey: ['videos'] });
      queryClient.invalidateQueries({ queryKey: ['videoRuns'] });
      notify({
        title: `Running ${target.label} again`,
        body: 'You can close this tab; results will be here.',
        action: { label: 'Show run', to: hrefOf({ run: runId }) },
        tone: 'success',
      });
    } catch (error) {
      setMessage(
        errorCode(error) === 'RERUN_VIDEO_MISSING'
          ? `${userMessage(error)} Choose Run again… and pick the video where it is now.`
          : userMessage(error),
      );
    } finally {
      setBusy(false);
      setConfirming(false);
    }
  };

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-2">
        <Button size={size} onClick={() => navigate('/run', { state: runState({ target }) })}>
          <Pencil className="h-4 w-4 mr-2" /> Run again…
        </Button>
        <Button size={size} variant="outline" onClick={startRunAgain} disabled={busy}>
          <RotateCcw className="h-4 w-4 mr-2" /> Run again unchanged
        </Button>
      </div>
      {message && <p className="text-sm text-destructive">{message}</p>}
      <AlertDialog open={confirming} onOpenChange={setConfirming}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Run this again?</AlertDialogTitle>
            <AlertDialogDescription asChild>
              <div className="space-y-3">
                <p>
                  A new run on the same {target.single ? 'video' : videosWord(videoCount)}, with the same pipelines and
                  settings. This one and its results are kept, so you can compare them. To change something first, use
                  “Run again…”.
                </p>
                {checked && checked.skipped.length > 0 && (
                  <div className="space-y-1 text-foreground">
                    <p className="font-medium">
                      {checked.skipped.length} video{checked.skipped.length === 1 ? ' isn’t' : 's aren’t'} where{' '}
                      {checked.skipped.length === 1 ? 'it was' : 'they were'}, and won&apos;t run:
                    </p>
                    <ul className="list-disc pl-5 text-xs max-h-32 overflow-y-auto" data-testid="missing-videos">
                      {checked.skipped.map((s) => (
                        <li key={s.job_id} className="break-all">{s.reason}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {checked?.relocated && checked.relocated.length > 0 && (
                  <p className="text-foreground">
                    Found {checked.relocated.length} moved video{checked.relocated.length === 1 ? '' : 's'} in{' '}
                    <span className="font-mono text-xs break-all">{folder?.path}</span>; they run from there.
                  </p>
                )}
                {locating && (
                  <div className="space-y-2 max-h-80 overflow-y-auto">
                    <p className="text-foreground">Open the folder the videos are in now:</p>
                    <ServerFolderPicker selection={folder} onSelect={setFolder} folderOnly />
                    <Button size="sm" onClick={lookInFolder} disabled={busy || !folder}>
                      Look for them here
                    </Button>
                  </div>
                )}
              </div>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            {checked && checked.skipped.length > 0 && canReadInPlace && !locating && (
              <Button variant="outline" onClick={() => setLocating(true)}>
                Locate…
              </Button>
            )}
            <AlertDialogAction onClick={runAgain} disabled={!target.single && checked !== null && checked.skipped.length >= videoCount}>
              {checked && checked.skipped.length > 0 ? 'Run the rest' : 'Run again'}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
};
