import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { Bookmark, Copy, Pencil, RotateCcw } from 'lucide-react';
import { apiClient, type BatchRerunResult } from '@/api/client';
import { APIError } from '@/api/handleError';
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
import { Input } from '@/components/ui/input';
import { parseApiError } from '@/lib/errorHandling';
import { wizardState, type StartSettings } from '@/lib/wizardStart';
import { ServerFolderPicker, type ServerFolderSelection } from '@/components/ServerFolderPicker';
import { useIngestAccess } from '@/hooks/useIngestAccess';

export type RunAgainTarget =
  | { kind: 'job'; id: string; label: string; settings: StartSettings }
  | { kind: 'batch'; id: string; label: string; videoCount: number; settings: StartSettings };

/**
 * The next step after a result (spec 019), shown on the job and batch pages
 * where the decision is made: run it again as is, change something and run
 * it again, use its settings on other videos, or save them as a preset.
 */
export const RunAgainActions = ({ target }: { target: RunAgainTarget }) => {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [confirming, setConfirming] = useState(false);
  const [naming, setNaming] = useState(false);
  const [presetName, setPresetName] = useState(target.label);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(null);
  // Spec 022: before a batch runs again, which videos aren't where they were,
  // and (after "Locate…") which of them were found in the chosen folder.
  const [checked, setChecked] = useState<BatchRerunResult | null>(null);
  const [locating, setLocating] = useState(false);
  const [folder, setFolder] = useState<ServerFolderSelection | null>(null);
  const { canReadInPlace } = useIngestAccess();

  const relocation = () => (folder ? { relocateFolder: folder.path, recursive: folder.recursive } : {});

  const startRunAgain = async () => {
    if (target.kind === 'job') {
      setConfirming(true);
      return;
    }
    setBusy(true);
    setMessage(null);
    setFolder(null);
    setLocating(false);
    try {
      setChecked(await apiClient.rerunBatch(target.id, {}, { check: true }));
    } catch {
      setChecked(null); // an older server: just run it, as before
    } finally {
      setBusy(false);
      setConfirming(true);
    }
  };

  const lookInFolder = async () => {
    if (target.kind !== 'batch' || !folder) return;
    setBusy(true);
    try {
      setChecked(await apiClient.rerunBatch(target.id, {}, { check: true, ...relocation() }));
      setLocating(false);
    } catch (error) {
      setMessage({ ok: false, text: parseApiError(error).message });
    } finally {
      setBusy(false);
    }
  };

  const editAndRunAgain = () =>
    navigate('/jobs/new', {
      state: wizardState(
        target.kind === 'job'
          ? { mode: 'rerun', jobId: target.id, label: target.label, ...target.settings }
          : { mode: 'rerunBatch', batchId: target.id, label: target.label, videoCount: target.videoCount, ...target.settings },
      ),
    });

  const useSettings = () =>
    navigate('/jobs/new', { state: wizardState({ mode: 'settings', label: target.label, ...target.settings }) });

  const runAgain = async () => {
    setBusy(true);
    setMessage(null);
    try {
      if (target.kind === 'job') {
        const job = await apiClient.rerunJob(target.id);
        navigate(`/jobs/${job.id}`);
      } else {
        const result = await apiClient.rerunBatch(target.id, {}, relocation());
        navigate(`/batches/${result.batch_id}`);
      }
    } catch (error) {
      const gone =
        error instanceof APIError &&
        (error.body as { error?: { code?: string } } | undefined)?.error?.code === 'RERUN_VIDEO_MISSING';
      setMessage({
        ok: false,
        text: gone
          ? `${parseApiError(error).message}. Use "Use these settings" and choose the video where it is now.`
          : parseApiError(error).message,
      });
    } finally {
      setBusy(false);
      setConfirming(false);
    }
  };

  const savePreset = async () => {
    setBusy(true);
    try {
      const preset = await apiClient.createPreset({
        name: presetName.trim(),
        selected_pipelines: target.settings.selectedPipelines ?? [],
        config: target.settings.config ?? {},
      });
      queryClient.invalidateQueries({ queryKey: ['presets'] });
      setMessage({ ok: true, text: `Saved preset “${preset.name}”.` });
      setNaming(false);
    } catch (error) {
      setMessage({
        ok: false,
        text:
          error instanceof APIError && error.status === 409
            ? `You already have a preset named “${presetName.trim()}”.`
            : parseApiError(error).message,
      });
    } finally {
      setBusy(false);
    }
  };

  const what = target.kind === 'job' ? 'job' : 'batch';
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-2">
        <Button size="sm" onClick={startRunAgain} disabled={busy}>
          <RotateCcw className="h-4 w-4 mr-2" /> Run again
        </Button>
        <Button size="sm" variant="outline" onClick={editAndRunAgain}>
          <Pencil className="h-4 w-4 mr-2" /> Edit and run again
        </Button>
        <Button size="sm" variant="outline" onClick={useSettings}>
          <Copy className="h-4 w-4 mr-2" /> Use these settings on other videos
        </Button>
        <Button size="sm" variant="outline" onClick={() => setNaming(!naming)}>
          <Bookmark className="h-4 w-4 mr-2" /> Save as preset
        </Button>
      </div>
      {naming && (
        <form
          className="flex items-center gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            if (presetName.trim()) savePreset();
          }}
        >
          <Input
            aria-label="Preset name"
            className="h-8 w-64"
            value={presetName}
            onChange={(e) => setPresetName(e.target.value)}
            autoFocus
          />
          <Button type="submit" size="sm" disabled={busy || !presetName.trim()}>
            Save
          </Button>
        </form>
      )}
      {message && (
        <p className={`text-sm ${message.ok ? 'text-muted-foreground' : 'text-destructive'}`}>{message.text}</p>
      )}
      <AlertDialog open={confirming} onOpenChange={setConfirming}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Run this {what} again?</AlertDialogTitle>
            <AlertDialogDescription asChild>
              <div className="space-y-3">
                <p>
                  A new {what} runs the same {target.kind === 'job' ? 'video' : 'videos'} with the same pipelines and
                  settings. This one and its results are kept, so you can compare them. To change something first,
                  use “Edit and run again”.
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
            <AlertDialogAction
              onClick={runAgain}
              disabled={target.kind === 'batch' && checked !== null && checked.skipped.length >= target.videoCount}
            >
              {checked && checked.skipped.length > 0 ? 'Run the rest' : 'Run again'}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
};
