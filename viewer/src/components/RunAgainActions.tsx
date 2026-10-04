import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { Bookmark, Copy, Pencil, RotateCcw } from 'lucide-react';
import { apiClient } from '@/api/client';
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
        const result = await apiClient.rerunBatch(target.id);
        navigate(`/batches/${result.batch_id}`);
      }
    } catch (error) {
      const gone =
        error instanceof APIError &&
        (error.body as { error?: { code?: string } } | undefined)?.error?.code === 'RERUN_VIDEO_MISSING';
      setMessage({
        ok: false,
        text: gone
          ? 'Its video is no longer on the server. Use "Use these settings" and choose the video again.'
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
        <Button size="sm" onClick={() => setConfirming(true)} disabled={busy}>
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
            <AlertDialogDescription>
              A new {what} runs the same {target.kind === 'job' ? 'video' : 'videos'} with the same pipelines and
              settings. This one and its results are kept, so you can compare them. To change something first, use
              “Edit and run again”.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={runAgain}>Run again</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
};
