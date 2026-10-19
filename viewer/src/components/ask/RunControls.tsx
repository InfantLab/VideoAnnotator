import { FolderInput, Play, Save, Square } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';

export interface RunState {
  status: 'running' | 'done' | 'stopped' | 'failed';
  planned: number;
  done: number;
}

interface RunControlsProps {
  run: RunState | null;
  samples: number;
  disabled: boolean;
  onRun: () => void;
  onStop: () => void;
  onMoreVideos: () => void;
  onSaveWorkflow: () => void;
}

function runWords(run: RunState): string {
  if (run.status === 'running') return `${run.done} of ${run.planned}`;
  if (run.status === 'stopped') return `Stopped after ${run.done} of ${run.planned}`;
  if (run.status === 'failed') return `The run failed after ${run.done} of ${run.planned}`;
  return `Done: ${run.done} of ${run.planned}`;
}

/** The whole video (or its range) as a run; answers fill in as they come (FR-017). */
export function RunControls({ run, samples, disabled, ...on }: RunControlsProps) {
  const going = run?.status === 'running';
  return (
    <div className="space-y-2" data-testid="run-controls">
      <div className="flex flex-wrap gap-2">
        {going ? (
          <Button variant="destructive" onClick={on.onStop} className="gap-1">
            <Square className="h-4 w-4" /> Stop
          </Button>
        ) : (
          <Button onClick={on.onRun} disabled={disabled || samples === 0} className="gap-1">
            <Play className="h-4 w-4" /> Run on whole video
          </Button>
        )}
        <Button variant="outline" onClick={on.onMoreVideos} disabled={disabled} className="gap-1">
          <FolderInput className="h-4 w-4" /> Run on more videos
        </Button>
        <Button variant="ghost" onClick={on.onSaveWorkflow} disabled={disabled} className="gap-1">
          <Save className="h-4 w-4" /> Save as workflow
        </Button>
      </div>
      {run && (
        <div className="space-y-1" data-testid="run-progress">
          {going && <Progress value={run.planned ? (run.done / run.planned) * 100 : 0} />}
          <p className="text-sm">{runWords(run)}</p>
        </div>
      )}
    </div>
  );
}
