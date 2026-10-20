import { AlertCircle, Loader2, Play } from 'lucide-react';

import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Term } from '@/components/Term';
import { totalDownloadLabel, type SetupNote } from '@/lib/runSetup';
import { videosWord } from '@/lib/runs';

interface StartRunProps {
  /** How many videos; null while a folder or project is still being listed. */
  count: number | null;
  /** A few of their names. */
  names: string[];
  /** Where they are: "in /data/pk", "already on the server". */
  where: string;
  pipelineTitles: string[];
  setupNotes: SetupNote[];
  name: string;
  onName: (name: string) => void;
  defaultName: string;
  project: string | null;
  starting: boolean;
  blocked: string | null;
  onStart: () => void;
}

/** What will run, on what, and the one button that starts it (spec 029 FR-017). */
export function StartRun({
  count,
  names,
  where,
  pipelineTitles,
  setupNotes,
  name,
  onName,
  defaultName,
  project,
  starting,
  blocked,
  onStart,
}: StartRunProps) {
  const download = totalDownloadLabel(setupNotes);
  return (
    <div className="space-y-4" data-testid="start-run">
      <dl className="grid gap-x-6 gap-y-2 text-sm sm:grid-cols-[8rem_1fr]">
        <dt className="text-muted-foreground">Videos</dt>
        <dd>
          {count === null ? 'Counting…' : videosWord(count)} {where}
          {names.length > 0 && (
            <span className="block text-xs text-muted-foreground">
              {names.slice(0, 5).join(', ')}
              {count !== null && count > 5 && `, and ${count - 5} more`}
            </span>
          )}
        </dd>
        <dt className="text-muted-foreground">To find</dt>
        <dd>{pipelineTitles.length ? pipelineTitles.join(', ') : 'Nothing chosen yet'}</dd>
        <dt className="text-muted-foreground">How long</dt>
        <dd>
          Estimated once the first video finishes, from how long these videos actually take.
          {setupNotes.length > 0 && (
            <span className="block text-xs text-muted-foreground">
              The first run downloads {download ?? 'model weights'}, so the first video takes several minutes longer.
            </span>
          )}
        </dd>
        {project && (
          <>
            <dt className="text-muted-foreground">
              <Term word="project">Project</Term>
            </dt>
            <dd>{project}</dd>
          </>
        )}
        <dt className="text-muted-foreground">
          <label htmlFor="run-name">
            <Term word="run">Run</Term> name
          </label>
        </dt>
        <dd>
          <Input id="run-name" value={name} onChange={(e) => onName(e.target.value)} placeholder={defaultName} className="max-w-sm" />
        </dd>
      </dl>
      {blocked && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{blocked}</AlertDescription>
        </Alert>
      )}
      <Button size="lg" onClick={onStart} disabled={starting || !!blocked || !count}>
        {starting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
        {starting ? 'Starting…' : count ? `Start run on ${videosWord(count)}` : 'Start run'}
      </Button>
      <p className="text-xs text-muted-foreground">You can close this tab once it has started; results will be on the Videos page.</p>
    </div>
  );
}
