import { Link, useNavigate } from 'react-router-dom';
import { Download, MoreHorizontal, Play } from 'lucide-react';

import { ExportMenu } from '@/components/ExportMenu';
import { ListNavigation } from '@/components/ListNavigation';
import { Button } from '@/components/ui/button';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import type { RunRef } from '@/lib/lanes';
import { videosHref, type ListContext } from '@/lib/listNavigation';
import { runState } from '@/lib/runTarget';
import type { ProjectRef } from '@/types/video';

/** The list the video came from, or, opened directly, the projects holding it (spec 028 FR-020). */
export function VideoContext({
  path,
  context,
  run,
  projects,
}: {
  path: string;
  context: ListContext | null;
  run?: string | null;
  projects: ProjectRef[];
}) {
  if (context) return <ListNavigation context={context} path={path} run={run} />;
  if (!projects.length) return null;
  return (
    <span className="min-w-0 truncate text-xs text-muted-foreground" data-testid="video-projects">
      in{' '}
      {projects.map((p, i) => (
        <span key={p.id}>
          {i > 0 && ', '}
          <Link
            to={videosHref({ kind: 'project', id: p.id })}
            className="text-foreground hover:underline"
            title={`${p.name}: all its videos`}
          >
            {p.name}
          </Link>
        </span>
      ))}
    </span>
  );
}

interface HeaderActionsProps {
  /** The video on the server, to run pipelines on; none for files from this computer. */
  video: { path: string; name: string } | null;
  /** The runs the lanes show: Export offers each. */
  shownRuns: RunRef[];
  onDebug: () => void;
  onLoadOther?: () => void;
}

/** Run…, Export and ⋯ (spec 028 FR-010, FR-021). */
export function HeaderActions({ video, shownRuns, onDebug, onLoadOther }: HeaderActionsProps) {
  const navigate = useNavigate();
  return (
    <>
      {video && (
        <Button
          variant="ghost"
          size="sm"
          className="h-8"
          title="Run pipelines on this video"
          onClick={() => navigate('/run', { state: runState({ target: { kind: 'videos', paths: [video.path], label: video.name } }) })}
        >
          <Play className="mr-1 h-4 w-4" />
          Run…
        </Button>
      )}
      {shownRuns.length > 0 && (
        <Popover>
          <PopoverTrigger asChild>
            <Button variant="ghost" size="sm" className="h-8">
              <Download className="mr-1 h-4 w-4" />
              Export
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-96 space-y-3 text-xs" align="end" data-testid="export-menu">
            {shownRuns.map((run) => (
              <section key={run.id}>
                {shownRuns.length > 1 && <p className="mb-1 font-medium">{run.label}</p>}
                <ExportMenu scope={{ kind: 'job', id: run.id }} />
              </section>
            ))}
          </PopoverContent>
        </Popover>
      )}
      <Popover>
        <PopoverTrigger asChild>
          <Button variant="ghost" size="icon" className="h-8 w-8" aria-label="More">
            <MoreHorizontal className="h-4 w-4" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-48 p-1 text-sm" align="end">
          <button type="button" className="w-full rounded px-2 py-1 text-left hover:bg-muted" onClick={onDebug}>
            Debug panel
          </button>
          {onLoadOther && (
            <button type="button" className="w-full rounded px-2 py-1 text-left hover:bg-muted" onClick={onLoadOther}>
              Load different files
            </button>
          )}
        </PopoverContent>
      </Popover>
    </>
  );
}
