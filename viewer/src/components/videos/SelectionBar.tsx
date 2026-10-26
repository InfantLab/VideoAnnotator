import type { ReactNode } from 'react';
import { useNavigate } from 'react-router-dom';
import { Play, X } from 'lucide-react';

import { ExportMenu } from '@/components/ExportMenu';
import { Button } from '@/components/ui/button';
import { runState } from '@/lib/runTarget';
import { videosWord } from '@/lib/runs';

interface SelectionBarProps {
  paths: string[];
  /** The project the page shows, so a run on its videos stays in it. */
  project?: { id: string; name: string } | null;
  onClear: () => void;
  /** Add to project, Remove from project. */
  children?: ReactNode;
}

/** What to do with the rows selected (spec 029 FR-003). */
export function SelectionBar({ paths, project, onClear, children }: SelectionBarProps) {
  const navigate = useNavigate();
  const run = () =>
    navigate('/run', {
      state: runState({
        target: {
          kind: 'videos',
          paths,
          label: videosWord(paths.length),
          projectId: project?.id ?? null,
          projectName: project?.name ?? null,
        },
      }),
    });
  return (
    <div
      className="sticky top-14 z-20 flex flex-wrap items-center gap-2 rounded-md border bg-background/95 p-2 shadow-sm backdrop-blur"
      role="toolbar"
      aria-label="Selected videos"
      data-testid="selection-bar"
    >
      <span className="px-2 text-sm font-medium">{paths.length} selected</span>
      <Button size="sm" onClick={run}>
        <Play className="mr-1 h-4 w-4" />
        Run on {videosWord(paths.length)}…
      </Button>
      <ExportMenu scope={{ kind: 'videos', paths }} popover />
      {children}
      <Button variant="ghost" size="sm" onClick={onClear} aria-label="Clear the selection" className="ml-auto">
        <X className="h-4 w-4" />
      </Button>
    </div>
  );
}
