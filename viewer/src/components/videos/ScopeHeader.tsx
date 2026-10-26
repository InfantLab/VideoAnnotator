import type { ReactNode } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Play } from 'lucide-react';

import type { ExportScope } from '@/api/client';
import { ExportMenu } from '@/components/ExportMenu';
import { Button } from '@/components/ui/button';
import { runState, type RunTarget } from '@/lib/runTarget';
import { finishedCount, runStatus, videosWord } from '@/lib/runs';
import { hrefOf } from '@/lib/videosScope';
import type { RunSummary, VideoItem } from '@/types/videos';

/** Rows still moving, for a scope that has no runs of its own (a folder, all videos). */
function moving(videos: VideoItem[]): { running: number; waiting: number } {
  let running = 0;
  let waiting = 0;
  for (const video of videos) {
    const states = Object.values(video.results).map((r) => r.state);
    if (states.includes('running')) running += 1;
    else if (states.includes('waiting')) waiting += 1;
  }
  return { running, waiting };
}

/** "● Peekaboo faces: running 31/48" for each of the scope's active runs (spec 029 US1 scenario 3). */
export function ActiveRuns({ runs, videos }: { runs: RunSummary[]; videos: VideoItem[] }) {
  const active = runs.filter((r) => r.active);
  if (active.length) {
    return (
      <ul className="space-y-1 text-sm" data-testid="scope-progress">
        {active.map((run) => (
          <li key={run.id}>
            <Link to={hrefOf({ run: run.id })} className="hover:underline">
              <span className="mr-1 animate-pulse text-blue-600">●</span>
              {run.name}: {runStatus(run)}
            </Link>
            <span className="ml-2 inline-block h-1.5 w-32 overflow-hidden rounded bg-muted align-middle">
              <span className="block h-full bg-primary" style={{ width: `${run.total ? (finishedCount(run) / run.total) * 100 : 0}%` }} />
            </span>
          </li>
        ))}
      </ul>
    );
  }
  const { running, waiting } = moving(videos);
  if (!running && !waiting) return null;
  return (
    <p className="text-sm" data-testid="scope-progress">
      <span className="mr-1 animate-pulse text-blue-600">●</span>
      {running > 0 && `${videosWord(running)} running`}
      {running > 0 && waiting > 0 && ', '}
      {waiting > 0 && `${videosWord(waiting)} waiting`}
    </p>
  );
}

interface ScopeHeaderProps {
  title: ReactNode;
  count: number;
  subtitle?: ReactNode;
  runTarget: RunTarget | null;
  exportScope: ExportScope | null;
  runs?: RunSummary[];
  videos: VideoItem[];
  /** More actions (a project's), before Export. */
  children?: ReactNode;
  below?: ReactNode;
  /** Rows are selected: the selection bar's actions replace Run and Export (viewer design P5). */
  selecting?: boolean;
}

/** The scope's name and count, Run… and Export ▾ for what's shown (spec 029 FR-003). */
export function ScopeHeader({ title, count, subtitle, runTarget, exportScope, runs = [], videos, children, below, selecting = false }: ScopeHeaderProps) {
  const navigate = useNavigate();
  return (
    <section className="space-y-2" data-testid="scope-header">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h1 className="truncate text-2xl font-semibold">{title}</h1>
          <p className="text-sm text-muted-foreground">
            {videosWord(count)}
            {subtitle && <> · {subtitle}</>}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {runTarget && !selecting && (
            <Button size="sm" onClick={() => navigate('/run', { state: runState({ target: runTarget }) })} disabled={count === 0}>
              <Play className="mr-1 h-4 w-4" />
              Run on {videosWord(count)}…
            </Button>
          )}
          {children}
          {exportScope && count > 0 && !selecting && <ExportMenu scope={exportScope} popover />}
        </div>
      </div>
      <ActiveRuns runs={runs} videos={videos} />
      {below}
    </section>
  );
}
