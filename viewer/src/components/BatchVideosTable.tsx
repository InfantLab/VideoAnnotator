// The videos of one batch: one compact line each, with what they share said
// once above them. Built for runs of tens of videos, where the flat jobs
// table (pipelines, delete and view on every row) becomes a long scroll of
// repeats. Deleting a single video lives on its job page.

import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Eye, RotateCcw, Search } from 'lucide-react';
import type { JobResponse } from '@/api/client';
import type { JobStatus } from '@/types/api';
import { JobCancelButton } from '@/components/JobCancelButton';
import { JobStatusBadge } from '@/components/JobStatusBadge';
import { canCancelJob } from '@/hooks/useJobCancellation';
import { jobErrorSummary } from '@/lib/jobOutcome';
import { queueLabel } from '@/lib/queuePosition';
import { settingsOf, wizardState } from '@/lib/wizardStart';
import {
  filterJobs,
  formatDuration,
  formatFileSize,
  OUTCOME_LABELS,
  OUTCOME_ORDER,
  outcomeOf,
  progressOf,
  sharedPipelines,
  videoFieldsOf,
  type VideoOutcome,
} from '@/lib/jobRow';

// Below this a search box is clutter; the whole run fits on screen.
const SEARCH_FROM = 10;

interface BatchVideosTableProps {
  jobs: JobResponse[];
  /** Anything still queued or running: show the progress column. */
  active: boolean;
  /** The run is still in first-run setup: show "Preparing" rather than 0%. */
  preparing?: boolean;
}

export function BatchVideosTable({ jobs, active, preparing = false }: BatchVideosTableProps) {
  const navigate = useNavigate();
  const [outcome, setOutcome] = useState<VideoOutcome | 'all'>('all');
  const [query, setQuery] = useState('');

  const shared = useMemo(() => sharedPipelines(jobs), [jobs]);
  const counts = useMemo(() => {
    const byOutcome = new Map<VideoOutcome, number>();
    for (const job of jobs) {
      const o = outcomeOf(job);
      byOutcome.set(o, (byOutcome.get(o) ?? 0) + 1);
    }
    return byOutcome;
  }, [jobs]);
  // A filter whose last video moved on (a failure retried) falls back to all.
  const effectiveOutcome = outcome !== 'all' && !counts.get(outcome) ? 'all' : outcome;
  const visible = useMemo(
    () => filterJobs(jobs, effectiveOutcome, query),
    [jobs, effectiveOutcome, query]
  );

  const rerun = (job: JobResponse) => {
    const label = videoFieldsOf(job).videoName;
    navigate('/jobs/new', { state: wizardState({ mode: 'rerun', jobId: job.id, label, ...settingsOf(job) }) });
  };

  const showFilters = counts.size > 1;
  const showSearch = jobs.length >= SEARCH_FROM;

  return (
    <div className="space-y-3">
      <div className="px-6 space-y-3">
        {shared && shared.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5 text-sm">
            <span className="text-muted-foreground mr-1">Pipelines, every video:</span>
            {shared.map((pipeline) => (
              <Badge key={pipeline} variant="outline" className="text-xs font-normal">
                {pipeline}
              </Badge>
            ))}
          </div>
        )}
        {(showFilters || showSearch) && (
          <div className="flex flex-wrap items-center gap-2">
            {showFilters && (
              <div className="flex flex-wrap gap-1.5" role="group" aria-label="Show videos">
                <Button
                  size="sm"
                  variant={effectiveOutcome === 'all' ? 'secondary' : 'ghost'}
                  className="h-7 px-2.5 text-xs"
                  onClick={() => setOutcome('all')}
                  aria-pressed={effectiveOutcome === 'all'}
                >
                  All {jobs.length}
                </Button>
                {OUTCOME_ORDER.filter((o) => counts.get(o)).map((o) => (
                  <Button
                    key={o}
                    size="sm"
                    variant={effectiveOutcome === o ? 'secondary' : 'ghost'}
                    className="h-7 px-2.5 text-xs"
                    onClick={() => setOutcome(o)}
                    aria-pressed={effectiveOutcome === o}
                  >
                    {counts.get(o)} {OUTCOME_LABELS[o]}
                  </Button>
                ))}
              </div>
            )}
            {showSearch && (
              <div className="relative ml-auto w-full sm:w-64">
                <Search className="absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
                <Input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Find a video"
                  aria-label="Find a video"
                  className="h-8 pl-7 text-sm"
                />
              </div>
            )}
          </div>
        )}
      </div>

      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="h-9">Video</TableHead>
            <TableHead className="h-9">Status</TableHead>
            {active && <TableHead className="h-9 w-[130px]">Progress</TableHead>}
            <TableHead className="h-9 text-right">Duration</TableHead>
            <TableHead className="h-9 text-right">Size</TableHead>
            {!shared && <TableHead className="h-9">Pipelines</TableHead>}
            <TableHead className="h-9 w-px" />
          </TableRow>
        </TableHeader>
        <TableBody>
          {visible.length === 0 ? (
            <TableRow>
              <TableCell colSpan={7} className="py-6 text-center text-muted-foreground">
                No videos match.
              </TableCell>
            </TableRow>
          ) : (
            visible.map((job) => {
              const { videoName, videoDuration, videoSize } = videoFieldsOf(job);
              const errorSummary = jobErrorSummary(job);
              const queued = queueLabel(job as JobResponse & Record<string, unknown>);
              const progress = progressOf(job);
              return (
                <TableRow
                  key={job.id}
                  onDoubleClick={() => navigate(`/jobs/${job.id}`)}
                  className="cursor-pointer hover:bg-muted/50"
                >
                  <TableCell className="py-2 font-medium max-w-[28rem]">
                    <div className="truncate" title={videoName}>
                      {videoName}
                    </div>
                    {errorSummary && (
                      <div
                        className={`truncate text-xs font-normal ${
                          job.status === 'failed' ? 'text-red-700' : 'text-orange-800'
                        }`}
                        title={errorSummary}
                      >
                        {errorSummary}
                      </div>
                    )}
                  </TableCell>
                  <TableCell className="py-2 whitespace-nowrap">
                    <JobStatusBadge
                      status={job.status}
                      errorMessage={job.error_message}
                      className="text-[10px] px-1.5 py-0"
                    />
                    {queued && <span className="ml-2 text-xs text-muted-foreground">{queued}</span>}
                  </TableCell>
                  {active && (
                    <TableCell className="py-2">
                      <div className="flex items-center gap-2">
                        <Progress value={progress} className="h-1.5 w-16" />
                        <span className="text-xs text-muted-foreground tabular-nums">
                          {preparing && job.status === 'running' && progress === 0
                            ? 'Preparing'
                            : `${Math.round(progress)}%`}
                        </span>
                      </div>
                    </TableCell>
                  )}
                  <TableCell className="py-2 text-right tabular-nums whitespace-nowrap">
                    {formatDuration(videoDuration)}
                  </TableCell>
                  <TableCell className="py-2 text-right tabular-nums whitespace-nowrap">
                    {formatFileSize(videoSize)}
                  </TableCell>
                  {!shared && (
                    <TableCell className="py-2 text-xs text-muted-foreground">
                      {(job.selected_pipelines ?? []).join(', ')}
                    </TableCell>
                  )}
                  <TableCell className="py-1.5 text-right whitespace-nowrap">
                    {job.status === 'completed' && (
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-7"
                        onClick={(e) => {
                          e.stopPropagation();
                          navigate(`/view/${job.id}`);
                        }}
                      >
                        <Eye className="h-4 w-4 mr-1" />
                        View
                      </Button>
                    )}
                    {job.status === 'failed' && (
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-7"
                        onClick={(e) => {
                          e.stopPropagation();
                          rerun(job);
                        }}
                      >
                        <RotateCcw className="h-4 w-4 mr-1" />
                        Fix and run again
                      </Button>
                    )}
                    {canCancelJob(job.status as JobStatus) && (
                      <JobCancelButton
                        jobId={job.id}
                        jobStatus={job.status as JobStatus}
                        size="icon"
                        variant="ghost"
                        className="h-7 w-7 text-muted-foreground"
                      />
                    )}
                  </TableCell>
                </TableRow>
              );
            })
          )}
        </TableBody>
      </Table>
    </div>
  );
}
