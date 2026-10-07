// The per-video job table.
//
// Extracted from `pages/Jobs.tsx` so the same rows can be shown in two places:
// the flat jobs list, and inside one batch (`pages/BatchDetail.tsx`). Batch
// context is what a researcher usually wants — this table is the drill-down.

import { useNavigate } from 'react-router-dom';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { Eye, RotateCcw } from 'lucide-react';
import type { JobResponse } from '@/api/client';
import type { JobStatus } from '@/types/api';
import { JobCancelButton } from '@/components/JobCancelButton';
import { JobDeleteButton } from '@/components/JobDeleteButton';
import { canCancelJob } from '@/hooks/useJobCancellation';
import { canDeleteJob } from '@/hooks/useJobDeletion';
import { jobErrorSummary } from '@/lib/jobOutcome';
import { queueLabel } from '@/lib/queuePosition';
import { settingsOf, wizardState } from '@/lib/wizardStart';
import { formatDuration, formatFileSize, progressOf, videoFieldsOf } from '@/lib/jobRow';
import { JobStatusBadge } from '@/components/JobStatusBadge';

interface JobsTableProps {
  jobs: JobResponse[];
  /** Called after a destructive action so the caller can refetch its own query. */
  onChanged?: () => void;
  /** Rendered in the table body when there are no jobs. */
  emptyState?: React.ReactNode;
  /** Hide the progress column where it adds nothing (e.g. an all-finished batch). */
  showProgress?: boolean;
  /** The run is still in first-run setup: show "Preparing" rather than 0% on running jobs. */
  preparing?: boolean;
}

export function JobsTable({
  jobs,
  onChanged,
  emptyState,
  showProgress = true,
  preparing = false,
}: JobsTableProps) {
  const navigate = useNavigate();

  // "Fix and run again" (spec 019): the job's own video, its settings to change.
  const handleRetryJob = (job: JobResponse) => {
    const label = ((job as JobResponse & Record<string, unknown>).video_filename as string | undefined) ?? `job ${job.id.slice(0, 8)}`;
    navigate('/jobs/new', { state: wizardState({ mode: 'rerun', jobId: job.id, label, ...settingsOf(job) }) });
  };

  const columnCount = showProgress ? 7 : 6;

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Video</TableHead>
          <TableHead>Status</TableHead>
          {showProgress && <TableHead className="w-[140px]">Progress</TableHead>}
          <TableHead>Duration</TableHead>
          <TableHead>Size</TableHead>
          <TableHead>Pipelines</TableHead>
          <TableHead>Actions</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {jobs.length === 0 ? (
          <TableRow>
            <TableCell colSpan={columnCount} className="text-center py-8">
              {emptyState ?? <p className="text-muted-foreground">No jobs found</p>}
            </TableCell>
          </TableRow>
        ) : (
          jobs.map((job) => {
            const { videoName, videoDuration, videoSize } = videoFieldsOf(job);
            const errorSummary = jobErrorSummary(job);
            const queued = queueLabel(job as JobResponse & Record<string, unknown>);
            // Real per-pipeline progress from the server (spec 006/008), not a
            // status-to-number guess.
            const progress = progressOf(job);

            return (
              <TableRow
                key={job.id}
                onDoubleClick={() => navigate(`/jobs/${job.id}`)}
                className="cursor-pointer hover:bg-muted/50"
              >
                <TableCell className="max-w-[220px] truncate font-medium" title={videoName}>
                  {videoName}
                </TableCell>
                <TableCell>
                  <JobStatusBadge status={job.status} errorMessage={job.error_message} />
                  {queued && <p className="mt-1 text-xs text-muted-foreground">{queued}</p>}
                  {errorSummary && (
                    <p
                      className={`mt-1 max-w-[280px] line-clamp-2 text-xs ${
                        job.status === 'failed' ? 'text-red-700' : 'text-orange-800'
                      }`}
                      title={errorSummary}
                    >
                      {errorSummary}
                    </p>
                  )}
                </TableCell>
                {showProgress && (
                  <TableCell>
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
                <TableCell>{formatDuration(videoDuration)}</TableCell>
                <TableCell>{formatFileSize(videoSize)}</TableCell>
                <TableCell>
                  <div className="flex flex-wrap gap-1">
                    {job.selected_pipelines?.slice(0, 2).map((pipeline) => (
                      <Badge key={pipeline} variant="outline" className="text-xs">
                        {pipeline}
                      </Badge>
                    ))}
                    {job.selected_pipelines && job.selected_pipelines.length > 2 && (
                      <Badge variant="outline" className="text-xs">
                        +{job.selected_pipelines.length - 2}
                      </Badge>
                    )}
                  </div>
                </TableCell>
                <TableCell>
                  <div className="flex items-center gap-2">
                    {canCancelJob(job.status as JobStatus) && (
                      <JobCancelButton
                        jobId={job.id}
                        jobStatus={job.status as JobStatus}
                        size="sm"
                        variant="outline"
                      />
                    )}
                    {job.status === 'failed' && (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleRetryJob(job);
                        }}
                      >
                        <RotateCcw className="h-4 w-4 mr-1" />
                        Fix and run again
                      </Button>
                    )}
                    {canDeleteJob(job.status as JobStatus) && (
                      <JobDeleteButton
                        jobId={job.id}
                        jobStatus={job.status as JobStatus}
                        size="sm"
                        variant="outline"
                        onDeleted={onChanged}
                        resultsFolder={job.results_folder?.display_path}
                      />
                    )}
                    {job.status === 'completed' && (
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={(e) => {
                                e.stopPropagation();
                                navigate(`/view/${job.id}`);
                              }}
                            >
                              <Eye className="h-4 w-4 mr-1" />
                              View
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent>
                            <p>View Results</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    )}
                  </div>
                </TableCell>
              </TableRow>
            );
          })
        )}
      </TableBody>
    </Table>
  );
}
