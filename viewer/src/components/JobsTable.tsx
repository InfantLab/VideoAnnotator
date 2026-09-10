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
import { AlertCircle, Eye, RotateCcw } from 'lucide-react';
import type { JobResponse } from '@/api/client';
import type { JobStatus } from '@/types/api';
import { JobCancelButton } from '@/components/JobCancelButton';
import { JobDeleteButton } from '@/components/JobDeleteButton';
import { canCancelJob } from '@/hooks/useJobCancellation';
import { canDeleteJob } from '@/hooks/useJobDeletion';

const STATUS_CLASSES: Record<string, string> = {
  pending: 'bg-yellow-100 text-yellow-800 hover:bg-yellow-100 border-yellow-200',
  running: 'bg-blue-100 text-blue-800 hover:bg-blue-100 border-blue-200',
  completed: 'bg-green-100 text-green-800 hover:bg-green-100 border-green-200',
  failed: 'bg-red-100 text-red-800 hover:bg-red-100 border-red-200',
  cancelled: 'bg-gray-100 text-gray-800 hover:bg-gray-100 border-gray-200',
  cancelling: 'bg-orange-100 text-orange-800 hover:bg-orange-100 border-orange-200',
};

const PARTIAL_SUCCESS_CLASS =
  'bg-orange-100 text-orange-800 hover:bg-orange-100 border-orange-200';

function getStatusBadge(status: string, errorMessage?: string | null) {
  // A job that completed but carries an error message succeeded only partially.
  const isPartialSuccess = status === 'completed' && !!errorMessage;
  const className = isPartialSuccess
    ? PARTIAL_SUCCESS_CLASS
    : STATUS_CLASSES[status] ?? STATUS_CLASSES.pending;

  const badge = (
    <Badge variant="outline" className={className}>
      {status.toUpperCase()}
      {(isPartialSuccess || (status === 'failed' && errorMessage)) && (
        <AlertCircle className="ml-1 h-3 w-3 inline" />
      )}
    </Badge>
  );

  if ((status === 'failed' || isPartialSuccess) && errorMessage) {
    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>{badge}</TooltipTrigger>
          <TooltipContent className="max-w-xs">
            <p className="font-semibold">{isPartialSuccess ? 'Partial Success:' : 'Error:'}</p>
            <p>{errorMessage}</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  }

  return badge;
}

function formatDuration(seconds: number | null) {
  if (!seconds) return 'N/A';
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function formatFileSize(bytes: number | null) {
  if (!bytes) return 'N/A';
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const getString = (value: unknown): string | undefined =>
  typeof value === 'string' ? value : undefined;

const getNumber = (value: unknown): number | null =>
  typeof value === 'number' && Number.isFinite(value) ? value : null;

/**
 * Defensive field access — the server has used several names for these over
 * its versions, and a viewer pointed at an older server should still show a
 * filename rather than "N/A".
 */
function videoFieldsOf(job: JobResponse) {
  const record = job as JobResponse & Record<string, unknown>;

  let videoName =
    getString(record.video_filename) ??
    getString(record.filename) ??
    getString(record.video_name);

  const videoPath = getString(record.video_path);
  if (!videoName && videoPath) {
    videoName = videoPath.split(/[/\\]/).pop() || videoPath;
  }

  return {
    videoName: videoName || 'N/A',
    videoDuration:
      getNumber(record.video_duration_seconds) ?? getNumber(record.duration_seconds),
    videoSize: getNumber(record.video_size_bytes) ?? getNumber(record.file_size_bytes),
  };
}

interface JobsTableProps {
  jobs: JobResponse[];
  /** Called after a destructive action so the caller can refetch its own query. */
  onChanged?: () => void;
  /** Rendered in the table body when there are no jobs. */
  emptyState?: React.ReactNode;
  /** Hide the progress column where it adds nothing (e.g. an all-finished batch). */
  showProgress?: boolean;
}

export function JobsTable({
  jobs,
  onChanged,
  emptyState,
  showProgress = true,
}: JobsTableProps) {
  const navigate = useNavigate();

  const handleRetryJob = (job: JobResponse) => {
    navigate('/jobs/new', {
      state: {
        retryJobId: job.id,
        retryJobConfig: job.config,
        retryJobPipelines: job.selected_pipelines,
        retryJobVideoFilename: (job as JobResponse & Record<string, unknown>)
          .video_filename as string | undefined,
      },
    });
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
            // Real per-pipeline progress from the server (spec 006/008), not a
            // status-to-number guess.
            const progress =
              getNumber((job as JobResponse & Record<string, unknown>).progress_percentage) ?? 0;

            return (
              <TableRow
                key={job.id}
                onDoubleClick={() => navigate(`/jobs/${job.id}`)}
                className="cursor-pointer hover:bg-muted/50"
              >
                <TableCell className="max-w-[220px] truncate font-medium" title={videoName}>
                  {videoName}
                </TableCell>
                <TableCell>{getStatusBadge(job.status, job.error_message)}</TableCell>
                {showProgress && (
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <Progress value={progress} className="h-1.5 w-16" />
                      <span className="text-xs text-muted-foreground tabular-nums">
                        {Math.round(progress)}%
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
                        Retry
                      </Button>
                    )}
                    {canDeleteJob(job.status as JobStatus) && (
                      <JobDeleteButton
                        jobId={job.id}
                        jobStatus={job.status as JobStatus}
                        size="sm"
                        variant="outline"
                        onDeleted={onChanged}
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
