// One submission batch: aggregate state up top, its videos below.
//
// This is where "12 videos" stops being 12 scattered rows and becomes a thing
// you can look at, wait for, and act on as a whole.

import { Link, useParams } from 'react-router-dom';
import { formatDistanceToNow } from 'date-fns';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Skeleton } from '@/components/ui/skeleton';
import { ArrowLeft, Loader2, RefreshCw, RotateCcw, XCircle } from 'lucide-react';
import { JobsTable } from '@/components/JobsTable';
import { ErrorDisplay } from '@/components/ErrorDisplay';
import { parseApiError } from '@/lib/errorHandling';
import { formatUptime } from '@/lib/formatters';
import { useBatch, useBatchActions, useBatchJobs } from '@/hooks/useBatches';
import {
  batchDisplayName,
  isBatchCancellable,
  isBatchRetryable,
  type BatchSummary,
} from '@/types/batches';

function StatCard({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <Card>
      <CardContent className="pt-6">
        <p className="text-sm text-muted-foreground">{label}</p>
        <p className="text-2xl font-semibold mt-1">{value}</p>
        {hint && <p className="text-xs text-muted-foreground mt-1">{hint}</p>}
      </CardContent>
    </Card>
  );
}

function etaText(batch: BatchSummary): { value: string; hint?: string } {
  const remaining = batch.by_status.pending + batch.by_status.running;
  if (remaining === 0) return { value: 'Finished', hint: 'Nothing left to run' };
  if (batch.estimated_seconds_remaining === null) {
    return {
      value: 'Estimating…',
      // Being explicit beats showing a made-up number: the server can only
      // estimate once it has watched a real job in this batch finish.
      hint: 'Available once the first video finishes',
    };
  }
  return {
    value: `~${formatUptime(Math.round(batch.estimated_seconds_remaining))}`,
    hint: 'Based on how long this batch’s finished videos actually took',
  };
}

const BatchDetail = () => {
  const { batchId } = useParams<{ batchId: string }>();
  const { data: batch, isLoading, error, refetch } = useBatch(batchId);
  const {
    data: jobsData,
    isLoading: jobsLoading,
    refetch: refetchJobs,
  } = useBatchJobs(batchId);
  const { cancelBatch, retryBatch, isCancelling, isRetrying } = useBatchActions(batchId ?? '');

  if (error) {
    return (
      <div className="container mx-auto px-6 py-8 max-w-6xl space-y-4">
        <ErrorDisplay error={parseApiError(error)} />
        <Button onClick={() => refetch()} variant="outline">
          <RefreshCw className="h-4 w-4 mr-2" />
          Retry
        </Button>
      </div>
    );
  }

  if (isLoading || !batch) {
    return (
      <div className="container mx-auto px-6 py-8 max-w-6xl space-y-6">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-24 w-full" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  // A batch stops existing when its last member job is deleted; the server
  // reports that as an empty summary rather than a 404.
  if (batch.total === 0) {
    return (
      <div className="container mx-auto px-6 py-8 max-w-6xl space-y-6">
        <Link to="/jobs">
          <Button variant="outline" size="sm">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to runs
          </Button>
        </Link>
        <Card>
          <CardContent className="pt-6 text-center py-12">
            <p className="text-muted-foreground">
              This batch no longer has any jobs — they may have been deleted.
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  const eta = etaText(batch);
  const finished = batch.total - batch.by_status.pending - batch.by_status.running;
  const cancellable = isBatchCancellable(batch);
  const retryable = isBatchRetryable(batch);

  return (
    <div className="container mx-auto px-6 py-8 max-w-6xl space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <Link to="/jobs">
            <Button variant="outline" size="sm" className="mb-3">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to runs
            </Button>
          </Link>
          <h2 className="text-2xl font-bold text-gray-900 truncate">{batchDisplayName(batch)}</h2>
          <p className="text-muted-foreground">
            {batch.total} video{batch.total === 1 ? '' : 's'}
            {batch.created_at && (
              <>
                {' '}
                · submitted{' '}
                {formatDistanceToNow(new Date(batch.created_at), { addSuffix: true })}
              </>
            )}
            {batch.dataset_id && <> · dataset {batch.dataset_id}</>}
          </p>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              refetch();
              refetchJobs();
            }}
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
          {cancellable && (
            <Button variant="outline" size="sm" onClick={() => cancelBatch()} disabled={isCancelling}>
              {isCancelling ? (
                <Loader2 className="h-4 w-4 mr-1 animate-spin" />
              ) : (
                <XCircle className="h-4 w-4 mr-1" />
              )}
              Cancel all
            </Button>
          )}
          {retryable && (
            <Button variant="outline" size="sm" onClick={() => retryBatch()} disabled={isRetrying}>
              {isRetrying ? (
                <Loader2 className="h-4 w-4 mr-1 animate-spin" />
              ) : (
                <RotateCcw className="h-4 w-4 mr-1" />
              )}
              Retry {batch.by_status.failed + batch.by_status.cancelled}
            </Button>
          )}
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        <StatCard
          label="Progress"
          value={`${finished} / ${batch.total}`}
          hint={`${Math.round(batch.completion_percentage)}% of videos finished`}
        />
        <StatCard label="Time remaining" value={eta.value} hint={eta.hint} />
        <StatCard
          label="Outcome"
          value={`${batch.by_status.completed} ok · ${batch.by_status.failed} failed`}
          hint={
            batch.by_status.cancelled > 0
              ? `${batch.by_status.cancelled} cancelled`
              : undefined
          }
        />
      </div>

      <Card>
        <CardContent className="pt-6 space-y-2">
          <Progress value={batch.completion_percentage} className="h-2" />
          <div className="flex flex-wrap gap-1.5 pt-1">
            {batch.by_status.running > 0 && (
              <Badge variant="outline" className="text-xs bg-blue-100 text-blue-800 border-blue-200">
                {batch.by_status.running} running
              </Badge>
            )}
            {batch.by_status.pending > 0 && (
              <Badge
                variant="outline"
                className="text-xs bg-yellow-100 text-yellow-800 border-yellow-200"
              >
                {batch.by_status.pending} queued
              </Badge>
            )}
            {batch.by_status.completed > 0 && (
              <Badge
                variant="outline"
                className="text-xs bg-green-100 text-green-800 border-green-200"
              >
                {batch.by_status.completed} done
              </Badge>
            )}
            {batch.by_status.failed > 0 && (
              <Badge variant="outline" className="text-xs bg-red-100 text-red-800 border-red-200">
                {batch.by_status.failed} failed
              </Badge>
            )}
            {batch.by_status.cancelled > 0 && (
              <Badge variant="outline" className="text-xs bg-gray-100 text-gray-800 border-gray-200">
                {batch.by_status.cancelled} cancelled
              </Badge>
            )}
          </div>
        </CardContent>
      </Card>

      <Card>
        <div className="px-6 pt-4 pb-2">
          <h3 className="font-medium">Videos in this run</h3>
          <p className="text-xs text-muted-foreground">
            Double-click a row for that video&apos;s job details.
          </p>
        </div>
        {jobsLoading && !jobsData ? (
          <div className="flex justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin" />
          </div>
        ) : (
          <JobsTable jobs={jobsData?.jobs ?? []} onChanged={() => refetchJobs()} />
        )}
      </Card>
    </div>
  );
};

export default BatchDetail;
