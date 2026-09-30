// One submission batch, rendered as a single unit of work.
//
// This is the piece that replaces N independent job rows: submitting 12 videos
// produces one card showing real aggregate progress, a real ETA, and bulk
// actions — not 12 rows spread across two pages of a paginated table.

import { Link } from 'react-router-dom';
import { formatDistanceToNow } from 'date-fns';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { ChevronRight, Loader2, RotateCcw, XCircle } from 'lucide-react';
import { formatUptime } from '@/lib/formatters';
import { useBatchActions } from '@/hooks/useBatches';
import {
  batchDisplayName,
  isBatchCancellable,
  isBatchRetryable,
  type BatchSummary,
} from '@/types/batches';

/**
 * Per-state counts as compact chips. Zero-count states are omitted entirely —
 * a batch that is simply running shouldn't display four zeroes.
 */
const STATUS_CHIPS: Array<{
  key: keyof BatchSummary['by_status'];
  label: string;
  className: string;
}> = [
  { key: 'running', label: 'running', className: 'bg-blue-100 text-blue-800 border-blue-200' },
  { key: 'pending', label: 'queued', className: 'bg-yellow-100 text-yellow-800 border-yellow-200' },
  { key: 'completed', label: 'done', className: 'bg-green-100 text-green-800 border-green-200' },
  { key: 'failed', label: 'failed', className: 'bg-red-100 text-red-800 border-red-200' },
  { key: 'cancelled', label: 'cancelled', className: 'bg-gray-100 text-gray-800 border-gray-200' },
];

/**
 * The batch's headline state, in the words a researcher would use.
 * `estimated_seconds_remaining` is null until the server has watched at least
 * one job finish, so before that we say we're estimating rather than inventing
 * a number — the old wizard's "files × 7 minutes" guess is exactly what this
 * replaces.
 */
function progressLabel(batch: BatchSummary): string {
  const { pending, running } = batch.by_status;
  const finished = batch.total - pending - running;

  if (pending + running === 0) {
    return `All ${batch.total} finished`;
  }
  const eta =
    batch.estimated_seconds_remaining !== null
      ? `~${formatUptime(Math.round(batch.estimated_seconds_remaining))} left`
      : 'estimating time left…';
  return `${finished} of ${batch.total} finished · ${eta}`;
}

export function BatchCard({ batch }: { batch: BatchSummary }) {
  const { cancelBatch, retryBatch, isCancelling, isRetrying } = useBatchActions(batch.batch_id);
  const cancellable = isBatchCancellable(batch);
  const retryable = isBatchRetryable(batch);
  const active = batch.by_status.running + batch.by_status.pending > 0;

  return (
    <Card className="transition-colors hover:bg-muted/30">
      <CardContent className="pt-6 space-y-4">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <Link
              to={`/batches/${batch.batch_id}`}
              className="group flex items-center gap-1 font-medium hover:underline"
            >
              <span className="truncate">{batchDisplayName(batch)}</span>
              <ChevronRight className="h-4 w-4 shrink-0 text-muted-foreground group-hover:translate-x-0.5 transition-transform" />
            </Link>
            <p className="text-sm text-muted-foreground mt-0.5">
              {batch.total} video{batch.total === 1 ? '' : 's'}
              {batch.created_at && (
                <> · submitted {formatDistanceToNow(new Date(batch.created_at), { addSuffix: true })}</>
              )}
              {batch.dataset_id && <> · dataset {batch.dataset_id}</>}
            </p>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            {cancellable && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => cancelBatch()}
                disabled={isCancelling}
              >
                {isCancelling ? (
                  <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                ) : (
                  <XCircle className="h-4 w-4 mr-1" />
                )}
                Cancel all
              </Button>
            )}
            {retryable && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => retryBatch()}
                disabled={isRetrying}
              >
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

        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">{progressLabel(batch)}</span>
            <span className="tabular-nums text-muted-foreground">
              {Math.round(batch.completion_percentage)}%
            </span>
          </div>
          <Progress value={batch.completion_percentage} className="h-2" />
        </div>

        <div className="flex flex-wrap items-center gap-1.5">
          {STATUS_CHIPS.filter(({ key }) => batch.by_status[key] > 0).map(
            ({ key, label, className }) => (
              <Badge key={key} variant="outline" className={`text-xs ${className}`}>
                {batch.by_status[key]} {label}
              </Badge>
            )
          )}
          {active && (
            <span className="text-xs text-muted-foreground ml-1 inline-flex items-center gap-1">
              <Loader2 className="h-3 w-3 animate-spin" />
              in progress
            </span>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
