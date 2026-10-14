import { Link, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { ChevronLeft, ChevronRight } from 'lucide-react';

import { apiClient } from '@/api/client';
import { Button } from '@/components/ui/button';
import { useBatch } from '@/hooks/useBatches';
import { batchPosition } from '@/lib/batchNavigation';
import { batchDisplayName } from '@/types/batches';

interface BatchNavigationProps {
  jobId: string;
  batchId: string;
}

/** "Batch · 3 of 12" with previous/next between the batch's videos, for the results viewer's header. */
export function BatchNavigation({ jobId, batchId }: BatchNavigationProps) {
  const navigate = useNavigate();
  const { data: batch } = useBatch(batchId);
  const { data: members } = useQuery({
    queryKey: ['batch-navigation', batchId],
    // The whole batch, not the batch page's first 50, so n of N and next are right.
    queryFn: () => apiClient.getJobs(1, 1000, { batchId }),
    staleTime: 30_000,
  });

  const position = members ? batchPosition(members.jobs, jobId) : null;
  const name = batch ? batchDisplayName(batch) : 'Batch';
  const label = (target: { video_filename?: string | null; id: string }) => target.video_filename || target.id;

  return (
    <nav aria-label="Batch navigation" className="flex items-center gap-1 text-sm">
      <Button
        variant="ghost"
        size="sm"
        disabled={!position?.previous}
        onClick={() => position?.previous && navigate(`/view/${position.previous.id}`)}
        title={position?.previous ? `Previous: ${label(position.previous)}` : 'No earlier video with results'}
        aria-label="Previous video in batch"
      >
        <ChevronLeft className="h-4 w-4" />
      </Button>
      <Link to={`/batches/${batchId}`} className="truncate max-w-[220px] hover:underline" title="Open the batch">
        {name}
        {position && (
          <span className="text-muted-foreground">
            {' '}· {position.position} of {position.total}
          </span>
        )}
      </Link>
      <Button
        variant="ghost"
        size="sm"
        disabled={!position?.next}
        onClick={() => position?.next && navigate(`/view/${position.next.id}`)}
        title={position?.next ? `Next: ${label(position.next)}` : 'No later video with results'}
        aria-label="Next video in batch"
      >
        <ChevronRight className="h-4 w-4" />
      </Button>
    </nav>
  );
}
