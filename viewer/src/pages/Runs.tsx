import { useState } from 'react';
import { Link } from 'react-router-dom';
import { Loader2, Play } from 'lucide-react';

import { APIError } from '@/api/handleError';
import { NeedsServer } from '@/components/NeedsServer';
import { PageHelp } from '@/components/PageHelp';
import { RunList } from '@/components/runs/RunRow';
import { Button } from '@/components/ui/button';
import { useRuns } from '@/hooks/useRuns';
import { userMessage } from '@/lib/userMessage';

const PAGE = 50;

/** Every run, one row each, those going on first (UAT D2, #1, #6). */
export default function Runs() {
  const [limit, setLimit] = useState(PAGE);
  const { data, error, isLoading, isFetching } = useRuns(limit);

  if (error instanceof APIError && error.status === 0) {
    return (
      <div className="container mx-auto px-6 py-8">
        <NeedsServer what="Runs lists the pipelines you've run on a VideoAnnotator server's videos, and how each went." />
      </div>
    );
  }

  return (
    <div className="container mx-auto max-w-5xl space-y-4 px-6 py-6">
      <h1 className="text-2xl font-semibold">Runs</h1>
      <PageHelp page="videos">Everything you've run, those going on first. Open a run to see its videos and take its results.</PageHelp>
      {error && !data && (
        <p className="text-sm text-destructive" role="alert">
          {userMessage(error)}
        </p>
      )}
      {isLoading && !data ? (
        <p className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" /> Loading runs…
        </p>
      ) : data && data.runs.length === 0 ? (
        <div className="space-y-3 rounded-md border border-dashed p-8 text-center text-sm text-muted-foreground" data-testid="runs-empty">
          <p>No runs yet.</p>
          <Button asChild>
            <Link to="/run">
              <Play className="mr-1 h-4 w-4" /> Run pipelines on videos
            </Link>
          </Button>
        </div>
      ) : (
        data && (
          <>
            <RunList runs={data.runs} />
            {data.total > data.runs.length && (
              <div className="text-center">
                <Button variant="ghost" size="sm" onClick={() => setLimit((n) => n + PAGE)} disabled={isFetching}>
                  {isFetching && <Loader2 className="mr-1 h-4 w-4 animate-spin" />}
                  Show older runs
                </Button>
              </div>
            )}
          </>
        )
      )}
    </div>
  );
}
