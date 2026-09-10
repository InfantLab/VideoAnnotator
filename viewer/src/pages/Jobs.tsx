import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { formatDistanceToNow } from "date-fns";
import { apiClient, type JobResponse } from "@/api/client";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Loader2, RefreshCw, Play, Layers } from "lucide-react";
import { Link } from "react-router-dom";
import { useToast } from "@/hooks/use-toast";
import vavIcon from "@/assets/v-a-v.icon.png";
import { ErrorDisplay } from "@/components/ErrorDisplay";
import { parseApiError } from "@/lib/errorHandling";
import { JobsTable } from "@/components/JobsTable";
import { BatchCard } from "@/components/BatchCard";
import { useBatches } from "@/hooks/useBatches";

/**
 * Enhances authentication error messages with actionable guidance
 * CANONICAL error messages - used consistently across the app
 */
function enhanceAuthError(error: unknown) {
  const parsed = parseApiError(error);
  const errorMessage = parsed.message;
  const currentToken = localStorage.getItem('videoannotator_api_token') || '';

  // Check if it's an authentication error
  // Note: v1.3.0 server may return 404 "Not Found" for unauthenticated /api/v1/jobs requests
  if (
    errorMessage.includes('API key required') ||
    errorMessage.includes('AUTH_REQUIRED') ||
    errorMessage.includes('401') ||
    errorMessage.includes('404') ||
    errorMessage.includes('Not Found') ||
    errorMessage.includes('Unauthorized') ||
    errorMessage.includes('authentication')
  ) {
    // Check if they have the placeholder token
    const isPlaceholderToken = currentToken === 'dev-token' || currentToken === 'test-token';

    if (isPlaceholderToken) {
      return {
        message: 'Authentication Required',
        hint: 'You have a placeholder token ("dev-token") that doesn\'t work. Go to Settings and clear the API Token field to connect anonymously.',
        fieldErrors: [],
        code: parsed.code,
        requestId: parsed.requestId
      };
    }

    return {
      message: 'Authentication Required',
      hint: 'The server requires authentication. Go to Settings to configure your API token.',
      fieldErrors: [],
      code: parsed.code,
      requestId: parsed.requestId
    };
  }

  return parsed;
}

type CreateJobsProps = {
  embedded?: boolean;
};

/**
 * Page control for one list. The runs list and the ungrouped-jobs list have
 * unrelated totals, so each gets its own — one shared control would page a
 * list that has nothing more to show.
 */
const ListPagination = ({
  page,
  total,
  perPage,
  onChange,
}: {
  page: number;
  total: number;
  perPage: number;
  onChange: (next: number) => void;
}) => {
  const lastPage = Math.ceil(total / perPage);
  if (lastPage <= 1) return null;

  return (
    <div className="flex justify-center gap-2">
      <Button
        variant="outline"
        onClick={() => onChange(Math.max(1, page - 1))}
        disabled={page === 1}
      >
        Previous
      </Button>
      <span className="px-4 py-2 text-sm">
        Page {page} of {lastPage}
      </span>
      <Button variant="outline" onClick={() => onChange(page + 1)} disabled={page >= lastPage}>
        Next
      </Button>
    </div>
  );
};

/**
 * The work list, organised by submission run rather than by individual job.
 *
 * Videos submitted together arrive as one batch (spec 008) and are shown as one
 * card with real aggregate progress — submitting 12 videos used to produce 12
 * unrelated rows spread across two pages of this table, which is unusable as
 * soon as a corpus is bigger than a handful of files.
 *
 * Jobs with no batch — submitted by the CLI, or by a viewer/server predating
 * batch tagging — still need somewhere to live, so they get their own section
 * below rather than being hidden.
 */
const CreateJobs = ({ embedded = false }: CreateJobsProps) => {
  // The two lists page independently: they have unrelated totals, and one
  // shared control would page a list that has nothing more to show.
  const [batchPage, setBatchPage] = useState(1);
  const [jobPage, setJobPage] = useState(1);
  const perPage = 10;
  const { toast } = useToast();

  const {
    data: batchesData,
    isLoading: batchesLoading,
    error: batchesError,
    refetch: refetchBatches,
  } = useBatches(batchPage, perPage);

  const {
    data: jobsData,
    isLoading,
    error,
    refetch,
    dataUpdatedAt,
  } = useQuery({
    queryKey: ["jobs", "unbatched", jobPage],
    queryFn: async () => {
      // Only jobs belonging to no batch — everything else is represented by a
      // batch card above, and showing it twice would double-count the work.
      const response = await apiClient.getJobs(jobPage, perPage, { unbatchedOnly: true });

      if (!response) {
        throw new Error('Jobs API returned empty response');
      }

      if (!Array.isArray(response.jobs)) {
        throw new Error('Invalid jobs API response format');
      }

      return response;
    },
    refetchInterval: (query) => {
      // Smart polling: adapt interval based on job activity
      const data = query.state.data;
      if (!data?.jobs) return 5000; // Default: 5s when no data yet

      const hasActiveJobs = data.jobs.some(
        (job: JobResponse) => job.status === 'pending' || job.status === 'running' || job.status === 'cancelling'
      );

      // Fast while work is in flight, backed off once everything is terminal.
      return hasActiveJobs ? 5000 : 30000;
    },
    refetchOnMount: false,
    refetchOnWindowFocus: false,
  });

  // Show error toast when error occurs
  useEffect(() => {
    if (error) {
      const enhancedError = enhanceAuthError(error);
      const errorText = `${enhancedError.message}\n\n${enhancedError.hint}`;

      toast({
        title: enhancedError.message,
        description: enhancedError.hint,
        variant: 'destructive',
        duration: 10000, // Show for 10 seconds (longer than default)
        action: (
          <button
            onClick={() => {
              navigator.clipboard.writeText(errorText);
              toast({
                title: "Copied!",
                description: "Error message copied to clipboard",
                duration: 2000,
              });
            }}
            className="inline-flex h-8 shrink-0 items-center justify-center rounded-md border border-muted/40 bg-transparent px-3 text-sm font-medium hover:bg-destructive/10 focus:outline-none focus:ring-2 focus:ring-ring"
          >
            Copy
          </button>
        ),
      });
    }
  }, [error, toast]);

  const refreshAll = () => {
    refetch();
    refetchBatches();
  };

  if (error) {
    const enhancedError = enhanceAuthError(error);
    const isAuthError = enhancedError.message === 'Authentication Required';

    return (
      <div className="py-8 space-y-4">
        <ErrorDisplay error={enhancedError} />
        <div className="flex gap-2 justify-center">
          {isAuthError && (
            <Link to="/settings">
              <Button variant="default">
                Configure API Token
              </Button>
            </Link>
          )}
          <Button onClick={() => refreshAll()} variant="outline">
            <RefreshCw className="h-4 w-4 mr-2" />
            Retry
          </Button>
        </div>
      </div>
    );
  }

  // Check for data integrity issues
  const hasDataIssue = jobsData && !Array.isArray(jobsData.jobs);

  if (hasDataIssue) {
    return (
      <div className="py-8">
        <Card className="p-6 border-yellow-500">
          <h3 className="text-lg font-semibold text-yellow-700 mb-2">
            ⚠️ Unexpected API Response
          </h3>
          <p className="text-sm text-gray-600 mb-4">
            The jobs API returned data in an unexpected format. This might indicate a server issue.
          </p>
          <details className="text-xs bg-gray-50 p-3 rounded">
            <summary className="cursor-pointer font-semibold">View raw response</summary>
            <pre className="mt-2 overflow-auto">{JSON.stringify(jobsData, null, 2)}</pre>
          </details>
          <div className="mt-4">
            <Button onClick={() => refetch()} variant="outline">
              <RefreshCw className="h-4 w-4 mr-2" />
              Retry
            </Button>
          </div>
        </Card>
      </div>
    );
  }

  // Calculate last checked time
  const lastChecked = dataUpdatedAt ? new Date(dataUpdatedAt) : null;
  const lastCheckedText = lastChecked
    ? `Last checked ${formatDistanceToNow(lastChecked, { addSuffix: true })}`
    : 'Never checked';

  const batches = batchesData?.batches ?? [];
  const hasActiveWork =
    batches.some((b) => b.by_status.pending + b.by_status.running > 0) ||
    jobsData?.jobs?.some(
      (job: JobResponse) =>
        job.status === 'pending' || job.status === 'running' || job.status === 'cancelling'
    );
  const pollingStatus = hasActiveWork
    ? '⚡ Auto-refreshing every 5s'
    : '💤 Auto-refreshing every 30s';

  // Anything the server didn't group into a batch. On a current server every
  // submission from this viewer is batched, so this is legacy and CLI work.
  // A server too old to know about `unbatched_only` ignores it and returns
  // everything, so filter defensively here too rather than showing batched
  // jobs twice.
  const ungroupedJobs = (jobsData?.jobs ?? []).filter(
    (job) => !(job as JobResponse & { batch_id?: string | null }).batch_id
  );

  const totalBatches = batchesData?.total ?? 0;
  const totalUngrouped = jobsData?.total ?? 0;
  const nothingAtAll = !batchesLoading && !isLoading && totalBatches === 0 && ungroupedJobs.length === 0;

  return (
    <div className={embedded ? "space-y-6" : "container mx-auto px-6 py-8 max-w-6xl space-y-6"}>
      {/* Header */}
      {!embedded && (
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-3">
            <img src={vavIcon} alt="VideoAnnotator" className="h-8 w-8" />
            <div>
              <h2 className="text-2xl font-bold text-gray-900">Annotation Runs</h2>
              <p className="text-gray-600">Monitor and manage your annotation work</p>
            </div>
          </div>
          <div className="flex flex-col items-end gap-2">
            <div className="flex gap-2">
              <Button onClick={() => refreshAll()} variant="outline" size="sm">
                <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
                Refresh
              </Button>
              <Link to="/jobs/new">
                <Button>
                  <Play className="h-4 w-4 mr-2" />
                  New Run
                </Button>
              </Link>
            </div>
            <div className="text-xs text-muted-foreground">
              {lastCheckedText} • {pollingStatus}
            </div>
          </div>
        </div>
      )}

      {nothingAtAll && (
        <Card>
          <CardContent className="pt-6 text-center py-12 space-y-4">
            <Layers className="h-10 w-10 mx-auto text-muted-foreground" />
            <p className="text-muted-foreground">No annotation runs yet</p>
            <Link to="/jobs/new">
              <Button>Create your first run</Button>
            </Link>
          </CardContent>
        </Card>
      )}

      {/* Batches — the primary unit of work */}
      {batchesLoading && batches.length === 0 && !nothingAtAll ? (
        <div className="space-y-3">
          <Skeleton className="h-32 w-full" />
          <Skeleton className="h-32 w-full" />
        </div>
      ) : (
        batches.length > 0 && (
          <div className="space-y-3">
            {batches.map((batch) => (
              <BatchCard key={batch.batch_id} batch={batch} />
            ))}
          </div>
        )
      )}

      <ListPagination
        page={batchPage}
        total={totalBatches}
        perPage={perPage}
        onChange={setBatchPage}
      />

      {/* A server too old to know about batches can't group anything; say so
          rather than showing an empty page next to a populated jobs table. */}
      {batchesError && ungroupedJobs.length > 0 && (
        <p className="text-xs text-muted-foreground">
          This server doesn&apos;t report submission batches, so jobs are listed individually below.
          Batch grouping needs VideoAnnotator v1.5.0 or newer.
        </p>
      )}

      {/* Jobs that aren't part of any batch */}
      {(ungroupedJobs.length > 0 || (isLoading && !jobsData)) && (
        <div className="space-y-2">
          {batches.length > 0 && (
            <h3 className="text-sm font-semibold text-muted-foreground pt-2">
              Individual jobs
            </h3>
          )}
          <Card>
            {isLoading && !jobsData ? (
              <div className="flex justify-center py-8">
                <Loader2 className="h-8 w-8 animate-spin" />
              </div>
            ) : (
              <JobsTable
                jobs={ungroupedJobs}
                onChanged={() => refreshAll()}
                emptyState={<p className="text-muted-foreground">No individual jobs</p>}
              />
            )}
          </Card>
          <ListPagination
            page={jobPage}
            total={totalUngrouped}
            perPage={perPage}
            onChange={setJobPage}
          />
        </div>
      )}
    </div>
  );
};

export default CreateJobs;
