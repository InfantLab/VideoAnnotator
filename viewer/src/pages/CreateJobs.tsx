import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { formatDistanceToNow } from "date-fns";
import { apiClient } from "@/api/client";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Loader2, RefreshCw, Eye, Play, Settings, AlertCircle, RotateCcw } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useToast } from "@/hooks/use-toast";
import vavIcon from "@/assets/v-a-v.icon.png";
import { JobCancelButton } from "@/components/JobCancelButton";
import { JobDeleteButton } from "@/components/JobDeleteButton";
import { canCancelJob } from "@/hooks/useJobCancellation";
import { canDeleteJob } from "@/hooks/useJobDeletion";
import { ErrorDisplay } from "@/components/ErrorDisplay";
import { parseApiError } from "@/lib/errorHandling";
import type { JobStatus } from "@/types/api";

/**
 * Enhances authentication error messages with actionable guidance
 * CANONICAL error messages - used consistently across the app
 */
function enhanceAuthError(error: any) {
  const errorMessage = error?.message || String(error);
  const currentToken = localStorage.getItem('videoannotator_api_token') || '';

  // Check if it's an authentication error
  if (
    errorMessage.includes('API key required') ||
    errorMessage.includes('AUTH_REQUIRED') ||
    errorMessage.includes('401') ||
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
        code: error?.code,
        requestId: error?.requestId
      };
    }

    return {
      message: 'Authentication Required',
      hint: 'The server requires authentication. Go to Settings to configure your API token.',
      fieldErrors: [],
      code: error?.code,
      requestId: error?.requestId
    };
  }

  return parseApiError(error);
}

const CreateJobs = () => {
  const navigate = useNavigate();
  const [page, setPage] = useState(1);
  const perPage = 10;
  const { toast } = useToast();

  const {
    data: jobsData,
    isLoading,
    error,
    refetch,
    dataUpdatedAt,
  } = useQuery({
    queryKey: ["jobs", page],
    queryFn: async () => {
      console.log('🔍 Fetching jobs...', { page, perPage });
      const response = await apiClient.getJobs(page, perPage);
      console.log('📦 Jobs API response:', response);

      // Validate response structure
      if (!response) {
        console.error('❌ Jobs API returned null/undefined');
        throw new Error('Jobs API returned empty response');
      }

      if (!Array.isArray(response.jobs)) {
        console.error('❌ Jobs API response missing jobs array:', response);
        throw new Error('Invalid jobs API response format');
      }

      console.log(`✅ Found ${response.jobs.length} jobs (total: ${response.total})`);
      return response;
    },
    refetchInterval: (query) => {
      // Smart polling: adapt interval based on job activity
      const data = query.state.data;
      if (!data?.jobs) return 5000; // Default: 5s when no data yet

      const hasActiveJobs = data.jobs.some(
        (job: any) => job.status === 'pending' || job.status === 'running' || job.status === 'cancelling'
      );

      if (hasActiveJobs) {
        // Fast polling when jobs are active
        return 5000; // 5 seconds
      }

      // Exponential backoff when idle: 30s initially, then 60s
      return 30000; // 30 seconds when all jobs complete/cancelled/failed
    },
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

  const handleRetryJob = (job: any) => {
    navigate('/create/new', {
      state: {
        retryJobId: job.id,
        retryJobConfig: job.config,
        retryJobPipelines: job.selected_pipelines,
        retryJobVideoFilename: job.video_filename,
      }
    });
  };

  const getStatusBadge = (status: string, errorMessage?: string | null) => {
    const statusMap = {
      pending: { variant: "secondary" as const, color: "text-yellow-600" },
      running: { variant: "default" as const, color: "text-blue-600" },
      completed: { variant: "default" as const, color: "text-green-600" },
      failed: { variant: "destructive" as const, color: "text-red-600" },
      cancelled: { variant: "secondary" as const, color: "text-gray-600" },
      cancelling: { variant: "secondary" as const, color: "text-orange-600" },
    };

    const config = statusMap[status as keyof typeof statusMap] || statusMap.pending;

    const badge = (
      <Badge variant={config.variant} className={config.color}>
        {status.toUpperCase()}
        {status === 'failed' && errorMessage && (
          <AlertCircle className="ml-1 h-3 w-3 inline" />
        )}
      </Badge>
    );

    // Show tooltip with error message for failed jobs
    if (status === 'failed' && errorMessage) {
      return (
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              {badge}
            </TooltipTrigger>
            <TooltipContent className="max-w-xs">
              <p className="font-semibold">Error:</p>
              <p>{errorMessage}</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      );
    }

    return badge;
  };

  const formatDuration = (seconds: number | null) => {
    if (!seconds) return "N/A";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const formatFileSize = (bytes: number | null) => {
    if (!bytes) return "N/A";
    const mb = bytes / (1024 * 1024);
    return `${mb.toFixed(1)} MB`;
  };

  if (error) {
    const enhancedError = enhanceAuthError(error);
    const isAuthError = enhancedError.message === 'Authentication Required';

    return (
      <div className="py-8 space-y-4">
        <ErrorDisplay error={enhancedError} />
        <div className="flex gap-2 justify-center">
          {isAuthError && (
            <Link to="/create/settings">
              <Button variant="default">
                <Settings className="h-4 w-4 mr-2" />
                Configure API Token
              </Button>
            </Link>
          )}
          <Button onClick={() => refetch()} variant="outline">
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

  // Determine polling status
  const hasActiveJobs = jobsData?.jobs?.some(
    (job: any) => job.status === 'pending' || job.status === 'running' || job.status === 'cancelling'
  );
  const pollingStatus = hasActiveJobs
    ? '⚡ Auto-refreshing every 5s'
    : '💤 Auto-refreshing every 30s';

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-3">
          <img src={vavIcon} alt="VideoAnnotator" className="h-8 w-8" />
          <div>
            <h2 className="text-2xl font-bold">Annotation Jobs</h2>
            <p className="text-gray-600">Monitor and manage your annotation jobs</p>
          </div>
        </div>
        <div className="flex flex-col items-end gap-2">
          <div className="flex gap-2">
            <Button onClick={() => refetch()} variant="outline" size="sm">
              <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
            <Link to="/create/new">
              <Button>
                <Play className="h-4 w-4 mr-2" />
                New Job
              </Button>
            </Link>
          </div>
          <div className="text-xs text-muted-foreground">
            {lastCheckedText} • {pollingStatus}
          </div>
        </div>
      </div>

      {/* Jobs Table */}
      {isLoading && !jobsData ? (
        <div className="flex justify-center py-8">
          <Loader2 className="h-8 w-8 animate-spin" />
        </div>
      ) : (
        <Card>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Job ID</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Video</TableHead>
                <TableHead>Duration</TableHead>
                <TableHead>Size</TableHead>
                <TableHead>Pipelines</TableHead>
                <TableHead>Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {jobsData?.jobs?.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={7} className="text-center py-8">
                    <p className="text-gray-500 mb-4">
                      {jobsData?.total > 0
                        ? `No jobs on this page (${jobsData.total} total jobs found)`
                        : "No jobs found"}
                    </p>
                    <div className="flex gap-2 justify-center">
                      {jobsData?.total > 0 && page > 1 ? (
                        <Button onClick={() => setPage(1)} variant="outline">
                          Go to First Page
                        </Button>
                      ) : (
                        <Link to="/create/new">
                          <Button>Create your first job</Button>
                        </Link>
                      )}
                    </div>
                  </TableCell>
                </TableRow>
              ) : (
                jobsData?.jobs?.map((job) => (
                  <TableRow key={job.id}>
                    <TableCell className="font-mono text-sm">
                      {job.id.slice(0, 8)}...
                    </TableCell>
                    <TableCell>
                      {getStatusBadge(job.status, job.error_message)}
                    </TableCell>
                    <TableCell className="max-w-[200px] truncate">
                      {job.video_filename || "N/A"}
                    </TableCell>
                    <TableCell>
                      {formatDuration(job.video_duration_seconds)}
                    </TableCell>
                    <TableCell>
                      {formatFileSize(job.video_size_bytes)}
                    </TableCell>
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
                            onClick={() => handleRetryJob(job)}
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
                            onDeleted={() => refetch()}
                          />
                        )}
                        <Link to={`/create/jobs/${job.id}`}>
                          <Button variant="outline" size="sm">
                            <Eye className="h-4 w-4 mr-1" />
                            View
                          </Button>
                        </Link>
                      </div>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </Card>
      )}

      {/* Pagination */}
      {jobsData && jobsData.total > perPage && (
        <div className="flex justify-center gap-2">
          <Button
            variant="outline"
            onClick={() => setPage(p => Math.max(1, p - 1))}
            disabled={page === 1}
          >
            Previous
          </Button>
          <span className="px-4 py-2 text-sm">
            Page {page} of {Math.ceil(jobsData.total / perPage)}
          </span>
          <Button
            variant="outline"
            onClick={() => setPage(p => p + 1)}
            disabled={page >= Math.ceil(jobsData.total / perPage)}
          >
            Next
          </Button>
        </div>
      )}
    </div>
  );
};

export default CreateJobs;