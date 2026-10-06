import { useState } from "react";
import { useParams, Link, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { apiClient, type JobResponse } from "@/api/client";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { ArrowLeft, ExternalLink, Download, Eye, RotateCcw, AlertCircle } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { ErrorDisplay } from "@/components/ErrorDisplay";
import { parseApiError } from "@/lib/errorHandling";
import vavIcon from "@/assets/v-a-v.icon.png";
import { JobCancelButton } from "@/components/JobCancelButton";
import { JobDeleteButton } from "@/components/JobDeleteButton";
import { ResultsLocation } from "@/components/ResultsLocation";
import { Checkbox } from "@/components/ui/checkbox";
import { canCancelJob } from "@/hooks/useJobCancellation";
import { canDeleteJob } from "@/hooks/useJobDeletion";
import type { JobStatus } from "@/types/api";
import { failedPipelinesOf, isCompletedWithErrors } from "@/lib/jobOutcome";
import { queueLabel } from "@/lib/queuePosition";
import { RunAgainActions } from "@/components/RunAgainActions";
import { CompareWith } from "@/components/CompareWith";
import { settingsOf, wizardState } from "@/lib/wizardStart";

const CreateJobDetail = () => {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();

  const {
    data: job,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => {
      if (!jobId) throw new Error("Job ID is required");
      return apiClient.getJob(jobId);
    },
    enabled: !!jobId,
    // React Query 5 passes the query, not its data: reading `.status` off the
    // argument meant this page never polled, so a running job looked frozen.
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      // Poll while job is active or cancelling
      return status === "running" || status === "pending" || status === "cancelling" ? 2000 : false;
    },
  });

  // Which pipelines produced nothing, and why: only worth asking once the job
  // has finished with an error message, or failed outright.
  const withErrors = !!job && isCompletedWithErrors(job);
  const failed = job?.status === "failed";
  const { data: results } = useQuery({
    queryKey: ["job-results", jobId],
    queryFn: () => apiClient.getJobResults(jobId!),
    enabled: !!jobId && (withErrors || failed),
    staleTime: 60_000,
  });
  const failedPipelines = Object.entries(failedPipelinesOf(results));
  const queued = job ? queueLabel(job as JobResponse & Record<string, unknown>) : null;

  const getStatusClassName = (status: string, errorMessage?: string | null) => {
    const statusMap = {
      pending: "bg-yellow-100 text-yellow-800 border-yellow-200",
      running: "bg-blue-100 text-blue-800 border-blue-200",
      completed: "bg-green-100 text-green-800 border-green-200",
      failed: "bg-red-100 text-red-800 border-red-200",
      cancelled: "bg-gray-100 text-gray-800 border-gray-200",
      cancelling: "bg-orange-100 text-orange-800 border-orange-200",
    };
    
    if (status === 'completed' && errorMessage) {
      return "bg-orange-100 text-orange-800 border-orange-200";
    }

    return statusMap[status as keyof typeof statusMap] || "bg-gray-100 text-gray-800 border-gray-200";
  };

  /**
   * Real progress, reported by the server as completed/total selected pipelines
   * (spec 006). This used to be a fixed status-to-number map — every running
   * job showed exactly 50% regardless of how much work was actually done.
   *
   * Only the terminal states are still derived: a job that finished is 100%
   * whatever its last reported figure was, and one that failed or was cancelled
   * keeps the progress it had reached, which is more informative than zero.
   */
  const getProgressValue = (job: { status: string; progress_percentage?: number }) => {
    const reported =
      typeof job.progress_percentage === 'number' && Number.isFinite(job.progress_percentage)
        ? job.progress_percentage
        : 0;
    if (job.status === 'completed') return 100;
    return Math.max(0, Math.min(100, Math.round(reported)));
  };

  // Button handlers
  const handleOpenInViewer = () => {
    if (!job) return;

    if (job.status !== 'completed') {
      alert(`Job ${job.id} is not yet completed (status: ${job.status}).\n\nOnly completed jobs can be opened in the viewer.`);
      return;
    }

    navigate(`/view/${job.id}`);
  };

  // The job's results zip; the video only when asked for (spec 022).
  const [includeVideo, setIncludeVideo] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const handleDownloadResults = async () => {
    if (!job) return;
    setIsDownloading(true);
    setDownloadError(null);
    try {
      const response = await apiClient.getJobArtifacts(job.id, { includeVideo });
      if (!response.ok) throw new Error(`the server answered ${response.status}`);
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `job_${job.id}_artifacts.zip`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 10_000);
    } catch (error) {
      console.error("Download failed:", error);
      setDownloadError(error instanceof Error ? error.message : String(error));
    } finally {
      setIsDownloading(false);
    }
  };

  // The video's name, to label this job in the wizard and in actions.
  const jobLabel = (() => {
    if (!job) return '';
    const record = job as JobResponse & Record<string, unknown>;
    const name = [record.video_filename, record.filename, record.video_name].find((v) => typeof v === 'string');
    return (name as string | undefined) ?? `job ${job.id.slice(0, 8)}`;
  })();

  /** "Fix settings and run again": the wizard with this job's video and settings (spec 019). */
  const handleEditAndRunAgain = () => {
    if (!job) return;
    navigate('/jobs/new', {
      state: wizardState({ mode: 'rerun', jobId: job.id, label: jobLabel, ...settingsOf(job) }),
    });
  };

  const handleViewRawData = () => {
    if (!job) return;

    // TODO: Navigate to raw data view or open in new tab
    // For now, show job data in new window
    const dataWindow = window.open("", "_blank");
    if (dataWindow) {
      dataWindow.document.write(`
        <html>
          <head><title>Job ${job.id} - Raw Data</title></head>
          <body>
            <h1>Job ${job.id} Raw Data</h1>
            <pre>${JSON.stringify(job, null, 2)}</pre>
          </body>
        </html>
      `);
      dataWindow.document.close();
    }
  };

  if (isLoading) {
    return (
      <div className="container mx-auto px-6 py-8 max-w-5xl">
        <div className="flex justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        </div>
      </div>
    );
  }

  if (error || !job) {
    return (
      <div className="container mx-auto px-6 py-8 max-w-5xl space-y-4">
        <Link to="/jobs">
          <Button variant="outline">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Jobs
          </Button>
        </Link>
        <ErrorDisplay error={parseApiError(error || 'Job not found')} />
      </div>
    );
  }

  // Defensive field access - server may use different field names
  const jobData = job as JobResponse & Record<string, unknown>;
  const getString = (value: unknown): string | undefined => (typeof value === 'string' ? value : undefined);
  const getNumber = (value: unknown): number | null =>
    typeof value === 'number' && Number.isFinite(value) ? value : null;

  let videoFilename = getString(jobData.video_filename) ?? getString(jobData.filename) ?? getString(jobData.video_name);

  // If no direct filename field, extract from video_path
  const videoPathMaybe = getString(jobData.video_path);
  if (!videoFilename && videoPathMaybe) {
    videoFilename = videoPathMaybe.split('/').pop() || videoPathMaybe;
  }

  videoFilename = videoFilename || "N/A";

  const videoSizeBytes = getNumber(jobData.video_size_bytes) ?? getNumber(jobData.file_size_bytes);
  const videoDurationSeconds = getNumber(jobData.video_duration_seconds) ?? getNumber(jobData.duration_seconds);
  const videoPath =
    getString(jobData.video_path) ??
    getString(jobData.file_path) ??
    getString(jobData.input_file) ??
    "N/A";

  return (
    <div className="container mx-auto px-6 py-8 max-w-5xl space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link to="/jobs">
            <Button variant="outline" size="sm">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to Jobs
            </Button>
          </Link>
          <div className="flex items-center gap-3">
            <img src={vavIcon} alt="VideoAnnotator" className="h-8 w-8" />
            <div>
              <h2 className="text-2xl font-bold">Job Details</h2>
              <p className="text-muted-foreground font-mono text-sm">{job.id}</p>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {canCancelJob(job.status as JobStatus) && (
            <JobCancelButton
              jobId={job.id}
              jobStatus={job.status as JobStatus}
              size="sm"
            />
          )}

          {job.status === "failed" && (
            <Button onClick={handleEditAndRunAgain} variant="outline" size="sm">
              <RotateCcw className="h-4 w-4 mr-2" />
              Fix settings and run again
            </Button>
          )}

          {canDeleteJob(job.status as JobStatus) && (
            <JobDeleteButton
              jobId={job.id}
              jobStatus={job.status as JobStatus}
              size="sm"
              onDeleted={() => navigate('/jobs')}
              resultsFolder={job.results_folder?.display_path}
            />
          )}

          {job.status === "completed" && (
            <Button onClick={handleOpenInViewer}>
              <Eye className="h-4 w-4 mr-2" />
              Open in Viewer
            </Button>
          )}
        </div>
      </div>

      {/* Completed, but some pipelines produced nothing */}
      {withErrors && (
        <Alert className="bg-orange-50 border-orange-200 text-orange-800">
          <AlertCircle className="h-4 w-4 !text-orange-600" />
          <AlertDescription className="ml-2 space-y-1">
            <p>
              <span className="font-semibold">Completed with errors:</span>{" "}
              {failedPipelines.length > 0
                ? "these pipelines produced no results."
                : job.error_message}
            </p>
            {failedPipelines.map(([name, reason]) => (
              <p key={name} className="text-sm">
                <span className="font-mono font-medium">{name}</span>: {reason}
              </p>
            ))}
            <Button variant="link" className="h-auto p-0 text-orange-800" onClick={handleEditAndRunAgain}>
              Fix settings and run again
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {/* Links between a job and its reruns, whatever their status */}
      {(job.rerun_of || (job.reruns?.length ?? 0) > 0) && (
        <div className="text-sm space-y-1 rounded-md border p-3">
          {job.rerun_of && (
            <p>
              Runs again <Link className="underline" to={`/jobs/${job.rerun_of}`}>job {job.rerun_of.slice(0, 8)}</Link>.
            </p>
          )}
          {(job.reruns?.length ?? 0) > 0 && (
            <p>
              Run again as{' '}
              {job.reruns!.map((id, i) => (
                <span key={id}>
                  {i > 0 && ', '}
                  <Link className="underline" to={`/jobs/${id}`}>job {id.slice(0, 8)}</Link>
                </span>
              ))}
              .
            </p>
          )}
        </div>
      )}

      {/* Run it again (spec 019): where the decision is made, not in a menu */}
      {['completed', 'failed', 'cancelled'].includes(job.status) && (
        <Card>
          <CardHeader>
            <CardTitle>Run it again</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <RunAgainActions target={{ kind: 'job', id: job.id, label: jobLabel, settings: settingsOf(job) }} />
            <CompareWith job={job} />
          </CardContent>
        </Card>
      )}

      {/* Status Card */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Job Status</span>
            <span className="flex items-center gap-2">
              {queued && <span className="text-sm font-normal text-muted-foreground">{queued}</span>}
              <Badge variant="outline" className={getStatusClassName(job.status, job.error_message)}>
                {job.status.toUpperCase()}
                {job.status === 'completed' && job.error_message && (
                  <AlertCircle className="ml-1 h-3 w-3 inline" />
                )}
              </Badge>
            </span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span>Progress</span>
                <span>{getProgressValue(job)}%</span>
              </div>
              <Progress value={getProgressValue(job)} className="h-2" />
            </div>

            {job.status === "running" && (
              <Alert>
                <AlertDescription>
                  Job is currently running. This page will update automatically.
                </AlertDescription>
              </Alert>
            )}

            {job.status === "failed" && (
              <Alert variant="destructive">
                <AlertDescription>
                  <div className="space-y-2">
                    <p className="font-semibold">Job failed during processing</p>
                    {failedPipelines.length > 0
                      ? failedPipelines.map(([name, reason]) => (
                          <p key={name} className="text-sm">
                            <span className="font-mono font-medium">{name}</span>: {reason}
                          </p>
                        ))
                      : job.error_message && (
                          <p className="text-sm">
                            <span className="font-medium">Error:</span> {job.error_message}
                          </p>
                        )}
                  </div>
                </AlertDescription>
              </Alert>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Video Information */}
      <Card>
        <CardHeader>
          <CardTitle>Video Information</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-sm font-medium text-muted-foreground">Filename</label>
              <p className="mt-1">{videoFilename}</p>
            </div>
            <div>
              <label className="text-sm font-medium text-muted-foreground">File Size</label>
              <p className="mt-1">
                {videoSizeBytes
                  ? `${(videoSizeBytes / (1024 * 1024)).toFixed(1)} MB`
                  : "N/A"
                }
              </p>
            </div>
            <div>
              <label className="text-sm font-medium text-muted-foreground">Duration</label>
              <p className="mt-1">
                {videoDurationSeconds
                  ? `${Math.floor(videoDurationSeconds / 60)}:${(videoDurationSeconds % 60).toFixed(0).padStart(2, '0')}`
                  : "N/A"
                }
              </p>
            </div>
            <div>
              <label className="text-sm font-medium text-muted-foreground">Path</label>
              <p className="mt-1 font-mono text-sm break-all">
                {videoPath}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Pipeline Configuration */}
      <Card>
        <CardHeader>
          <CardTitle>Pipeline Configuration</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-muted-foreground">Selected Pipelines</label>
              <div className="mt-2 flex flex-wrap gap-2">
                {job.selected_pipelines?.map((pipeline) => (
                  <Badge key={pipeline} variant="outline">
                    {pipeline}
                  </Badge>
                )) || <span className="text-muted-foreground">No pipelines selected</span>}
              </div>
            </div>

            {job.config && (
              <div>
                <label className="text-sm font-medium text-muted-foreground">Configuration</label>
                <pre className="mt-2 p-3 bg-muted rounded-md text-sm overflow-x-auto text-foreground">
                  {JSON.stringify(job.config, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {job.video_available === false && (
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Video not found at <span className="font-mono text-xs break-all">{job.video_display_path ?? job.video_path}</span> (
            {job.video_unavailable_reason ?? 'moved or deleted since the job ran'}). Its results are all still
            here; only playback needs the video.
          </AlertDescription>
        </Alert>
      )}

      {job.results_folder?.exists === false && (
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Results not found at{' '}
            <span className="font-mono text-xs break-all">{job.results_folder.display_path}</span>. The folder
            was moved, renamed or deleted outside VideoAnnotator; put it back to see this job&apos;s files.
          </AlertDescription>
        </Alert>
      )}

      <ResultsLocation folder={job.results_folder} label="This video's results" />

      {/* Results Section (when completed) */}
      {job.status === "completed" && (
        <Card>
          <CardHeader>
            <CardTitle>Results</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <p className="text-muted-foreground">
                {withErrors
                  ? "Results from the pipelines that succeeded are ready for viewing."
                  : "Job completed successfully! Results are ready for viewing."}
              </p>

              <div className="flex gap-2">
                <Button onClick={handleOpenInViewer}>
                  <Eye className="h-4 w-4 mr-2" />
                  Open in Viewer
                </Button>
                <Button variant="outline" onClick={handleDownloadResults} disabled={isDownloading}>
                  <Download className="h-4 w-4 mr-2" />
                  {isDownloading ? "Preparing zip…" : "Download Results"}
                </Button>
                <Button variant="outline" onClick={handleViewRawData}>
                  <ExternalLink className="h-4 w-4 mr-2" />
                  View Raw Data
                </Button>
              </div>
              {downloadError && (
                <p className="text-sm text-destructive">Couldn&apos;t download the results: {downloadError}</p>
              )}
              <label className="flex items-center gap-2 text-xs text-muted-foreground">
                <Checkbox checked={includeVideo} onCheckedChange={(checked) => setIncludeVideo(checked === true)} />
                Include the video in the download (it holds each pipeline&apos;s output either way)
              </label>
            </div>
          </CardContent>
        </Card>
      )}

    </div>
  );
};

export default CreateJobDetail;