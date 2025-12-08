import { useParams, Link, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/api/client";
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
import { canCancelJob } from "@/hooks/useJobCancellation";
import { canDeleteJob } from "@/hooks/useJobDeletion";
import type { JobStatus } from "@/types/api";

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
    refetchInterval: (data) => {
      if (!data) return false;
      const status = data.status;
      // Poll while job is active or cancelling
      return status === "running" || status === "pending" || status === "cancelling" ? 2000 : false;
    },
  });

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

  const getProgressValue = (status: string) => {
    const progressMap = {
      pending: 0,
      running: 50,
      completed: 100,
      failed: 0,
      cancelled: 0,
      cancelling: 25,
    };
    return progressMap[status as keyof typeof progressMap] || 0;
  };

  // Button handlers
  const handleOpenInViewer = () => {
    if (!job) return;

    if (job.status !== 'completed') {
      alert(`Job ${job.id} is not yet completed (status: ${job.status}).\n\nOnly completed jobs can be opened in the viewer.`);
      return;
    }

    // TODO: Implement proper job results loading
    // This requires API endpoints to fetch job output files
    const shouldProceed = confirm(
      `Opening job ${job.id} in viewer...\n\n` +
      `Current limitation: The integration between completed jobs and the annotation viewer is not yet fully implemented.\n\n` +
      `Would you like to:\n` +
      `• Click "OK" to go to the main viewer (you'll need to manually upload result files)\n` +
      `• Click "Cancel" to stay on this page\n\n` +
      `Full integration coming in v0.4.0`
    );

    if (shouldProceed) {
      navigate('/');
    }
  };

  const handleDownloadResults = async () => {
    if (!job) return;

    try {
      // TODO: Implement actual download from API
      // For now, show placeholder
      alert(`Download functionality coming soon for job ${job.id}`);
    } catch (error) {
      console.error("Download failed:", error);
      alert("Download failed. Please try again.");
    }
  };

  const handleRetryJob = () => {
    if (!job) return;

    // Extract video filename with fallback
    const jobData = job as any;
    const videoFilename = jobData.video_filename || jobData.filename || jobData.video_name;

    // Navigate to create new job with pre-filled settings
    navigate('/create/new', {
      state: {
        retryJobId: job.id,
        retryJobConfig: job.config,
        retryJobPipelines: job.selected_pipelines,
        retryJobVideoFilename: videoFilename,
      }
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
      <div className="flex justify-center py-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (error || !job) {
    return (
      <div className="space-y-4">
        <Link to="/create/jobs">
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
  const jobData = job as any;
  let videoFilename = jobData.video_filename || jobData.filename || jobData.video_name;

  // If no direct filename field, extract from video_path
  if (!videoFilename && jobData.video_path) {
    videoFilename = jobData.video_path.split('/').pop() || jobData.video_path;
  }

  videoFilename = videoFilename || "N/A";

  const videoSizeBytes = jobData.video_size_bytes || jobData.file_size_bytes || null;
  const videoDurationSeconds = jobData.video_duration_seconds || jobData.duration_seconds || null;
  const videoPath = jobData.video_path || jobData.file_path || jobData.input_file || "N/A";

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link to="/create/jobs">
            <Button variant="outline" size="sm">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back
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
            <Button onClick={handleRetryJob} variant="outline" size="sm">
              <RotateCcw className="h-4 w-4 mr-2" />
              Retry Job
            </Button>
          )}

          {canDeleteJob(job.status as JobStatus) && (
            <JobDeleteButton
              jobId={job.id}
              jobStatus={job.status as JobStatus}
              size="sm"
              onDeleted={() => navigate('/create/jobs')}
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

      {/* Partial Success Warning */}
      {job.status === 'completed' && job.error_message && (
        <Alert className="bg-orange-50 border-orange-200 text-orange-800">
          <AlertCircle className="h-4 w-4 !text-orange-600" />
          <AlertDescription className="ml-2">
            <span className="font-semibold">Partial Success:</span> {job.error_message}
          </AlertDescription>
        </Alert>
      )}

      {/* Status Card */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Job Status</span>
            <Badge variant="outline" className={getStatusClassName(job.status, job.error_message)}>
              {job.status.toUpperCase()}
              {job.status === 'completed' && job.error_message && (
                <AlertCircle className="ml-1 h-3 w-3 inline" />
              )}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span>Progress</span>
                <span>{getProgressValue(job.status)}%</span>
              </div>
              <Progress value={getProgressValue(job.status)} className="h-2" />
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
                    {job.error_message && (
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

      {/* Results Section (when completed) */}
      {job.status === "completed" && (
        <Card>
          <CardHeader>
            <CardTitle>Results</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <p className="text-muted-foreground">
                Job completed successfully! Results are ready for viewing.
              </p>

              <div className="flex gap-2">
                <Button onClick={handleOpenInViewer}>
                  <Eye className="h-4 w-4 mr-2" />
                  Open in Viewer
                </Button>
                <Button variant="outline" onClick={handleDownloadResults}>
                  <Download className="h-4 w-4 mr-2" />
                  Download Results
                </Button>
                <Button variant="outline" onClick={handleViewRawData}>
                  <ExternalLink className="h-4 w-4 mr-2" />
                  View Raw Data
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Logs Section */}
      <Card>
        <CardHeader>
          <CardTitle>Logs</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="bg-black text-green-400 p-4 rounded-md font-mono text-sm h-64 overflow-y-auto">
            {/* TODO: Implement real-time log streaming */}
            <div className="space-y-1">
              <div>[{new Date().toISOString()}] Job {job.id} created</div>
              <div>[{new Date().toISOString()}] Video uploaded: {videoFilename}</div>
              {job.status !== "pending" && (
                <div>[{new Date().toISOString()}] Processing started...</div>
              )}
              {job.status === "completed" && (
                <div>[{new Date().toISOString()}] Job completed successfully</div>
              )}
              {job.status === "failed" && (
                <div>[{new Date().toISOString()}] Job failed: Check error details</div>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default CreateJobDetail;