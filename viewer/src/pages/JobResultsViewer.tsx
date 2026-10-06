import React, { useEffect, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useZipDownloader } from '@/hooks/useZipDownloader';
import { DownloadProgress } from '@/components/DownloadProgress';
import { VideoAnnotationViewer } from '@/components/VideoAnnotationViewer';
import { ErrorBoundary } from '@/components/ErrorBoundary';
import { Button } from '@/components/ui/button';
import { ArrowLeft, AlertCircle, FolderOpen } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { isDemoJobId, getDemoLabel } from '@/lib/localLibrary/installDemoDataset';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/api/client';
import { failedPipelinesOf } from '@/lib/jobOutcome';
import { BatchNavigation } from '@/components/BatchNavigation';

const JobResultsViewerForJob = ({ jobId }: { jobId: string | undefined }) => {
  const navigate = useNavigate();
  const {
    state,
    progress,
    error,
    videoFile,
    annotationData,
    startDownload,
    reset
  } = useZipDownloader();

  const isDemo = useMemo(() => jobId ? isDemoJobId(jobId) : false, [jobId]);
  const demoLabel = useMemo(() => jobId ? getDemoLabel(jobId) : null, [jobId]);

  // Pipelines that ran but failed, so their tracks say why instead of "(No
  // data)". Best effort: older servers and demo jobs have no results endpoint.
  const { data: results } = useQuery({
    queryKey: ['job-results', jobId],
    queryFn: () => apiClient.getJobResults(jobId!),
    enabled: !!jobId && !isDemo,
    retry: false,
    staleTime: 60_000,
  });
  const failedPipelines = useMemo(() => failedPipelinesOf(results), [results]);

  const { data: job } = useQuery({
    queryKey: ['job', jobId],
    queryFn: () => apiClient.getJob(jobId!),
    enabled: !!jobId && !isDemo,
    retry: false,
    staleTime: 60_000,
  });
  const batchId: string | undefined = job?.batch_id ?? undefined;

  useEffect(() => {
    if (jobId && state === 'idle') {
      startDownload(jobId);
    }
  }, [jobId, state, startDownload]);

  const backPath = isDemo ? '/results' : batchId ? `/batches/${batchId}` : '/jobs';
  const backName = isDemo ? 'Results' : batchId ? 'Batch' : 'Jobs';

  const handleBack = () => {
    navigate(backPath);
  };

  const backLabel = `Back to ${backName}`;

  const handleRetry = () => {
    reset();
    if (jobId) {
      startDownload(jobId);
    }
  };

  if (state === 'needs_folder' && jobId) {
    return (
      <div className="container mx-auto p-8 max-w-2xl">
        <Button variant="ghost" onClick={handleBack} className="mb-4">
          <ArrowLeft className="mr-2 h-4 w-4" />
          {backLabel}
        </Button>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FolderOpen className="h-5 w-5" />
              Choose a folder for your results
            </CardTitle>
            <CardDescription>
              The viewer keeps each job&apos;s results (the video and its annotations) in a folder on
              this computer, so opening the job again doesn&apos;t download it again, and the files
              stay yours. Choose a folder once; the browser remembers it.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-wrap gap-3">
            <Button onClick={() => startDownload(jobId, 'pick')}>
              <FolderOpen className="mr-2 h-4 w-4" />
              Choose folder
            </Button>
            <Button variant="outline" onClick={() => startDownload(jobId, 'skip')}>
              View without saving
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (state === 'error') {
    return (
      <div className="container mx-auto p-8 max-w-2xl">
        <Button variant="ghost" onClick={handleBack} className="mb-4">
          <ArrowLeft className="mr-2 h-4 w-4" />
          {backLabel}
        </Button>

        <Alert variant="destructive" className="mb-6">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Failed to load results</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>

        <div className="flex justify-center">
          <Button onClick={handleRetry}>Retry</Button>
        </div>
      </div>
    );
  }

  const missingVideoMessage =
    job && job.video_available === false && !videoFile
      ? `Video not found at ${job.video_path ?? 'its original location'} (moved or deleted since the job ran).`
      : undefined;

  if (state === 'ready' && annotationData && (videoFile || missingVideoMessage)) {
    return (
      <ErrorBoundary>
         <VideoAnnotationViewer
           initialVideoFile={videoFile}
           initialAnnotationData={annotationData}
           backLabel={backName}
           backPath={backPath}
           failedPipelines={failedPipelines}
           headerNav={batchId && jobId ? <BatchNavigation jobId={jobId} batchId={batchId} /> : undefined}
           missingVideoMessage={missingVideoMessage}
         />
      </ErrorBoundary>
    );
  }

  return (
    <div className="h-screen flex flex-col items-center justify-center bg-background gap-4">
      <DownloadProgress
        state={state}
        progress={progress}
        error={error || undefined}
      />
      {isDemo && demoLabel && (
        <p className="text-sm text-muted-foreground">Loading demo: {demoLabel}</p>
      )}
    </div>
  );
};

// Keyed by job: moving to the next video in a batch changes only the URL, and
// the downloader's state would otherwise keep showing the previous video.
const JobResultsViewer = () => {
  const { jobId } = useParams<{ jobId: string }>();
  return <JobResultsViewerForJob key={jobId} jobId={jobId} />;
};

export default JobResultsViewer;
