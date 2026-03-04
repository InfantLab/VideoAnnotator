import React, { useEffect, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useZipDownloader } from '@/hooks/useZipDownloader';
import { DownloadProgress } from '@/components/DownloadProgress';
import { VideoAnnotationViewer } from '@/components/VideoAnnotationViewer';
import { ErrorBoundary } from '@/components/ErrorBoundary';
import { Button } from '@/components/ui/button';
import { ArrowLeft, AlertCircle } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { isDemoJobId, getDemoLabel } from '@/lib/localLibrary/installDemoDataset';

const JobResultsViewer = () => {
  const { jobId } = useParams<{ jobId: string }>();
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

  useEffect(() => {
    if (jobId && state === 'idle') {
      startDownload(jobId);
    }
  }, [jobId, state, startDownload]);

  const handleBack = () => {
    navigate(isDemo ? '/library' : '/jobs');
  };

  const backLabel = isDemo ? 'Back to Library' : 'Back to Jobs';

  const handleRetry = () => {
    reset();
    if (jobId) {
      startDownload(jobId);
    }
  };

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

  if (state === 'ready' && videoFile && annotationData) {
    return (
      <ErrorBoundary>
         <VideoAnnotationViewer
           initialVideoFile={videoFile}
           initialAnnotationData={annotationData}
           backLabel={isDemo ? 'Library' : 'Jobs'}
           backPath={isDemo ? '/library' : '/jobs'}
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

export default JobResultsViewer;
