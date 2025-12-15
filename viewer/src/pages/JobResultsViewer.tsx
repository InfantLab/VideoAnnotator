import React, { useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useZipDownloader } from '@/hooks/useZipDownloader';
import { DownloadProgress } from '@/components/DownloadProgress';
import { VideoAnnotationViewer } from '@/components/VideoAnnotationViewer';
import { ErrorBoundary } from '@/components/ErrorBoundary';
import { Button } from '@/components/ui/button';
import { ArrowLeft, AlertCircle } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

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

  useEffect(() => {
    if (jobId && state === 'idle') {
      startDownload(jobId);
    }
  }, [jobId, state, startDownload]);

  const handleBack = () => {
    navigate('/create/jobs');
  };

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
          Back to Jobs
        </Button>
        
        <Alert variant="destructive" className="mb-6">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Failed to load results</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>

        <div className="flex justify-center">
          <Button onClick={handleRetry}>Retry Download</Button>
        </div>
      </div>
    );
  }

  if (state === 'ready' && videoFile && annotationData) {
    return (
      <div className="h-screen flex flex-col">
        <div className="bg-background border-b p-2 flex items-center">
          <Button variant="ghost" size="sm" onClick={handleBack}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Jobs
          </Button>
          <span className="ml-4 font-semibold">Job Results: {jobId}</span>
        </div>
        <div className="flex-1 overflow-hidden">
          <ErrorBoundary>
             <VideoAnnotationViewer 
               initialVideoFile={videoFile}
               initialAnnotationData={annotationData}
             /> 
          </ErrorBoundary>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen flex items-center justify-center bg-background">
      <DownloadProgress 
        state={state} 
        progress={progress} 
        error={error || undefined} 
      />
    </div>
  );
};

export default JobResultsViewer;
