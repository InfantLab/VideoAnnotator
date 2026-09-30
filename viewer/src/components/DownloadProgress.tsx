import React from 'react';
import { Progress } from '@/components/ui/progress';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Loader2, Download, FolderOpen, AlertCircle } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

export type DownloadState = 'idle' | 'selecting_dir' | 'downloading' | 'unzipping' | 'ready' | 'error';

interface DownloadProgressProps {
  state: DownloadState;
  progress: number;
  error?: string;
  fileName?: string;
}

export const DownloadProgress: React.FC<DownloadProgressProps> = ({
  state,
  progress,
  error,
  fileName = 'Job Artifacts'
}) => {
  const getStatusText = () => {
    switch (state) {
      case 'selecting_dir':
        return 'Waiting for folder selection...';
      case 'downloading':
        return `Downloading ${fileName}...`;
      case 'unzipping':
        return 'Extracting files...';
      case 'ready':
        return 'Ready!';
      case 'error':
        return 'Download failed';
      default:
        return 'Initializing...';
    }
  };

  const getIcon = () => {
    switch (state) {
      case 'selecting_dir':
        return <FolderOpen className="h-8 w-8 text-blue-500 animate-pulse" />;
      case 'downloading':
        return <Download className="h-8 w-8 text-blue-500 animate-bounce" />;
      case 'unzipping':
        return <Loader2 className="h-8 w-8 text-blue-500 animate-spin" />;
      case 'error':
        return <AlertCircle className="h-8 w-8 text-red-500" />;
      default:
        return <Loader2 className="h-8 w-8 text-gray-400" />;
    }
  };

  if (state === 'idle' || state === 'ready') return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 backdrop-blur-sm">
      <Card className="w-full max-w-md mx-4">
        <CardHeader className="flex flex-row items-center gap-4 space-y-0 pb-2">
          {getIcon()}
          <CardTitle className="text-xl" aria-live="polite">{getStatusText()}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 pt-4">
          {state === 'error' ? (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{error || 'An unknown error occurred.'}</AlertDescription>
            </Alert>
          ) : (
            <div className="space-y-2">
              <Progress value={progress} className="h-2" />
              <div className="flex justify-between text-sm text-muted-foreground" aria-live="polite">
                <span>{Math.round(progress)}%</span>
                {state === 'downloading' && <span>Downloading...</span>}
                {state === 'unzipping' && <span>Processing...</span>}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};
