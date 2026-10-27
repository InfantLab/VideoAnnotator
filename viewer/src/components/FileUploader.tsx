import React, { useState, useCallback, useRef } from 'react';
import { Upload, Video, FileText, Music, AlertCircle, CheckCircle2, Loader2, X } from 'lucide-react';
import { kindColour, kindOfPipeline } from '@/lib/resultKinds';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { StandardAnnotationData } from '@/types/annotations';
import { useToast } from '@/hooks/use-toast';
import { parseApiError } from '@/lib/errorHandling';
import { showErrorToast, showValidationErrorToast } from '@/lib/toastHelpers';
import type { ParsedError } from '@/types/api';
import {
  describeFileType,
  detectFileType,
  isAnnotationType,
  validateFileSet,
  validateFileSize,
  type DetectedFile,
  type DetectedFileType
} from '@/lib/fileDetection';
import {
  mergeAnnotationData
} from '@/lib/parsers/merger';

export interface LoadedFiles {
  video: File | null;
  /** Every result file, as given (the Video page builds its table from their text). */
  results: File[];
  data: StandardAnnotationData;
}

interface FileUploaderProps {
  onVideoLoad: (file: File) => void;
  onAnnotationLoad: (data: StandardAnnotationData) => void;
  /** Called with everything loaded, once the files are read (spec 028 standalone mode). */
  onFilesLoaded?: (loaded: LoadedFiles) => void;
}

interface FileStatus {
  file: File;
  detected: DetectedFile;
  status: 'pending' | 'processing' | 'success' | 'error';
  error?: string;
}

const FileTypeIcon = ({ type }: { type: DetectedFileType }) => {
  const kind = kindOfPipeline(type === 'openface3_faces' ? 'face_openface3_embedding' : type);
  const Icon = kind?.icon ?? (type === 'video' ? Video : type === 'audio' ? Music : type === 'unknown' ? AlertCircle : FileText);
  return <Icon className="w-4 h-4" style={kind ? { color: kindColour(kind) } : undefined} aria-hidden="true" />;
};

const FileTypeLabel = ({ type, confidence }: { type: DetectedFileType; confidence: number }) => (
  <span className={confidence < 0.5 ? 'text-amber-700 dark:text-amber-400' : undefined}>
    {describeFileType(type)}
    {confidence < 0.8 && confidence >= 0.5 && ' (probably)'}
  </span>
);

export const FileUploader = ({ onVideoLoad, onAnnotationLoad, onFilesLoaded }: FileUploaderProps) => {
  const [fileStatuses, setFileStatuses] = useState<FileStatus[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [processingStage, setProcessingStage] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const detectFiles = useCallback(async (files: File[]) => {
    if (files.length === 0) return;

    setProcessingStage('Detecting file types...');
    setProcessingProgress(10);

    const detectedFiles: FileStatus[] = [];
    for (const file of files) {
      const detected = await detectFileType(file);
      const size = validateFileSize(file, detected.type);
      detectedFiles.push(
        size.valid
          ? { file, detected, status: 'pending' }
          : { file, detected, status: 'error', error: size.error }
      );
    }

    // Add to existing files instead of replacing
    setFileStatuses(prev => [...prev, ...detectedFiles]);
    setProcessingProgress(30);
  }, []);

  const handleFilesSelected = useCallback(async (files: File[]) => {
    if (files.length === 0) return;
    await detectFiles(files);
  }, [detectFiles]);

  const processFiles = useCallback(async () => {
    if (fileStatuses.length === 0) return;

    try {
      setIsProcessing(true);
      setProcessingStage('Validating file set...');
      setProcessingProgress(10);

      // Validate file set
      const validation = validateFileSet(fileStatuses.map(fs => fs.detected));

      if (!validation.valid) {
        toast({
          title: "Missing required files",
          description: validation.missing.join(', '),
          variant: "destructive",
        });
        setIsProcessing(false);
        return;
      }

      // Show warnings if any
      if (validation.warnings.length > 0) {
        toast({
          title: "File set warnings",
          description: validation.warnings.join(', '),
        });
      }

      setProcessingStage('Processing files...');
      setProcessingProgress(50);

      const detectedFiles = fileStatuses.map(fs => fs.detected);

      // Parse and merge data using existing merger
      const result = await mergeAnnotationData(detectedFiles, (stage, progress, total) => {
        setProcessingStage(stage);
        setProcessingProgress(50 + (progress / total) * 40);
      });

      // Update file statuses to success
      setFileStatuses(prev => prev.map(status => ({
        ...status,
        status: 'success'
      })));

      setProcessingProgress(100);

      // Show results
      if (result.metadata.warnings.length > 0) {
        toast({
          title: "Files loaded with warnings",
          description: `${result.metadata.warnings.length} warnings occurred during parsing.`,
          variant: "destructive",
        });
      } else {
        toast({
          title: "Files loaded successfully",
          description: `Processed ${result.metadata.filesProcessed} files.`,
        });
      }

      // Find video file and pass data to parent
      const videoFile = fileStatuses.find(f => f.detected.type === 'video')?.file;
      if (videoFile) {
        onVideoLoad(videoFile);
      }

      onAnnotationLoad(result.data);
      onFilesLoaded?.({
        video: videoFile ?? null,
        results: fileStatuses.filter((f) => f.detected.type !== 'video' && f.detected.type !== 'audio').map((f) => f.file),
        data: result.data,
      });

    } catch (error) {
      showErrorToast(toast, parseApiError(error));

      // Update file statuses to show errors
      const parsedError = parseApiError(error);
      setFileStatuses(prev => prev.map(status => ({
        ...status,
        status: 'error',
        error: parsedError.message
      })));
    } finally {
      setIsProcessing(false);
      setProcessingStage('');
      setProcessingProgress(0);
    }
  }, [fileStatuses, onVideoLoad, onAnnotationLoad, onFilesLoaded, toast]);

  const handleFileInputChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files) {
      handleFilesSelected(Array.from(files));
    }
  }, [handleFilesSelected]);

  const handleDrop = useCallback((event: React.DragEvent<HTMLElement>) => {
    event.preventDefault();
    const files = event.dataTransfer.files;
    if (files) {
      handleFilesSelected(Array.from(files));
    }
  }, [handleFilesSelected]);

  const handleDragOver = useCallback((event: React.DragEvent<HTMLElement>) => {
    event.preventDefault();
  }, []);

  const removeFile = useCallback((index: number) => {
    setFileStatuses(prev => prev.filter((_, i) => i !== index));
  }, []);

  const clearAllFiles = useCallback(() => {
    setFileStatuses([]);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, []);

  const hasFiles = fileStatuses.length > 0;
  const hasVideoFile = fileStatuses.some(f => f.detected.type === 'video');
  const hasPipelineData = fileStatuses.some(f => isAnnotationType(f.detected.type));
  const canProcess = hasVideoFile && hasPipelineData && !isProcessing;

  return (
    <Card className="p-6">
      <div className="space-y-6">
        {/* A real button, so the keyboard reaches it too (UAT #60); dropping still works. */}
        <button
          type="button"
          className={`w-full rounded-lg border-2 border-dashed p-8 text-center transition-colors ${isProcessing
            ? 'border-muted bg-muted/10'
            : 'border-border hover:border-primary focus-visible:border-primary focus-visible:outline-none'
            }`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onClick={() => !isProcessing && fileInputRef.current?.click()}
          disabled={isProcessing}
        >
          {isProcessing ? (
            <span className="block space-y-4">
              <Loader2 className="w-8 h-8 mx-auto animate-spin text-primary" />
              <span className="block space-y-2">
                <span className="block text-sm font-medium">{processingStage}</span>
                <Progress value={processingProgress} className="w-full max-w-md mx-auto" />
              </span>
            </span>
          ) : (
            <span className="block space-y-2">
              <Upload className="w-8 h-8 mx-auto text-primary" aria-hidden="true" />
              <span className="block text-lg font-medium">
                {hasFiles ? 'Choose more files' : 'Choose files'}
              </span>
              <span className="block text-sm text-muted-foreground">or drop them here</span>
            </span>
          )}
        </button>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".mp4,.webm,.avi,.mov,.json,.vtt,.rttm,.eaf,.wav,.mp3"
          onChange={handleFileInputChange}
          className="hidden"
          disabled={isProcessing}
        />
        <p className="text-sm text-muted-foreground">
          A video (MP4, WebM, MOV or AVI) with the results that go with it: files from a VideoAnnotator run,
          subtitles (.vtt), speaker turns (.rttm) or ELAN annotations (.eaf).
        </p>

        {hasFiles && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-medium">{fileStatuses.length === 1 ? '1 file' : `${fileStatuses.length} files`}</h3>
              <Button variant="ghost" size="sm" onClick={clearAllFiles} disabled={isProcessing}>
                Clear
              </Button>
            </div>

            <ul className="grid gap-2 max-h-64 overflow-y-auto">
              {fileStatuses.map((status, index) => (
                <li key={index} className="flex items-center gap-3 p-3 border rounded-lg bg-card">
                  <div className="flex items-center gap-2 flex-1 min-w-0">
                    <FileTypeIcon type={status.detected.type} />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate" title={status.file.name}>{status.file.name}</p>
                      <p className="text-xs text-muted-foreground">
                        <FileTypeLabel type={status.detected.type} confidence={status.detected.confidence} />
                        {status.error && <span className="text-destructive"> · {status.error}</span>}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    {status.status === 'success' && <CheckCircle2 className="w-4 h-4 text-green-600" aria-label="read" />}
                    {status.status === 'error' && <AlertCircle className="w-4 h-4 text-red-600" aria-label="can't be read" />}
                    {status.status === 'processing' && <Loader2 className="w-4 h-4 animate-spin text-primary" aria-label="reading" />}
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => removeFile(index)}
                      disabled={isProcessing}
                      aria-label={`Remove ${status.file.name}`}
                    >
                      <X className="w-4 h-4" />
                    </Button>
                  </div>
                </li>
              ))}
            </ul>

            {(!hasVideoFile || !hasPipelineData) && (
              <p className="text-sm text-amber-700 dark:text-amber-400" role="status">
                {!hasVideoFile && !hasPipelineData
                  ? 'Add a video and at least one results file.'
                  : !hasVideoFile
                    ? 'Add the video these results are for.'
                    : 'Add at least one results file for this video.'}
              </p>
            )}

            <Button onClick={processFiles} disabled={!canProcess} className="w-full" size="lg">
              {isProcessing ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Opening…
                </>
              ) : (
                `Open ${fileStatuses.length === 1 ? '1 file' : `${fileStatuses.length} files`}`
              )}
            </Button>
          </div>
        )}
      </div>
    </Card>
  );
};
