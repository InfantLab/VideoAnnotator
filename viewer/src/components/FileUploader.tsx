import React, { useState, useCallback, useRef } from 'react';
import { Upload, Video, FileText, Music, Users, Eye, AlertCircle, CheckCircle2, Loader2, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { StandardAnnotationData } from '@/types/annotations';
import { useToast } from '@/hooks/use-toast';
import {
  detectFileType,
  detectJSONType,
  getFileTypeDescription,
  validateFileSize,
  validateFileSet,
  generateFilesSummary,
  type FileTypeInfo
} from '@/lib/fileUtils';
import {
  mergeAnnotationData
} from '@/lib/parsers/merger';

interface FileUploaderProps {
  onVideoLoad: (file: File) => void;
  onAnnotationLoad: (data: StandardAnnotationData) => void;
}

interface FileStatus {
  file: File;
  detected: FileTypeInfo;
  status: 'pending' | 'processing' | 'success' | 'error';
  error?: string;
}

const FileTypeIcon = ({ type }: { type: FileTypeInfo['type'] }) => {
  switch (type) {
    case 'video': return <Video className="w-4 h-4" />;
    case 'audio': return <Music className="w-4 h-4" />;
    case 'person_tracking': return <Users className="w-4 h-4" />;
    case 'speech_recognition': return <FileText className="w-4 h-4" />;
    case 'speaker_diarization': return <Users className="w-4 h-4" />;
    case 'scene_detection': return <Eye className="w-4 h-4" />;
    default: return <AlertCircle className="w-4 h-4" />;
  }
};

const FileTypeLabel = ({ type, confidence }: { type: FileTypeInfo['type']; confidence?: string }) => {
  return (
    <span className={`text-xs px-2 py-1 rounded ${confidence === 'high' ? 'bg-green-100 text-green-800' :
        confidence === 'medium' ? 'bg-yellow-100 text-yellow-800' :
          'bg-red-100 text-red-800'
      }`}>
      {getFileTypeDescription(type)}
    </span>
  );
};

export const FileUploader = ({ onVideoLoad, onAnnotationLoad }: FileUploaderProps) => {
  const [fileStatuses, setFileStatuses] = useState<FileStatus[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [processingStage, setProcessingStage] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const handleFilesSelected = useCallback(async (files: File[]) => {
    if (files.length === 0) return;

    try {
      setIsProcessing(true);
      setProcessingStage('Detecting file types...');
      setProcessingProgress(10);

      // Process each file individually
      const detectedFiles: FileStatus[] = [];
      for (const file of files) {
        // Validate file size first
        const sizeValidation = validateFileSize(file);
        if (!sizeValidation.valid) {
          detectedFiles.push({
            file,
            detected: detectFileType(file),
            status: 'error',
            error: sizeValidation.error
          });
          continue;
        }

        let detected = detectFileType(file);

        // For JSON files, do content analysis
        if (detected.type === 'unknown' && detected.extension === 'json') {
          detected = await detectJSONType(file);
        }

        detectedFiles.push({
          file,
          detected,
          status: 'pending'
        });
      }

      setFileStatuses(detectedFiles);
      setProcessingProgress(30);

      // Validate file set
      const fileList = detectedFiles.map(fs => fs.file);
      const validation = validateFileSet(fileList);

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

      // Convert to legacy merger format for now
      const legacyDetectedFiles = detectedFiles.map(fs => ({
        file: fs.file,
        type: fs.detected.type as any, // Convert to legacy type
        confidence: fs.detected.confidence === 'high' ? 0.9 :
          fs.detected.confidence === 'medium' ? 0.7 : 0.5
      }));

      // Parse and merge data using existing merger
      const result = await mergeAnnotationData(legacyDetectedFiles, (stage, progress, total) => {
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
      const videoFile = detectedFiles.find(f => f.detected.type === 'video')?.file;
      if (videoFile) {
        onVideoLoad(videoFile);
      }

      onAnnotationLoad(result.data);

    } catch (error) {
      toast({
        title: "Error processing files",
        description: error instanceof Error ? error.message : "Unknown error occurred",
        variant: "destructive",
      });

      // Update file statuses to show errors
      setFileStatuses(prev => prev.map(status => ({
        ...status,
        status: 'error',
        error: error instanceof Error ? error.message : 'Unknown error'
      })));
    } finally {
      setIsProcessing(false);
      setProcessingStage('');
      setProcessingProgress(0);
    }
  }, [onVideoLoad, onAnnotationLoad, toast]);

  const handleFileInputChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files) {
      handleFilesSelected(Array.from(files));
    }
  }, [handleFilesSelected]);

  const handleDrop = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const files = event.dataTransfer.files;
    if (files) {
      handleFilesSelected(Array.from(files));
    }
  }, [handleFilesSelected]);

  const handleDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
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

  const loadSampleData = useCallback(async () => {
    try {
      setIsProcessing(true);
      setProcessingStage('Loading sample data...');
      setProcessingProgress(50);

      // Create sample video file info (no actual file)
      const sampleData: StandardAnnotationData = {
        video_info: {
          filename: "sample-video.mp4",
          duration: 60,
          width: 1920,
          height: 1080,
          frame_rate: 30
        },
        metadata: {
          created: new Date().toISOString(),
          version: '1.0.0',
          pipelines: ['sample'],
          source: 'custom'
        }
      };

      setProcessingProgress(100);

      toast({
        title: "Sample data loaded",
        description: "You can now explore the viewer with sample annotation data.",
      });

      onAnnotationLoad(sampleData);
    } catch (error) {
      toast({
        title: "Error loading sample data",
        description: "Failed to create sample data",
        variant: "destructive",
      });
    } finally {
      setIsProcessing(false);
      setProcessingStage('');
      setProcessingProgress(0);
    }
  }, [onAnnotationLoad, toast]);

  const hasFiles = fileStatuses.length > 0;
  const hasVideoFile = fileStatuses.some(f => f.detected.type === 'video');
  const hasPipelineData = fileStatuses.some(f =>
    ['person_tracking', 'speech_recognition', 'speaker_diarization', 'scene_detection'].includes(f.detected.type)
  );
  const canProcess = hasVideoFile && hasPipelineData && !isProcessing;

  return (
    <Card className="p-8 max-w-4xl mx-auto">
      <div className="space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <Upload className="w-12 h-12 mx-auto text-primary" />
          <h2 className="text-2xl font-bold">Load Video & Annotations</h2>
          <p className="text-muted-foreground">
            Select multiple files from VideoAnnotator output or drag & drop them here.
          </p>
        </div>

        {/* File Drop Zone */}
        <div
          className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${isProcessing
              ? 'border-muted bg-muted/10'
              : 'border-border hover:border-primary cursor-pointer'
            }`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onClick={() => !isProcessing && fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".mp4,.webm,.avi,.mov,.json,.vtt,.rttm,.wav,.mp3"
            onChange={handleFileInputChange}
            className="hidden"
            disabled={isProcessing}
          />

          {isProcessing ? (
            <div className="space-y-4">
              <Loader2 className="w-8 h-8 mx-auto animate-spin text-primary" />
              <div className="space-y-2">
                <p className="text-sm font-medium">{processingStage}</p>
                <Progress value={processingProgress} className="w-full max-w-md mx-auto" />
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="space-y-2">
                <p className="text-lg font-medium">
                  {hasFiles ? 'Add more files or process current selection' : 'Click to select files or drag & drop'}
                </p>
                <p className="text-sm text-muted-foreground">
                  Supported: Video (.mp4, .webm), Annotations (.json), Subtitles (.vtt), Speaker data (.rttm), Audio (.wav)
                </p>
              </div>
            </div>
          )}
        </div>

        {/* File List */}
        {hasFiles && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-medium">Selected Files ({fileStatuses.length})</h3>
              <Button
                variant="outline"
                size="sm"
                onClick={clearAllFiles}
                disabled={isProcessing}
              >
                Clear All
              </Button>
            </div>

            <div className="grid gap-2 max-h-64 overflow-y-auto">
              {fileStatuses.map((status, index) => (
                <div
                  key={index}
                  className="flex items-center gap-3 p-3 border rounded-lg bg-card"
                >
                  <div className="flex items-center gap-2 flex-1 min-w-0">
                    <FileTypeIcon type={status.detected.type} />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">{status.file.name}</p>
                      <p className="text-xs text-muted-foreground">
                        <FileTypeLabel type={status.detected.type} confidence={status.detected.confidence} />
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    {status.status === 'success' && <CheckCircle2 className="w-4 h-4 text-green-500" />}
                    {status.status === 'error' && <AlertCircle className="w-4 h-4 text-red-500" />}
                    {status.status === 'processing' && <Loader2 className="w-4 h-4 animate-spin text-primary" />}

                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => removeFile(index)}
                      disabled={isProcessing}
                    >
                      <X className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>

            {/* File Validation Summary */}
            <div className="p-3 border rounded-lg bg-muted/10">
              <div className="text-sm space-y-1">
                <div className="flex items-center gap-2">
                  {hasVideoFile ? (
                    <CheckCircle2 className="w-4 h-4 text-green-500" />
                  ) : (
                    <AlertCircle className="w-4 h-4 text-red-500" />
                  )}
                  <span className={hasVideoFile ? 'text-green-700' : 'text-red-700'}>
                    Video file {hasVideoFile ? 'found' : 'required'}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  {hasPipelineData ? (
                    <CheckCircle2 className="w-4 h-4 text-green-500" />
                  ) : (
                    <AlertCircle className="w-4 h-4 text-red-500" />
                  )}
                  <span className={hasPipelineData ? 'text-green-700' : 'text-red-700'}>
                    Pipeline data {hasPipelineData ? 'found' : 'required'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="space-y-3">
          {hasFiles && (
            <Button
              onClick={() => { }} // Will trigger processing automatically when files are selected
              disabled={!canProcess}
              className="w-full"
              size="lg"
            >
              {isProcessing ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Processing...
                </>
              ) : (
                `Process ${fileStatuses.length} Files`
              )}
            </Button>
          )}

          <div className="text-center">
            <span className="text-sm text-muted-foreground">or</span>
          </div>

          <Button
            variant="outline"
            onClick={loadSampleData}
            disabled={isProcessing}
            className="w-full"
          >
            Try with Sample Data
          </Button>
        </div>

        {/* Format Information */}
        <div className="text-xs text-muted-foreground space-y-1 border-t pt-4">
          <p><strong>Supported VideoAnnotator formats:</strong></p>
          <p>• Person Tracking: COCO JSON format with keypoints</p>
          <p>• Speech Recognition: WebVTT files (.vtt)</p>
          <p>• Speaker Diarization: RTTM files (.rttm)</p>
          <p>• Scene Detection: JSON arrays with scene data</p>
          <p>• Video: MP4, WebM, AVI, MOV</p>
          <p>• Audio: WAV, MP3 (optional)</p>
        </div>
      </div>
    </Card>
  );
};