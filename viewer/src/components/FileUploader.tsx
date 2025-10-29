import React, { useState, useCallback, useRef } from 'react';
import { Upload, Video, FileText, Music, Users, Eye, AlertCircle, CheckCircle2, Loader2, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { StandardAnnotationData } from '@/types/annotations';
import { useToast } from '@/hooks/use-toast';
import { parseApiError } from '@/lib/errorHandling';
import { showErrorToast, showValidationErrorToast } from '@/lib/toastHelpers';
import type { ParsedError } from '@/types/api';
import {
  detectFileType,
  detectJSONType,
  getFileTypeDescription,
  validateFileSize,
  validateFileSet,
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
    case 'face_analysis': return <Eye className="w-4 h-4" />;
    case 'openface3_faces': return <Eye className="w-4 h-4" />;
    case 'complete_results': return <CheckCircle2 className="w-4 h-4" />;
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

  const detectFiles = useCallback(async (files: File[]) => {
    if (files.length === 0) return;

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

      // For JSON files, do content analysis with both methods
      if (detected.type === 'unknown' && detected.extension === 'json') {
        try {
          // First try the simple fileUtils detection
          detected = await detectJSONType(file);
          
          // If still unknown, use the more sophisticated merger detection
          if (detected.type === 'unknown') {
            const { detectFileType: mergerDetect } = await import('@/lib/parsers/merger');
            const mergerResult = await mergerDetect(file);
            
            // Convert merger result to fileUtils format
            detected = {
              type: mergerResult.type as any, // Now that fileUtils supports all types, no conversion needed
              extension: 'json',
              mimeType: 'application/json',
              confidence: mergerResult.confidence > 0.7 ? 'high' : 
                         mergerResult.confidence > 0.4 ? 'medium' : 'low',
              reason: `Detected via content analysis (${mergerResult.confidence.toFixed(2)} confidence)`
            };
          }
        } catch (error) {
          console.warn('JSON detection failed:', error);
          // Keep as unknown if both methods fail
        }
      }

      detectedFiles.push({
        file,
        detected,
        status: 'pending'
      });
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
      const fileList = fileStatuses.map(fs => fs.file);
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

      // Convert to merger format
      const detectedFiles = await Promise.all(fileStatuses.map(async (fs) => {
        // For JSON files that still show as unknown, do deeper detection using merger
        if (fs.detected.type === 'unknown' && fs.detected.extension === 'json') {
          const { detectFileType } = await import('@/lib/parsers/merger');
          const mergerResult = await detectFileType(fs.file);
          return mergerResult;
        } else {
          // Convert fileUtils result to merger format
          const confidence = fs.detected.confidence === 'high' ? 0.9 :
            fs.detected.confidence === 'medium' ? 0.7 : 0.5;
          return {
            file: fs.file,
            type: fs.detected.type as any,
            confidence,
            pipeline: fs.detected.type !== 'video' && fs.detected.type !== 'audio' ? fs.detected.type : undefined
          };
        }
      }));

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
  }, [fileStatuses, onVideoLoad, onAnnotationLoad, toast]);

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

  const handleDemoLoad = useCallback(async (datasetKey: string) => {
    try {
      setIsProcessing(true);
      setProcessingStage(`Loading ${datasetKey} demo dataset...`);
      setProcessingProgress(10);

      // Import the demo loading utilities
      const { loadDemoAnnotations, loadDemoVideo } = await import('../utils/debugUtils');
      
      setProcessingStage('Fetching demo files...');
      setProcessingProgress(30);

      // Load video and annotations in parallel
      const [videoFile, annotation] = await Promise.all([
        loadDemoVideo(datasetKey as any),
        loadDemoAnnotations(datasetKey as any)
      ]);

      setProcessingProgress(80);

      if (videoFile && annotation) {
        setProcessingStage('Loading demo data...');
        setProcessingProgress(90);

        onVideoLoad(videoFile);
        onAnnotationLoad(annotation);
        
        setProcessingProgress(100);
        
        toast({
          title: "Demo dataset loaded",
          description: `Successfully loaded ${datasetKey} with ${annotation.metadata?.pipelines?.length || 0} pipelines.`,
        });
      } else {
        throw new Error('Failed to load demo video or annotations');
      }
    } catch (error) {
      toast({
        title: "Error loading demo dataset",
        description: error instanceof Error ? error.message : "Failed to load demo data",
        variant: "destructive",
      });
    } finally {
      setIsProcessing(false);
      setProcessingStage('');
      setProcessingProgress(0);
    }
  }, [onVideoLoad, onAnnotationLoad, toast]);

  const hasFiles = fileStatuses.length > 0;
  const hasVideoFile = fileStatuses.some(f => f.detected.type === 'video');
  const hasPipelineData = fileStatuses.some(f =>
    ['person_tracking', 'speech_recognition', 'speaker_diarization', 'scene_detection', 'face_analysis', 'openface3_faces', 'complete_results'].includes(f.detected.type)
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
        <div className="space-y-4">
          {hasFiles && (
            <Button
              onClick={processFiles}
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
                `🚀 Process ${fileStatuses.length} Files`
              )}
            </Button>
          )}

          <div className="text-center">
            <span className="text-sm text-muted-foreground">or choose a demo dataset</span>
          </div>

          {/* Demo Dataset Selection */}
          <div className="grid grid-cols-2 gap-3">
            <Button
              variant="outline"
              onClick={() => handleDemoLoad('peekaboo-rep3-v1.1.1')}
              disabled={isProcessing}
              className="text-left flex-col h-auto p-4"
            >
              <div className="font-medium">🍼 Peekaboo Rep3</div>
              <div className="text-xs text-muted-foreground">Parent-child + OpenFace3</div>
            </Button>
            <Button
              variant="outline"
              onClick={() => handleDemoLoad('peekaboo-rep2-v1.1.1')}
              disabled={isProcessing}
              className="text-left flex-col h-auto p-4"
            >
              <div className="font-medium">🍼 Peekaboo Rep2</div>
              <div className="text-xs text-muted-foreground">Alternative + OpenFace3</div>
            </Button>
            <Button
              variant="outline"
              onClick={() => handleDemoLoad('tearingpaper-rep1-v1.1.1')}
              disabled={isProcessing}
              className="text-left flex-col h-auto p-4"
            >
              <div className="font-medium">📄 Tearing Paper</div>
              <div className="text-xs text-muted-foreground">Object + OpenFace3</div>
            </Button>
            <Button
              variant="outline"
              onClick={() => handleDemoLoad('thatsnotahat-rep1-v1.1.1')}
              disabled={isProcessing}
              className="text-left flex-col h-auto p-4"
            >
              <div className="font-medium">🎩 That's Not A Hat</div>
              <div className="text-xs text-muted-foreground">Social + OpenFace3</div>
            </Button>
            <Button
              variant="outline"
              onClick={() => handleDemoLoad('veatic-3-silent')}
              disabled={isProcessing}
              className="text-left flex-col h-auto p-4 col-span-2"
            >
              <div className="font-medium">🔇 VEATIC Silent Video</div>
              <div className="text-xs text-muted-foreground">Longer duration, no speech/audio - ideal for pose tracking</div>
            </Button>
          </div>
        </div>

        {/* Format Information */}
        <div className="text-xs text-muted-foreground space-y-1 border-t pt-4">
          <p><strong>Supported VideoAnnotator formats:</strong></p>
          <p>• Person Tracking: COCO JSON format with keypoints</p>
          <p>• Face Analysis: OpenFace3 JSON with 98 landmarks, emotions, head pose, gaze</p>
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