import { useState, useCallback } from 'react';
import { Upload, Video, FileText } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { AnnotationData } from '@/types/annotations';
import { useToast } from '@/hooks/use-toast';

interface FileUploaderProps {
  onVideoLoad: (file: File) => void;
  onAnnotationLoad: (data: AnnotationData) => void;
}

export const FileUploader = ({ onVideoLoad, onAnnotationLoad }: FileUploaderProps) => {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [annotationFile, setAnnotationFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();

  const handleVideoFileChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('video/')) {
      setVideoFile(file);
      toast({
        title: "Video loaded",
        description: `${file.name} is ready to be processed.`,
      });
    } else {
      toast({
        title: "Invalid file",
        description: "Please select a valid video file.",
        variant: "destructive",
      });
    }
  }, [toast]);

  const handleAnnotationFileChange = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (file.type !== 'application/json' && !file.name.endsWith('.json')) {
      toast({
        title: "Invalid file",
        description: "Please select a valid JSON annotation file.",
        variant: "destructive",
      });
      return;
    }

    try {
      setIsLoading(true);
      const text = await file.text();
      const data = JSON.parse(text) as AnnotationData;
      
      // Basic validation
      if (!data.video || !data.frames || !data.events) {
        throw new Error('Invalid annotation format');
      }

      setAnnotationFile(file);
      toast({
        title: "Annotations loaded",
        description: `${file.name} contains ${Object.keys(data.frames).length} frames and ${data.events.length} events.`,
      });
    } catch (error) {
      toast({
        title: "Error loading annotations",
        description: "Please check that your JSON file is valid and properly formatted.",
        variant: "destructive",
      });
      console.error('Error loading annotation file:', error);
    } finally {
      setIsLoading(false);
    }
  }, [toast]);

  const handleStartViewing = useCallback(async () => {
    if (!videoFile || !annotationFile) return;

    try {
      setIsLoading(true);
      
      // Load annotation data
      const text = await annotationFile.text();
      const annotationData = JSON.parse(text) as AnnotationData;
      
      // Pass data to parent
      onVideoLoad(videoFile);
      onAnnotationLoad(annotationData);
      
      toast({
        title: "Files loaded successfully",
        description: "Ready to start annotation viewing.",
      });
    } catch (error) {
      toast({
        title: "Error loading files",
        description: "Please try again with valid files.",
        variant: "destructive",
      });
      console.error('Error loading files:', error);
    } finally {
      setIsLoading(false);
    }
  }, [videoFile, annotationFile, onVideoLoad, onAnnotationLoad, toast]);

  const loadSampleData = useCallback(() => {
    // Create sample annotation data for demonstration
    const sampleData: AnnotationData = {
      video: {
        filename: "sample-video.mp4",
        duration: 60,
        frameRate: 30,
        width: 1920,
        height: 1080
      },
      frames: {
        0: {
          frameNumber: 0,
          timestamp: 0,
          persons: [{
            id: 1,
            keypoints: [
              { x: 100, y: 150, confidence: 0.9 },
              { x: 120, y: 140, confidence: 0.8 },
              { x: 80, y: 160, confidence: 0.7 }
            ],
            connections: [[0, 1], [1, 2]],
            confidence: 0.85
          }],
          faces: [{
            id: 1,
            boundingBox: { x: 90, y: 100, width: 40, height: 50 },
            emotion: "happy",
            confidence: 0.9
          }],
          audioEmotion: {
            emotion: "neutral",
            confidence: 0.7
          },
          motionIntensity: 0.3
        }
      },
      events: [
        {
          id: "1",
          type: "subtitle",
          label: "Speech",
          content: "Hello, this is a sample subtitle.",
          startTime: 0,
          endTime: 3,
          confidence: 0.95
        },
        {
          id: "2",
          type: "action",
          label: "Wave",
          startTime: 1,
          endTime: 2.5,
          participants: [1],
          confidence: 0.8
        }
      ],
      audio: {
        waveform: Array.from({ length: 1000 }, (_, i) => Math.sin(i * 0.1) * 0.5),
        sampleRate: 44100,
        duration: 60
      },
      metadata: {
        created: new Date().toISOString(),
        version: "1.0.0",
        pipeline: "sample-pipeline"
      }
    };

    onAnnotationLoad(sampleData);
    toast({
      title: "Sample data loaded",
      description: "You can now explore the viewer with sample annotation data.",
    });
  }, [onAnnotationLoad, toast]);

  return (
    <Card className="p-8 max-w-2xl mx-auto">
      <div className="text-center space-y-6">
        <div className="space-y-2">
          <Upload className="w-12 h-12 mx-auto text-primary" />
          <h2 className="text-2xl font-bold">Load Video & Annotations</h2>
          <p className="text-muted-foreground">
            Select a video file and its corresponding JSON annotation file to begin viewing.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-4">
          {/* Video File Upload */}
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-sm font-medium">
              <Video className="w-4 h-4" />
              Video File
            </div>
            <label className="block">
              <input
                type="file"
                accept="video/*"
                onChange={handleVideoFileChange}
                className="hidden"
              />
              <div className="border-2 border-dashed border-border rounded-lg p-4 text-center hover:border-primary cursor-pointer transition-colors">
                {videoFile ? (
                  <div className="text-sm">
                    <div className="font-medium text-primary">{videoFile.name}</div>
                    <div className="text-muted-foreground">
                      {(videoFile.size / 1024 / 1024).toFixed(1)} MB
                    </div>
                  </div>
                ) : (
                  <div className="text-sm text-muted-foreground">
                    Click to select video file
                  </div>
                )}
              </div>
            </label>
          </div>

          {/* Annotation File Upload */}
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-sm font-medium">
              <FileText className="w-4 h-4" />
              Annotation File (JSON)
            </div>
            <label className="block">
              <input
                type="file"
                accept=".json,application/json"
                onChange={handleAnnotationFileChange}
                className="hidden"
              />
              <div className="border-2 border-dashed border-border rounded-lg p-4 text-center hover:border-primary cursor-pointer transition-colors">
                {annotationFile ? (
                  <div className="text-sm">
                    <div className="font-medium text-primary">{annotationFile.name}</div>
                    <div className="text-muted-foreground">
                      {(annotationFile.size / 1024).toFixed(1)} KB
                    </div>
                  </div>
                ) : (
                  <div className="text-sm text-muted-foreground">
                    Click to select JSON file
                  </div>
                )}
              </div>
            </label>
          </div>
        </div>

        <div className="space-y-3">
          <Button
            onClick={handleStartViewing}
            disabled={!videoFile || !annotationFile || isLoading}
            className="w-full"
            size="lg"
          >
            {isLoading ? 'Loading...' : 'Start Viewing'}
          </Button>

          <div className="text-center">
            <span className="text-sm text-muted-foreground">or</span>
          </div>

          <Button
            variant="outline"
            onClick={loadSampleData}
            className="w-full"
          >
            Try with Sample Data
          </Button>
        </div>

        <div className="text-xs text-muted-foreground">
          <p>Supported video formats: MP4, WebM, AVI, MOV</p>
          <p>Annotation files must be in JSON format with the required schema</p>
        </div>
      </div>
    </Card>
  );
};