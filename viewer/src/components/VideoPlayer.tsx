import { forwardRef, useEffect, useRef, useCallback, useState } from 'react';
import { AnnotationData, OverlaySettings, FrameData } from '@/types/annotations';

interface VideoPlayerProps {
  videoFile: File;
  annotationData: AnnotationData;
  currentTime: number;
  overlaySettings: OverlaySettings;
  onTimeUpdate: (time: number) => void;
  onDurationChange: (duration: number) => void;
  onPlayStateChange: (playing: boolean) => void;
}

export const VideoPlayer = forwardRef<HTMLVideoElement, VideoPlayerProps>(
  ({ videoFile, annotationData, currentTime, overlaySettings, onTimeUpdate, onDurationChange, onPlayStateChange }, ref) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const [videoUrl, setVideoUrl] = useState<string>('');
    const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

    // Create video URL from file
    useEffect(() => {
      if (videoFile) {
        const url = URL.createObjectURL(videoFile);
        setVideoUrl(url);
        return () => URL.revokeObjectURL(url);
      }
    }, [videoFile]);

    // Get current frame data based on current time
    const getCurrentFrameData = useCallback((): FrameData | null => {
      if (!annotationData) return null;
      
      const frameNumber = Math.floor(currentTime * annotationData.video.frameRate);
      return annotationData.frames[frameNumber] || null;
    }, [currentTime, annotationData]);

    // Draw pose overlay
    const drawPose = useCallback((ctx: CanvasRenderingContext2D, frameData: FrameData) => {
      if (!overlaySettings.pose || !frameData.persons) return;

      frameData.persons.forEach((person, index) => {
        // Use different colors for different people
        const hue = (index * 137.508) % 360; // Golden angle for good color distribution
        ctx.strokeStyle = `hsl(${hue}, 70%, 60%)`;
        ctx.fillStyle = `hsl(${hue}, 70%, 60%)`;
        ctx.lineWidth = 2;

        // Draw keypoints
        person.keypoints.forEach((keypoint) => {
          if (keypoint.confidence > 0.5) {
            ctx.beginPath();
            ctx.arc(keypoint.x, keypoint.y, 3, 0, 2 * Math.PI);
            ctx.fill();
          }
        });

        // Draw connections
        person.connections.forEach(([i, j]) => {
          const kp1 = person.keypoints[i];
          const kp2 = person.keypoints[j];
          if (kp1.confidence > 0.5 && kp2.confidence > 0.5) {
            ctx.beginPath();
            ctx.moveTo(kp1.x, kp1.y);
            ctx.lineTo(kp2.x, kp2.y);
            ctx.stroke();
          }
        });
      });
    }, [overlaySettings.pose]);

    // Draw face emotion overlay
    const drawFaceEmotions = useCallback((ctx: CanvasRenderingContext2D, frameData: FrameData) => {
      if (!overlaySettings.faceEmotion || !frameData.faces) return;

      ctx.strokeStyle = 'hsl(var(--face-emotion))';
      ctx.fillStyle = 'hsl(var(--face-emotion))';
      ctx.lineWidth = 2;
      ctx.font = '14px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';

      frameData.faces.forEach((face) => {
        // Draw bounding box
        ctx.strokeRect(face.boundingBox.x, face.boundingBox.y, face.boundingBox.width, face.boundingBox.height);
        
        // Draw emotion label
        const labelY = face.boundingBox.y - 5;
        const labelText = `${face.emotion} (${Math.round(face.confidence * 100)}%)`;
        
        // Background for text
        const textMetrics = ctx.measureText(labelText);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(face.boundingBox.x, labelY - 16, textMetrics.width + 8, 20);
        
        // Text
        ctx.fillStyle = 'hsl(var(--face-emotion))';
        ctx.fillText(labelText, face.boundingBox.x + 4, labelY - 2);
      });
    }, [overlaySettings.faceEmotion]);

    // Draw audio emotion overlay
    const drawAudioEmotion = useCallback((ctx: CanvasRenderingContext2D, frameData: FrameData) => {
      if (!overlaySettings.audioEmotion || !frameData.audioEmotion) return;

      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.fillRect(10, 10, 200, 40);
      
      ctx.fillStyle = 'hsl(var(--audio-emotion))';
      ctx.font = '16px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      ctx.fillText(
        `Voice: ${frameData.audioEmotion.emotion}`,
        15, 35
      );
    }, [overlaySettings.audioEmotion]);

    // Draw subtitle overlay
    const drawSubtitles = useCallback((ctx: CanvasRenderingContext2D) => {
      if (!overlaySettings.subtitles || !annotationData) return;

      const currentSubtitle = annotationData.events.find(
        event => event.type === 'subtitle' && 
                 currentTime >= event.startTime && 
                 currentTime <= event.endTime
      );

      if (currentSubtitle && currentSubtitle.content) {
        const canvasHeight = ctx.canvas.height;
        const text = currentSubtitle.content;
        
        ctx.font = '18px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
        ctx.textAlign = 'center';
        
        const textWidth = ctx.measureText(text).width;
        const x = ctx.canvas.width / 2;
        const y = canvasHeight - 50;
        
        // Background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.fillRect(x - textWidth / 2 - 10, y - 25, textWidth + 20, 35);
        
        // Text
        ctx.fillStyle = 'hsl(var(--subtitle-color))';
        ctx.fillText(text, x, y - 5);
        ctx.textAlign = 'left';
      }
    }, [overlaySettings.subtitles, annotationData, currentTime]);

    // Draw event overlay
    const drawEvents = useCallback((ctx: CanvasRenderingContext2D) => {
      if (!overlaySettings.events || !annotationData) return;

      const currentEvents = annotationData.events.filter(
        event => event.type !== 'subtitle' && 
                 currentTime >= event.startTime && 
                 currentTime <= event.endTime
      );

      currentEvents.forEach((event, index) => {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(10, 60 + index * 35, 250, 30);
        
        ctx.fillStyle = 'hsl(var(--event-color))';
        ctx.font = '14px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
        ctx.fillText(`Event: ${event.label}`, 15, 80 + index * 35);
      });
    }, [overlaySettings.events, annotationData, currentTime]);

    // Render all overlays
    const renderOverlays = useCallback(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const frameData = getCurrentFrameData();
      if (!frameData) return;

      // Draw all enabled overlays
      drawPose(ctx, frameData);
      drawFaceEmotions(ctx, frameData);
      drawAudioEmotion(ctx, frameData);
      drawSubtitles(ctx);
      drawEvents(ctx);
    }, [getCurrentFrameData, drawPose, drawFaceEmotions, drawAudioEmotion, drawSubtitles, drawEvents]);

    // Update overlays when current time changes
    useEffect(() => {
      renderOverlays();
    }, [currentTime, overlaySettings, renderOverlays]);

    // Handle video events
    const handleLoadedMetadata = useCallback(() => {
      const video = ref as React.MutableRefObject<HTMLVideoElement>;
      if (video.current) {
        onDurationChange(video.current.duration);
        setDimensions({
          width: video.current.videoWidth,
          height: video.current.videoHeight
        });
      }
    }, [onDurationChange, ref]);

    const handleTimeUpdate = useCallback(() => {
      const video = ref as React.MutableRefObject<HTMLVideoElement>;
      if (video.current) {
        onTimeUpdate(video.current.currentTime);
      }
    }, [onTimeUpdate, ref]);

    const handlePlay = useCallback(() => {
      onPlayStateChange(true);
    }, [onPlayStateChange]);

    const handlePause = useCallback(() => {
      onPlayStateChange(false);
    }, [onPlayStateChange]);

    // Resize canvas to match video
    useEffect(() => {
      const canvas = canvasRef.current;
      const container = containerRef.current;
      if (!canvas || !container || !dimensions.width || !dimensions.height) return;

      const containerRect = container.getBoundingClientRect();
      const aspectRatio = dimensions.width / dimensions.height;
      
      let width = containerRect.width;
      let height = width / aspectRatio;
      
      if (height > containerRect.height) {
        height = containerRect.height;
        width = height * aspectRatio;
      }

      canvas.width = width;
      canvas.height = height;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
    }, [dimensions]);

    return (
      <div ref={containerRef} className="relative w-full h-full flex items-center justify-center">
        <div className="relative">
          <video
            ref={ref}
            src={videoUrl}
            className="max-w-full max-h-full"
            onLoadedMetadata={handleLoadedMetadata}
            onTimeUpdate={handleTimeUpdate}
            onPlay={handlePlay}
            onPause={handlePause}
            style={{ display: 'block' }}
          />
          <canvas
            ref={canvasRef}
            className="absolute top-0 left-0 pointer-events-none"
            style={{ 
              width: '100%', 
              height: '100%',
              objectFit: 'contain'
            }}
          />
        </div>
      </div>
    );
  }
);