import { useEffect, useRef, useCallback, useState } from 'react';
import { StandardAnnotationData, TimelineSettings } from '@/types/annotations';

interface TimelineProps {
  annotationData: StandardAnnotationData;
  currentTime: number;
  duration: number;
  settings: TimelineSettings;
  onSeek: (time: number) => void;
}

export const Timeline = ({ annotationData, currentTime, duration, settings, onSeek }: TimelineProps) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const motionCanvasRef = useRef<HTMLCanvasElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  // Handle click and drag on timeline
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    setIsDragging(true);
    handleMouseMove(e);
  }, [handleMouseMove]);

  const handleMouseMove = useCallback((e: React.MouseEvent | MouseEvent) => {
    if (!containerRef.current) return;

    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percentage = Math.max(0, Math.min(1, x / rect.width));
    const newTime = percentage * duration;

    onSeek(newTime);
  }, [duration, onSeek]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  // Handle mouse events
  useEffect(() => {
    if (isDragging) {
      const handleGlobalMouseMove = (e: MouseEvent) => handleMouseMove(e);
      const handleGlobalMouseUp = () => setIsDragging(false);

      document.addEventListener('mousemove', handleGlobalMouseMove);
      document.addEventListener('mouseup', handleGlobalMouseUp);

      return () => {
        document.removeEventListener('mousemove', handleGlobalMouseMove);
        document.removeEventListener('mouseup', handleGlobalMouseUp);
      };
    }
  }, [isDragging, handleMouseMove]);

  // Draw motion graph from person tracking data
  const drawMotion = useCallback(() => {
    if (!settings.showMotion) return;

    const canvas = motionCanvasRef.current;
    if (!canvas || !annotationData.person_tracking) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { width, height } = canvas;

    ctx.clearRect(0, 0, width, height);
    ctx.strokeStyle = 'hsl(200, 70%, 60%)';
    ctx.lineWidth = 2;

    // Group poses by time and calculate motion intensity
    const timeSlots = new Map<number, number>();
    annotationData.person_tracking.forEach(pose => {
      const timeSlot = Math.floor(pose.timestamp * 10) / 10; // 0.1s resolution
      const currentIntensity = timeSlots.get(timeSlot) || 0;
      // Use bbox area as motion intensity proxy
      const motionIntensity = pose.bbox ? (pose.bbox[2] * pose.bbox[3]) / 10000 : 0;
      timeSlots.set(timeSlot, Math.max(currentIntensity, motionIntensity));
    });

    if (timeSlots.size === 0) return;

    const maxMotion = Math.max(...timeSlots.values());
    if (maxMotion === 0) return;

    ctx.beginPath();
    const sortedTimes = Array.from(timeSlots.entries()).sort((a, b) => a[0] - b[0]);

    sortedTimes.forEach(([timestamp, intensity], index) => {
      const x = (timestamp / duration) * width;
      const y = height - (intensity / maxMotion) * height;

      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
  }, [settings.showMotion, annotationData, duration]);

  // Note: Audio waveform would need to be generated from the audio file
  // This is removed for now since VideoAnnotator doesn't provide embedded waveforms
  const drawWaveform = useCallback(() => {
    // Placeholder - would need Web Audio API to generate waveform from audio file
    if (!settings.showWaveform) return;
    // Implementation would require loading the audio file and analyzing it
  }, [settings.showWaveform]);

  // Resize canvases
  const resizeCanvases = useCallback(() => {
    const container = containerRef.current;
    if (!container) return;

    const width = container.clientWidth;
    const height = 40;

    if (motionCanvasRef.current) {
      motionCanvasRef.current.width = width;
      motionCanvasRef.current.height = height;
      motionCanvasRef.current.style.width = `${width}px`;
      motionCanvasRef.current.style.height = `${height}px`;
    }
  }, []);

  // Initial setup and resize handling
  useEffect(() => {
    resizeCanvases();
    const handleResize = () => {
      resizeCanvases();
      drawWaveform();
      drawMotion();
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [resizeCanvases, drawWaveform, drawMotion]);

  // Redraw when data changes
  useEffect(() => {
    drawWaveform();
    drawMotion();
  }, [drawWaveform, drawMotion]);

  const playheadPosition = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div className="h-full bg-video-timeline p-4">
      <div className="h-full flex flex-col">
        {/* Timeline Header */}
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-medium text-foreground">Timeline</h3>
          <div className="text-xs text-muted-foreground">
            {Math.floor(currentTime / 60)}:{(currentTime % 60).toFixed(1).padStart(4, '0')} /
            {Math.floor(duration / 60)}:{(duration % 60).toFixed(1).padStart(4, '0')}
          </div>
        </div>

        {/* Main Timeline */}
        <div
          ref={containerRef}
          className="relative flex-1 bg-secondary rounded cursor-pointer"
          onMouseDown={handleMouseDown}
        >
          {/* Playhead */}
          <div
            className="absolute top-0 w-0.5 h-full bg-video-playhead z-10 pointer-events-none"
            style={{ left: `${playheadPosition}%` }}
          />

          {/* Subtitle Track (WebVTT) */}
          {settings.showSubtitles && annotationData.speech_recognition && (
            <div className="absolute top-2 left-0 right-0 h-4">
              {annotationData.speech_recognition.map((cue, index) => {
                const left = (cue.startTime / duration) * 100;
                const width = ((cue.endTime - cue.startTime) / duration) * 100;
                return (
                  <div
                    key={cue.id || index}
                    className="absolute h-3 bg-blue-500 rounded opacity-80 hover:opacity-100"
                    style={{
                      left: `${left}%`,
                      width: `${width}%`,
                      top: '0px'
                    }}
                    title={cue.text}
                  />
                );
              })}
            </div>
          )}

          {/* Speaker Track (RTTM) */}
          {settings.showSpeakers && annotationData.speaker_diarization && (
            <div className="absolute top-6 left-0 right-0 h-4">
              {annotationData.speaker_diarization.map((segment, index) => {
                const left = (segment.start_time / duration) * 100;
                const width = ((segment.end_time - segment.start_time) / duration) * 100;
                const hue = (segment.speaker_id.charCodeAt(0) * 137.508) % 360;
                return (
                  <div
                    key={index}
                    className="absolute h-3 rounded opacity-80 hover:opacity-100"
                    style={{
                      left: `${left}%`,
                      width: `${width}%`,
                      top: '0px',
                      backgroundColor: `hsl(${hue}, 70%, 60%)`
                    }}
                    title={`Speaker: ${segment.speaker_id}`}
                  />
                );
              })}
            </div>
          )}

          {/* Scene Track */}
          {settings.showScenes && annotationData.scene_detection && (
            <div className="absolute top-10 left-0 right-0 h-4">
              {annotationData.scene_detection.map((scene, index) => {
                const left = (scene.start_time / duration) * 100;
                const width = ((scene.end_time - scene.start_time) / duration) * 100;
                return (
                  <div
                    key={scene.id || index}
                    className="absolute h-3 bg-green-500 rounded opacity-80 hover:opacity-100"
                    style={{
                      left: `${left}%`,
                      width: `${width}%`,
                      top: '0px'
                    }}
                    title={`Scene: ${scene.scene_type}`}
                  />
                );
              })}
            </div>
          )}

          {/* Motion Graph */}
          {settings.showMotion && (
            <canvas
              ref={motionCanvasRef}
              className="absolute bottom-0 left-0 opacity-80"
            />
          )}
        </div>

        {/* Enhanced Time Markers with Progress Info */}
        <div className="flex justify-between items-center text-xs text-muted-foreground mt-1">
          <span>0:00</span>
          
          {/* Center: Current Progress */}
          <div className="flex items-center gap-2 text-primary font-mono">
            <span>{Math.floor(currentTime / 60)}:{(currentTime % 60).toFixed(2).padStart(5, '0')}</span>
            <span className="text-muted-foreground">|</span>
            <span>{((currentTime / duration) * 100).toFixed(1)}%</span>
          </div>
          
          <span>{Math.floor(duration / 60)}:{(duration % 60).toFixed(0).padStart(2, '0')}</span>
        </div>

        {/* Progress Indicator */}
        <div className="mt-1 text-xs text-muted-foreground text-center">
          Click timeline to scrub • Tracks sync with overlay controls
        </div>
      </div>
    </div>
  );
};