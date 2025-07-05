import { useEffect, useRef, useCallback, useState } from 'react';
import { AnnotationData, TimelineSettings } from '@/types/annotations';

interface TimelineProps {
  annotationData: AnnotationData;
  currentTime: number;
  duration: number;
  settings: TimelineSettings;
  onSeek: (time: number) => void;
}

export const Timeline = ({ annotationData, currentTime, duration, settings, onSeek }: TimelineProps) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const waveformCanvasRef = useRef<HTMLCanvasElement>(null);
  const motionCanvasRef = useRef<HTMLCanvasElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  // Handle click and drag on timeline
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    setIsDragging(true);
    handleMouseMove(e);
  }, []);

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

  // Draw waveform
  const drawWaveform = useCallback(() => {
    if (!settings.showWaveform) return;
    
    const canvas = waveformCanvasRef.current;
    if (!canvas || !annotationData.audio) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const { width, height } = canvas;
    const { waveform } = annotationData.audio;
    
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = 'hsl(var(--waveform-color))';
    
    const samplesPerPixel = waveform.length / width;
    
    for (let x = 0; x < width; x++) {
      const startIndex = Math.floor(x * samplesPerPixel);
      const endIndex = Math.floor((x + 1) * samplesPerPixel);
      
      let max = 0;
      for (let i = startIndex; i < endIndex && i < waveform.length; i++) {
        max = Math.max(max, Math.abs(waveform[i]));
      }
      
      const barHeight = max * height;
      ctx.fillRect(x, (height - barHeight) / 2, 1, barHeight);
    }
  }, [settings.showWaveform, annotationData]);

  // Draw motion graph
  const drawMotion = useCallback(() => {
    if (!settings.showMotion) return;
    
    const canvas = motionCanvasRef.current;
    if (!canvas || !annotationData.frames) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const { width, height } = canvas;
    
    ctx.clearRect(0, 0, width, height);
    ctx.strokeStyle = 'hsl(var(--motion-color))';
    ctx.lineWidth = 2;
    
    const frames = Object.values(annotationData.frames);
    const maxMotion = Math.max(...frames.map(f => f.motionIntensity));
    
    ctx.beginPath();
    frames.forEach((frame, index) => {
      const x = (frame.timestamp / duration) * width;
      const y = height - (frame.motionIntensity / maxMotion) * height;
      
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
  }, [settings.showMotion, annotationData, duration]);

  // Resize canvases
  const resizeCanvases = useCallback(() => {
    const container = containerRef.current;
    if (!container) return;
    
    const width = container.clientWidth;
    const height = 40;
    
    [waveformCanvasRef, motionCanvasRef].forEach(ref => {
      if (ref.current) {
        ref.current.width = width;
        ref.current.height = height;
        ref.current.style.width = `${width}px`;
        ref.current.style.height = `${height}px`;
      }
    });
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

          {/* Subtitle Track */}
          {settings.showSubtitles && (
            <div className="absolute top-2 left-0 right-0 h-4">
              {annotationData.events
                .filter(event => event.type === 'subtitle')
                .map(event => {
                  const left = (event.startTime / duration) * 100;
                  const width = ((event.endTime - event.startTime) / duration) * 100;
                  return (
                    <div
                      key={event.id}
                      className="absolute h-3 bg-timeline-subtitle rounded opacity-80 hover:opacity-100"
                      style={{ 
                        left: `${left}%`, 
                        width: `${width}%`,
                        top: '0px'
                      }}
                      title={event.content}
                    />
                  );
                })}
            </div>
          )}

          {/* Event Track */}
          {settings.showEvents && (
            <div className="absolute top-6 left-0 right-0 h-4">
              {annotationData.events
                .filter(event => event.type !== 'subtitle')
                .map(event => {
                  const left = (event.startTime / duration) * 100;
                  const width = ((event.endTime - event.startTime) / duration) * 100;
                  return (
                    <div
                      key={event.id}
                      className="absolute h-3 bg-timeline-event rounded opacity-80 hover:opacity-100"
                      style={{ 
                        left: `${left}%`, 
                        width: `${width}%`,
                        top: '0px'
                      }}
                      title={event.label}
                    />
                  );
                })}
            </div>
          )}

          {/* Waveform */}
          {settings.showWaveform && (
            <canvas
              ref={waveformCanvasRef}
              className="absolute bottom-16 left-0 opacity-60"
            />
          )}

          {/* Motion Graph */}
          {settings.showMotion && (
            <canvas
              ref={motionCanvasRef}
              className="absolute bottom-0 left-0 opacity-80"
            />
          )}
        </div>

        {/* Time Markers */}
        <div className="flex justify-between text-xs text-muted-foreground mt-1">
          <span>0:00</span>
          <span>{Math.floor(duration / 60)}:{(duration % 60).toFixed(0).padStart(2, '0')}</span>
        </div>
      </div>
    </div>
  );
};