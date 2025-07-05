import { useState, useCallback, useRef } from 'react';
import { VideoPlayer } from './VideoPlayer';
import { Timeline } from './Timeline';
import { OverlayControls } from './OverlayControls';
import { VideoControls } from './VideoControls';
import { FileUploader } from './FileUploader';
import { WelcomeScreen } from './WelcomeScreen';
import { Card } from '@/components/ui/card';
import { AnnotationData, OverlaySettings, TimelineSettings } from '@/types/annotations';

export const VideoAnnotationViewer = () => {
  const [showWelcome, setShowWelcome] = useState(true);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [annotationData, setAnnotationData] = useState<AnnotationData | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1);
  
  const [overlaySettings, setOverlaySettings] = useState<OverlaySettings>({
    pose: true,
    faceEmotion: true,
    audioEmotion: true,
    subtitles: true,
    events: true,
  });

  const [timelineSettings, setTimelineSettings] = useState<TimelineSettings>({
    showSubtitles: true,
    showEvents: true,
    showWaveform: true,
    showMotion: true,
  });

  const videoRef = useRef<HTMLVideoElement>(null);

  const handleGetStarted = useCallback(() => {
    setShowWelcome(false);
  }, []);

  const handleVideoLoad = useCallback((file: File) => {
    setVideoFile(file);
  }, []);

  const handleAnnotationLoad = useCallback((data: AnnotationData) => {
    setAnnotationData(data);
    setShowWelcome(false);
  }, []);

  const handleTimeUpdate = useCallback((time: number) => {
    setCurrentTime(time);
  }, []);

  const handleSeek = useCallback((time: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = time;
      setCurrentTime(time);
    }
  }, []);

  const handlePlayPause = useCallback(() => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  }, [isPlaying]);

  const handleFrameStep = useCallback((direction: 'forward' | 'backward') => {
    if (videoRef.current && annotationData) {
      const frameStep = 1 / annotationData.video.frameRate;
      const newTime = direction === 'forward' 
        ? Math.min(currentTime + frameStep, duration)
        : Math.max(currentTime - frameStep, 0);
      handleSeek(newTime);
    }
  }, [currentTime, duration, annotationData, handleSeek]);

  const handlePlaybackRateChange = useCallback((rate: number) => {
    setPlaybackRate(rate);
    if (videoRef.current) {
      videoRef.current.playbackRate = rate;
    }
  }, []);

  // Show welcome screen first
  if (showWelcome) {
    return <WelcomeScreen onGetStarted={handleGetStarted} />;
  }

  // Show file uploader if no files loaded
  if (!videoFile || !annotationData) {
    return (
      <div className="min-h-screen bg-background p-6">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold mb-2">Video Annotation Viewer</h1>
            <p className="text-muted-foreground">
              Load a video file and its corresponding annotation data to begin analysis
            </p>
          </div>
          <FileUploader 
            onVideoLoad={handleVideoLoad}
            onAnnotationLoad={handleAnnotationLoad}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="flex flex-col h-screen">
        {/* Header */}
        <div className="flex-shrink-0 p-4 border-b border-border">
          <div className="flex items-center justify-between">
            <h1 className="text-xl font-semibold">Video Annotation Viewer</h1>
            <div className="text-sm text-muted-foreground">
              {annotationData.video.filename}
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Video and Controls Section */}
          <div className="flex-1 flex">
            {/* Video Player Area */}
            <div className="flex-1 flex flex-col bg-video-bg">
              <div className="flex-1 relative">
                <VideoPlayer
                  ref={videoRef}
                  videoFile={videoFile}
                  annotationData={annotationData}
                  currentTime={currentTime}
                  overlaySettings={overlaySettings}
                  onTimeUpdate={handleTimeUpdate}
                  onDurationChange={setDuration}
                  onPlayStateChange={setIsPlaying}
                />
              </div>
              
              {/* Video Controls */}
              <div className="flex-shrink-0 p-4">
                <VideoControls
                  isPlaying={isPlaying}
                  currentTime={currentTime}
                  duration={duration}
                  playbackRate={playbackRate}
                  onPlayPause={handlePlayPause}
                  onSeek={handleSeek}
                  onFrameStep={handleFrameStep}
                  onPlaybackRateChange={handlePlaybackRateChange}
                />
              </div>
            </div>

            {/* Side Panel for Controls */}
            <div className="w-80 bg-card border-l border-border flex flex-col">
              <div className="p-4 border-b border-border">
                <h3 className="font-medium mb-4">Overlay Controls</h3>
                <OverlayControls
                  settings={overlaySettings}
                  onChange={setOverlaySettings}
                />
              </div>
              
              <div className="p-4">
                <h3 className="font-medium mb-4">Timeline Settings</h3>
                <div className="space-y-2">
                  {Object.entries(timelineSettings).map(([key, value]) => (
                    <label key={key} className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={value}
                        onChange={(e) =>
                          setTimelineSettings(prev => ({
                            ...prev,
                            [key]: e.target.checked
                          }))
                        }
                        className="rounded"
                      />
                      <span className="text-sm capitalize">
                        {key.replace(/([A-Z])/g, ' $1').toLowerCase()}
                      </span>
                    </label>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Timeline Section */}
          <div className="flex-shrink-0 h-64 bg-video-timeline border-t border-border">
            <Timeline
              annotationData={annotationData}
              currentTime={currentTime}
              duration={duration}
              settings={timelineSettings}
              onSeek={handleSeek}
            />
          </div>
        </div>
      </div>
    </div>
  );
};