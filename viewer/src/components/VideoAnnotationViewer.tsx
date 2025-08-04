import { useState, useCallback, useRef } from 'react';
import { DEMO_DATA_SETS, loadDemoAnnotations, loadDemoVideo } from '../utils/debugUtils';
import { VideoPlayer } from './VideoPlayer';
import { Timeline } from './Timeline';
import { OverlayControls } from './OverlayControls';
import { VideoControls } from './VideoControls';
import { FileUploader } from './FileUploader';
import { WelcomeScreen } from './WelcomeScreen';
import { Footer } from './Footer';
import { Card } from '@/components/ui/card';
import { StandardAnnotationData, OverlaySettings, TimelineSettings } from '@/types/annotations';

export const VideoAnnotationViewer = () => {
  const [showWelcome, setShowWelcome] = useState(true);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [annotationData, setAnnotationData] = useState<StandardAnnotationData | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1);

  const [overlaySettings, setOverlaySettings] = useState<OverlaySettings>({
    pose: true,
    subtitles: true,
    speakers: true,
    scenes: true,
  });

  const [timelineSettings, setTimelineSettings] = useState<TimelineSettings>({
    showSubtitles: true,
    showSpeakers: true,
    showScenes: true,
    showMotion: true,
  });

  const videoRef = useRef<HTMLVideoElement>(null);

  const handleGetStarted = useCallback(() => {
    setShowWelcome(false);
  }, []);

  const handleViewDemo = useCallback(async () => {
    try {
      const demoKey = Object.keys(DEMO_DATA_SETS)[0] as keyof typeof DEMO_DATA_SETS;

      // Load video and annotations in parallel
      const [videoFile, annotation] = await Promise.all([
        loadDemoVideo(demoKey),
        loadDemoAnnotations(demoKey)
      ]);

      if (videoFile && annotation) {
        setVideoFile(videoFile);
        setAnnotationData(annotation);
        setShowWelcome(false);
      } else {
        throw new Error('Failed to load demo video or annotations');
      }
    } catch (error) {
      console.error('Failed to load demo:', error);
      alert('Failed to load demo data. Please check the console for details.');
    }
  }, []);

  const handleVideoLoad = useCallback((file: File) => {
    setVideoFile(file);
  }, []);

  const handleAnnotationLoad = useCallback((data: StandardAnnotationData) => {
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
      // Use a default frame rate if not available, or calculate from video duration
      const estimatedFrameRate = annotationData.video_info?.frame_rate || 30;
      const frameStep = 1 / estimatedFrameRate;
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
    return <WelcomeScreen onGetStarted={handleGetStarted} onViewDemo={handleViewDemo} />;
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
          <div className="flex flex-col items-center gap-4 mb-6">
            <button
              className="px-6 py-2 rounded bg-primary text-primary-foreground font-semibold shadow hover:bg-primary/90 transition"
              onClick={handleViewDemo}
              type="button"
            >
              ▶️ View Demo
            </button>
            <span className="text-xs text-muted-foreground">Loads sample video and annotations from demo folder</span>
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
              {annotationData.video_info?.filename || videoFile.name}
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

        {/* Footer */}
        <Footer />
      </div>
    </div>
  );
};