import React, { useState, useCallback, useRef } from 'react';
import { DEMO_DATA_SETS, loadDemoAnnotations, loadDemoVideo } from '../utils/debugUtils';
import { VideoPlayer } from './VideoPlayer';
import { Timeline } from './Timeline';
import { UnifiedControls } from './UnifiedControls';
import { VideoControls } from './VideoControls';
import { FileViewer } from './FileViewer';
import { FileUploader } from './FileUploader';
import { WelcomeScreen } from './WelcomeScreen';
import { Footer } from './Footer';
import { DebugPanel } from './DebugPanel';
import { OpenFace3Controls, OpenFace3Settings, defaultOpenFace3Settings } from './OpenFace3Controls';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
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
    faces: true,
    emotions: true,
  });

  const [timelineSettings, setTimelineSettings] = useState<TimelineSettings>({
    showSubtitles: true,
    showSpeakers: true,
    showScenes: true,
    showMotion: true,
    showFaces: true,
    showEmotions: true,
  });

  const [openface3Settings, setOpenface3Settings] = useState<OpenFace3Settings>(defaultOpenFace3Settings);

  const [showDebugPanel, setShowDebugPanel] = useState(false);

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

  const handleBackToHome = useCallback(() => {
    // Reset all state and return to welcome screen
    setVideoFile(null);
    setAnnotationData(null);
    setCurrentTime(0);
    setIsPlaying(false);
    setDuration(0);
    setPlaybackRate(1);
    setShowWelcome(true);
  }, []);

  // Debug panel keyboard shortcut
  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    if (event.ctrlKey && event.shiftKey && event.key === 'D') {
      event.preventDefault();
      setShowDebugPanel(true);
    }
  }, []);

  // Add keyboard listener (works on all pages)
  React.useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  // Show welcome screen first
  if (showWelcome) {
    return (
      <>
        <WelcomeScreen onGetStarted={handleGetStarted} onViewDemo={handleViewDemo} />
        {/* Debug Panel - Available on all pages */}
        <DebugPanel 
          isOpen={showDebugPanel}
          onClose={() => setShowDebugPanel(false)}
        />
      </>
    );
  }

  // Show file uploader if no files loaded
  if (!videoFile || !annotationData) {
    return (
      <>
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
            
            {/* Debug button for file uploader page */}
            <div className="fixed bottom-4 right-4">
              <Button 
                variant="ghost" 
                size="sm"
                onClick={() => setShowDebugPanel(true)}
                className="text-xs bg-background border"
                title="Debug Panel (Ctrl+Shift+D)"
              >
                🐛
              </Button>
            </div>
          </div>
        </div>
        
        {/* Debug Panel - Available on all pages */}
        <DebugPanel 
          isOpen={showDebugPanel}
          onClose={() => setShowDebugPanel(false)}
        />
      </>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="flex flex-col h-screen">
        {/* Header */}
        <div className="flex-shrink-0 p-4 border-b border-border">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Button 
                variant="ghost" 
                size="sm"
                onClick={handleBackToHome}
                className="flex items-center gap-2"
                title="Back to Home"
              >
                ← Home
              </Button>
              <h1 className="text-xl font-semibold">Video Annotation Viewer</h1>
            </div>
            <div className="flex items-center gap-4">
              <div className="text-sm text-muted-foreground">
                {annotationData.video_info?.filename || videoFile.name}
              </div>
              <div className="flex gap-2">
                <FileViewer 
                  annotationData={annotationData}
                  trigger={
                    <Button variant="outline" size="sm" className="flex items-center gap-2">
                      📄 View Data
                    </Button>
                  }
                />
                <Button 
                  variant="ghost" 
                  size="sm"
                  onClick={() => setShowDebugPanel(true)}
                  className="text-xs"
                  title="Debug Panel (Ctrl+Shift+D)"
                >
                  🐛
                </Button>
              </div>
            </div>
          </div>
        </div>

        {/* Main Content - Full height container */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Two Column Layout: Video Player + Controls | Right Panel */}
          <div className="flex-1 flex min-h-0">
            {/* Column 1: Video Player with Controls and Timeline */}
            <div className="flex-1 flex flex-col bg-video-bg min-w-0">
              {/* Video Player - takes remaining space after controls */}
              <div className="flex-1 relative min-h-0">
                <VideoPlayer
                  ref={videoRef}
                  videoFile={videoFile}
                  annotationData={annotationData}
                  currentTime={currentTime}
                  overlaySettings={overlaySettings}
                  openface3Settings={openface3Settings}
                  onTimeUpdate={handleTimeUpdate}
                  onDurationChange={setDuration}
                  onPlayStateChange={setIsPlaying}
                />
              </div>

              {/* Video Controls */}
              <div className="flex-shrink-0 p-2 border-b border-border">
                <VideoControls
                  isPlaying={isPlaying}
                  currentTime={currentTime}
                  duration={duration}
                  playbackRate={playbackRate}
                  frameRate={annotationData?.video_info?.frame_rate || 30}
                  onPlayPause={handlePlayPause}
                  onSeek={handleSeek}
                  onFrameStep={handleFrameStep}
                  onPlaybackRateChange={handlePlaybackRateChange}
                />
              </div>

              {/* Timeline Section */}
              <div className="flex-shrink-0 h-40 bg-video-timeline border-b border-border">
                <Timeline
                  annotationData={annotationData}
                  currentTime={currentTime}
                  duration={duration}
                  settings={timelineSettings}
                  onSeek={handleSeek}
                />
              </div>

              {/* Footer - compact version */}
              <div className="flex-shrink-0">
                <Footer />
              </div>
            </div>

            {/* Column 2: Unified Controls */}
            <div className="w-96 bg-card border-l border-border flex-shrink-0">
              <div className="p-4 h-full overflow-y-auto space-y-6">
                <UnifiedControls
                  overlaySettings={overlaySettings}
                  timelineSettings={timelineSettings}
                  onOverlayChange={setOverlaySettings}
                  onTimelineChange={setTimelineSettings}
                  annotationData={annotationData}
                />
                
                {/* OpenFace3 Controls */}
                <OpenFace3Controls
                  settings={openface3Settings}
                  onChange={setOpenface3Settings}
                  faceData={annotationData?.openface3_faces}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Debug Panel */}
      <DebugPanel 
        isOpen={showDebugPanel}
        onClose={() => setShowDebugPanel(false)}
      />
    </div>
  );
};