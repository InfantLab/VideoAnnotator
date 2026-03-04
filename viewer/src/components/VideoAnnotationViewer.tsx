import React, { useState, useCallback, useRef, useEffect } from 'react';
import { VideoPlayer } from './VideoPlayer';
import { Timeline } from './Timeline';
import { UnifiedControls } from './UnifiedControls';
import { VideoControls } from './VideoControls';
import { FileViewer } from './FileViewer';
import { FileUploader } from './FileUploader';
import { Footer } from './Footer';
import { DebugPanel } from './DebugPanel';
import { OpenFace3Controls } from './OpenFace3Controls';
import { defaultOpenFace3Settings, type OpenFace3Settings } from './openface3Settings';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { StandardAnnotationData, OverlaySettings, TimelineSettings } from '@/types/annotations';
import { Link, useNavigate } from 'react-router-dom';
import { getJobDatasetIndex, type JobDatasetIndexEntry } from '@/lib/localLibrary/libraryStore';
import { getDemoLabel, DEMO_JOB_ID_PREFIX } from '@/lib/localLibrary/installDemoDataset';

interface VideoAnnotationViewerProps {
  /** Pipeline IDs that were used to generate the current annotation data */
  jobPipelines?: string[];
  /** Pre-loaded video file (optional) */
  initialVideoFile?: File | null;
  /** Pre-loaded annotation data (optional) */
  initialAnnotationData?: StandardAnnotationData | null;
  /** Label for the back button (default: "Home") */
  backLabel?: string;
  /** Path for the back button (default: "/") */
  backPath?: string;
}

export const VideoAnnotationViewer: React.FC<VideoAnnotationViewerProps> = ({
  jobPipelines = [],
  initialVideoFile = null,
  initialAnnotationData = null,
  backLabel = 'Home',
  backPath = '/',
}) => {
  const navigate = useNavigate();
  const [videoFile, setVideoFile] = useState<File | null>(initialVideoFile);
  const [annotationData, setAnnotationData] = useState<StandardAnnotationData | null>(initialAnnotationData);
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
  const [libraryDatasets, setLibraryDatasets] = useState<Array<{ jobId: string; entry: JobDatasetIndexEntry }>>([]);

  const videoRef = useRef<HTMLVideoElement>(null);

  // Fetch library datasets when in file-upload mode
  useEffect(() => {
    if (!videoFile || !annotationData) {
      getJobDatasetIndex().then(index => {
        const rows = Object.entries(index)
          .map(([jobId, entry]) => ({ jobId, entry }))
          .sort((a, b) => b.entry.createdAt.localeCompare(a.entry.createdAt));
        setLibraryDatasets(rows);
      }).catch(() => setLibraryDatasets([]));
    }
  }, [videoFile, annotationData]);

  const handleLoadNewFiles = useCallback(() => {
    setVideoFile(null);
    setAnnotationData(null);
    setCurrentTime(0);
    setIsPlaying(false);
    setDuration(0);
    setPlaybackRate(1);
  }, []);

  const handleVideoLoad = useCallback((file: File) => {
    setVideoFile(file);
  }, []);

  const handleAnnotationLoad = useCallback((data: StandardAnnotationData) => {
    setAnnotationData(data);
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

  const handleBack = useCallback(() => {
    navigate(backPath);
  }, [navigate, backPath]);

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

  // Show file uploader if no files loaded
  if (!videoFile || !annotationData) {
    return (
      <>
        <div className="min-h-screen bg-background p-6">
          <div className="max-w-4xl mx-auto">
            <div className="text-center mb-8">
              <h1 className="text-3xl font-bold mb-2">View Local Files</h1>
              <p className="text-muted-foreground">
                Drop or select a video file and annotation data (JSON, VTT, RTTM) to view them together.
              </p>
              <p className="text-sm text-muted-foreground mt-2">
                Looking for demo datasets or server jobs? Visit the{' '}
                <Link to="/library" className="text-primary underline underline-offset-2 hover:text-primary/80">Library</Link>
                {' '}or{' '}
                <Link to="/jobs" className="text-primary underline underline-offset-2 hover:text-primary/80">Jobs</Link>
                {' '}page.
              </p>
            </div>
            <FileUploader
              onVideoLoad={handleVideoLoad}
              onAnnotationLoad={handleAnnotationLoad}
            />

            {/* Library Datasets */}
            {libraryDatasets.length > 0 && (
              <Card className="p-5 mt-6">
                <h3 className="font-semibold mb-1">Your Library</h3>
                <p className="text-sm text-muted-foreground mb-3">
                  Open a previously saved dataset.
                </p>
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {libraryDatasets.map(({ jobId, entry }) => {
                    const label = getDemoLabel(jobId);
                    return (
                      <div
                        key={jobId}
                        className="flex items-center justify-between gap-3 border rounded-md p-3 hover:bg-muted/50 transition-colors"
                      >
                        <div className="min-w-0">
                          <div className="font-medium truncate text-sm">
                            {label ? `Demo: ${label}` : entry.videoFileName || `Job ${jobId}`}
                          </div>
                          {entry.createdAt && (
                            <div className="text-xs text-muted-foreground">
                              {new Date(entry.createdAt).toLocaleDateString()}
                            </div>
                          )}
                        </div>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => navigate(`/view/${jobId}`)}
                        >
                          Open
                        </Button>
                      </div>
                    );
                  })}
                </div>
              </Card>
            )}

            {/* Debug button for file uploader page */}
            <div className="fixed bottom-4 right-4">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowDebugPanel(true)}
                className="text-xs bg-background border"
                title="Debug Panel (Ctrl+Shift+D)"
              >
                Debug
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
        <div className="flex-shrink-0 px-4 py-2 border-b border-border">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3 min-w-0">
              <Button
                variant="ghost"
                size="sm"
                onClick={handleBack}
                className="flex items-center gap-2 flex-shrink-0"
                title={`Back to ${backLabel}`}
              >
                ← {backLabel}
              </Button>
              <div className="min-w-0">
                <h1 className="text-lg font-semibold truncate">
                  {annotationData.video_info?.filename || videoFile.name}
                </h1>
              </div>
            </div>
            <div className="flex items-center gap-2 flex-shrink-0">
              <Button
                variant="outline"
                size="sm"
                onClick={handleLoadNewFiles}
              >
                Load Different Files
              </Button>
              <FileViewer
                annotationData={annotationData}
                trigger={
                  <Button variant="outline" size="sm">
                    View Data
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
                Debug
              </Button>
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
                  jobPipelines={jobPipelines}
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