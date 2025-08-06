import { useState } from 'react';
import { OverlaySettings, TimelineSettings, StandardAnnotationData } from '@/types/annotations';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { Link } from 'lucide-react';
import { FileViewer } from './FileViewer';

interface UnifiedControlsProps {
  overlaySettings: OverlaySettings;
  timelineSettings: TimelineSettings;
  onOverlayChange: (settings: OverlaySettings) => void;
  onTimelineChange: (settings: TimelineSettings) => void;
  annotationData?: StandardAnnotationData | null;
}

export const UnifiedControls = ({ 
  overlaySettings,
  timelineSettings,
  onOverlayChange,
  onTimelineChange,
  annotationData 
}: UnifiedControlsProps) => {
  const [isLocked, setIsLocked] = useState(false);

  // Map component keys to FileViewer tabs
  const getViewerTabForComponent = (component: string): string => {
    switch (component) {
      case 'person':
        return 'person';
      case 'face':
        return 'face';
      case 'speech':
        return 'speech';
      case 'speakers':
        return 'speakers';
      case 'scenes':
        return 'scenes';
      default:
        return 'summary';
    }
  };

  // Check data availability
  const isDataAvailable = (component: string): boolean => {
    if (!annotationData) return false;
    switch (component) {
      case 'person':
        return !!annotationData.person_tracking && annotationData.person_tracking.length > 0;
      case 'face':
        return !!annotationData.face_analysis && annotationData.face_analysis.length > 0;
      case 'speech':
        return !!annotationData.speech_recognition && annotationData.speech_recognition.length > 0;
      case 'speakers':
        return !!annotationData.speaker_diarization && annotationData.speaker_diarization.length > 0;
      case 'scenes':
        return !!annotationData.scene_detection && annotationData.scene_detection.length > 0;
      default:
        return false;
    }
  };

  const handleOverlayToggle = (key: keyof OverlaySettings) => {
    const newSettings = {
      ...overlaySettings,
      [key]: !overlaySettings[key]
    };
    onOverlayChange(newSettings);
    
    // Auto-sync timeline if locked
    if (isLocked) {
      syncTimelineWithOverlay(newSettings);
    }
  };

  const handleTimelineToggle = (key: keyof TimelineSettings) => {
    if (!isLocked) {
      onTimelineChange({
        ...timelineSettings,
        [key]: !timelineSettings[key]
      });
    }
  };

  const syncTimelineWithOverlay = (overlaySettings: OverlaySettings) => {
    const syncedSettings: TimelineSettings = {
      showMotion: overlaySettings.pose || false,
      showFaces: overlaySettings.faces || false,
      showEmotions: overlaySettings.emotions || false,
      showSubtitles: overlaySettings.subtitles || false,
      showSpeakers: overlaySettings.speakers || false,
      showScenes: overlaySettings.scenes || false,
    };
    onTimelineChange(syncedSettings);
  };

  const handleLockToggle = () => {
    const newLocked = !isLocked;
    setIsLocked(newLocked);
    
    if (newLocked) {
      syncTimelineWithOverlay(overlaySettings);
    }
  };

  const handleToggleAllOverlays = () => {
    const allEnabled = Object.values(overlaySettings).every(value => value === true);
    const newSettings = Object.keys(overlaySettings).reduce((acc, key) => {
      acc[key as keyof OverlaySettings] = !allEnabled;
      return acc;
    }, {} as OverlaySettings);
    onOverlayChange(newSettings);
    
    if (isLocked) {
      syncTimelineWithOverlay(newSettings);
    }
  };

  const handleToggleAllTimeline = () => {
    if (!isLocked) {
      const allEnabled = Object.values(timelineSettings).every(value => value === true);
      const newSettings = Object.keys(timelineSettings).reduce((acc, key) => {
        acc[key as keyof TimelineSettings] = !allEnabled;
        return acc;
      }, {} as TimelineSettings);
      onTimelineChange(newSettings);
    }
  };

  // Unified component configuration
  const components = [
    {
      id: 'person',
      title: '👤 Person Tracking',
      color: 'hsl(25, 95%, 53%)', // Orange
      label: 'COCO Pose Keypoints',
      description: '17-point skeleton overlay & timeline',
      overlayKey: 'pose' as keyof OverlaySettings,
      timelineKey: 'showMotion' as keyof TimelineSettings,
      available: isDataAvailable('person')
    },
    {
      id: 'face',
      title: '😊 Face Detection',
      color: 'hsl(142, 76%, 36%)', // Green
      label: 'LAION Face Analysis',
      description: 'Face bounding boxes & detection timeline',
      overlayKey: 'faces' as keyof OverlaySettings,
      timelineKey: 'showFaces' as keyof TimelineSettings,
      available: isDataAvailable('face')
    },
    {
      id: 'emotions',
      title: '🎭 Emotion Recognition',
      color: 'hsl(142, 76%, 36%)', // Green (same as face)
      label: 'Facial Emotions',
      description: 'Emotion labels & analysis timeline',
      overlayKey: 'emotions' as keyof OverlaySettings,
      timelineKey: 'showEmotions' as keyof TimelineSettings,
      available: isDataAvailable('face') // Uses same data as faces
    },
    {
      id: 'speech',
      title: '🔵 Speech Recognition',
      color: 'hsl(221, 83%, 53%)', // Blue
      label: 'WebVTT Transcription',
      description: 'Subtitle overlay & transcription timeline',
      overlayKey: 'subtitles' as keyof OverlaySettings,
      timelineKey: 'showSubtitles' as keyof TimelineSettings,
      available: isDataAvailable('speech')
    },
    {
      id: 'speakers',
      title: '🟣 Speaker Diarization',
      color: 'hsl(271, 81%, 56%)', // Purple
      label: 'RTTM Speaker Segments',
      description: 'Speaker identification & diarization timeline',
      overlayKey: 'speakers' as keyof OverlaySettings,
      timelineKey: 'showSpeakers' as keyof TimelineSettings,
      available: isDataAvailable('speakers')
    },
    {
      id: 'scenes',
      title: '🎬 Scene Detection',
      color: 'hsl(187, 100%, 42%)', // Teal
      label: 'Scene Boundaries',
      description: 'Scene transition markers & detection timeline',
      overlayKey: 'scenes' as keyof OverlaySettings,
      timelineKey: 'showScenes' as keyof TimelineSettings,
      available: isDataAvailable('scenes')
    }
  ];

  const anyAvailable = components.some(comp => comp.available);
  const allOverlaysEnabled = Object.values(overlaySettings).every(value => value === true);
  const allTimelinesEnabled = Object.values(timelineSettings).every(value => value === true);

  return (
    <Card className="p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">
          📊 Annotations Control
        </h3>
        <Button
          variant="ghost"
          size="sm"
          onClick={handleLockToggle}
          disabled={!anyAvailable}
          className="text-lg p-2"
          title={isLocked ? "Unlock timeline from overlays" : "Lock timeline to overlays"}
        >
          {isLocked ? '🔒' : '🔓'}
        </Button>
      </div>

      <Separator className="mb-4" />

      {/* Column Headers */}
      {isLocked ? (
        // Locked: Single column layout  
        <div className="grid grid-cols-9 gap-4 mb-3 text-xs font-medium text-muted-foreground">
          <div className="col-span-5">Component</div>
          <div className="col-span-2 text-center">JSON</div>
          <div className="col-span-2 text-center">
            <div>Toggle</div>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleToggleAllOverlays}
              disabled={!anyAvailable}
              className="text-xs h-4 px-2 mt-1"
            >
              {allOverlaysEnabled ? 'All On' : 'All Off'} 
            </Button>
          </div>
        </div>
      ) : (
        // Unlocked: Dual column layout
        <div className="grid grid-cols-12 gap-4 mb-3 text-xs font-medium text-muted-foreground">
          <div className="col-span-5">Component</div>
          <div className="col-span-2 text-center">JSON</div>
          <div className="col-span-2 text-center">
            <div>Video Overlay</div>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleToggleAllOverlays}
              disabled={!anyAvailable}
              className="text-xs h-4 px-2 mt-1"
            >
              {allOverlaysEnabled ? 'All On' : 'All Off'}
            </Button>
          </div>
          <div className="col-span-2 text-center">
            <div>Timeline Track</div>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleToggleAllTimeline}
              disabled={!anyAvailable || isLocked}
              className="text-xs h-4 px-2 mt-1"
            >
              {allTimelinesEnabled ? 'All On' : 'All Off'}
            </Button>
          </div>
          <div className="col-span-1"></div>
        </div>
      )}

      <Separator className="mb-4" />

      {/* Component Rows */}
      <div className="space-y-3">
        {components.map((component) => (
          isLocked ? (
            // Locked: Single column layout
            <div 
              key={component.id} 
              className={`grid grid-cols-9 gap-4 items-center p-3 rounded-lg border transition-colors ${
                component.available ? 'bg-card border-border' : 'bg-muted border-muted opacity-60'
              }`}
            >
              {/* Component Info */}
              <div className="col-span-5">
                <Label className={`text-sm font-medium block ${component.available ? 'text-foreground' : 'text-muted-foreground'}`}>
                  {component.title}
                </Label>
                <div className="text-xs text-muted-foreground">
                  {component.description}
                </div>
                {!component.available && (
                  <div className="text-xs text-muted-foreground">(No data)</div>
                )}
              </div>

              {/* JSON Viewer Button */}
              <div className="col-span-2 flex justify-center">
                {component.available && annotationData ? (
                  <FileViewer
                    annotationData={annotationData}
                    defaultTab={getViewerTabForComponent(component.id)}
                    trigger={
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-8 w-8 p-0 text-xs"
                        title="View JSON data"
                      >
                        📄
                      </Button>
                    }
                  />
                ) : (
                  <div className="h-8 w-8 flex items-center justify-center text-muted-foreground text-xs">
                    —
                  </div>
                )}
              </div>

              {/* Single Toggle (Overlay controls both) */}
              <div className="col-span-2 flex justify-center">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleOverlayToggle(component.overlayKey)}
                  disabled={!component.available}
                  className="h-8 w-8 p-0 rounded-full hover:scale-110 transition-transform"
                  title={`${overlaySettings[component.overlayKey] ? 'Disable' : 'Enable'} ${component.title}`}
                >
                  <div
                    className="w-6 h-6 rounded-full border-2 transition-all"
                    style={{ 
                      backgroundColor: component.available && overlaySettings[component.overlayKey] ? component.color : 'transparent',
                      borderColor: component.available ? component.color : '#ccc'
                    }}
                  />
                </Button>
              </div>
            </div>
          ) : (
            // Unlocked: Dual column layout
            <div 
              key={component.id} 
              className={`grid grid-cols-12 gap-4 items-center p-3 rounded-lg border transition-colors ${
                component.available ? 'bg-card border-border' : 'bg-muted border-muted opacity-60'
              }`}
            >
              {/* Component Info */}
              <div className="col-span-5">
                <Label className={`text-sm font-medium block ${component.available ? 'text-foreground' : 'text-muted-foreground'}`}>
                  {component.title}
                </Label>
                <div className="text-xs text-muted-foreground">
                  {component.description}
                </div>
                {!component.available && (
                  <div className="text-xs text-muted-foreground">(No data)</div>
                )}
              </div>

              {/* JSON Viewer Button */}
              <div className="col-span-2 flex justify-center">
                {component.available && annotationData ? (
                  <FileViewer
                    annotationData={annotationData}
                    defaultTab={getViewerTabForComponent(component.id)}
                    trigger={
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-8 w-8 p-0 text-xs"
                        title="View JSON data"
                      >
                        📄
                      </Button>
                    }
                  />
                ) : (
                  <div className="h-8 w-8 flex items-center justify-center text-muted-foreground text-xs">
                    —
                  </div>
                )}
              </div>

              {/* Overlay Toggle */}
              <div className="col-span-2 flex justify-center">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleOverlayToggle(component.overlayKey)}
                  disabled={!component.available}
                  className="h-8 w-8 p-0 rounded-full hover:scale-110 transition-transform"
                  title={`${overlaySettings[component.overlayKey] ? 'Disable' : 'Enable'} overlay for ${component.title}`}
                >
                  <div
                    className="w-6 h-6 rounded-full border-2 transition-all"
                    style={{ 
                      backgroundColor: component.available && overlaySettings[component.overlayKey] ? component.color : 'transparent',
                      borderColor: component.available ? component.color : '#ccc'
                    }}
                  />
                </Button>
              </div>

              {/* Timeline Toggle */}
              <div className="col-span-2 flex justify-center">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleTimelineToggle(component.timelineKey)}
                  disabled={!component.available || isLocked}
                  className="h-8 w-8 p-0 rounded-full hover:scale-110 transition-transform"
                  title={`${timelineSettings[component.timelineKey] ? 'Disable' : 'Enable'} timeline for ${component.title}`}
                >
                  <div
                    className="w-6 h-6 rounded-full border-2 transition-all"
                    style={{ 
                      backgroundColor: component.available && timelineSettings[component.timelineKey] ? component.color : 'transparent',
                      borderColor: component.available ? component.color : '#ccc'
                    }}
                  />
                </Button>
              </div>

              {/* Lock Indicator */}
              <div className="col-span-1 flex justify-center">
                {/* Empty space when unlocked */}
              </div>
            </div>
          )
        ))}
      </div>

      {!anyAvailable && (
        <div className="mt-4 p-4 bg-muted rounded-lg text-center">
          <p className="text-sm text-muted-foreground">
            🔍 No annotation data loaded
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            Upload video and annotation files to enable controls
          </p>
        </div>
      )}
    </Card>
  );
};