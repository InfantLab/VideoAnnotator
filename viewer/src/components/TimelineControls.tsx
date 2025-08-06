import { useState } from 'react';
import { TimelineSettings, StandardAnnotationData } from '@/types/annotations';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { Link } from 'lucide-react';
import { FileViewer } from './FileViewer';

interface TimelineControlsProps {
  settings: TimelineSettings;
  onChange: (settings: TimelineSettings) => void;
  annotationData?: StandardAnnotationData | null;
  overlaySettings?: any; // For sync functionality
}

export const TimelineControls = ({ 
  settings, 
  onChange, 
  annotationData,
  overlaySettings 
}: TimelineControlsProps) => {
  const [isLocked, setIsLocked] = useState(false);

  // Map timeline keys to FileViewer tabs (consistent with OverlayControls)
  const getViewerTabForTimelineKey = (key: keyof TimelineSettings): string => {
    switch (key) {
      case 'showMotion':
        return 'person';
      case 'showSubtitles':
        return 'speech';
      case 'showSpeakers':
        return 'speakers';
      case 'showScenes':
        return 'scenes';
      case 'showFaces':
      case 'showEmotions':
        return 'face'; // Fixed: matches OverlayControls mapping
      default:
        return 'summary';
    }
  };
  const handleToggle = (key: keyof TimelineSettings) => {
    if (!isLocked) {
      onChange({
        ...settings,
        [key]: !settings[key]
      });
    }
    // If locked, timeline changes are controlled by overlay changes
  };

  const handleLockToggle = () => {
    const newLocked = !isLocked;
    setIsLocked(newLocked);
    
    if (newLocked && overlaySettings) {
      // Sync timeline with overlay settings when locking
      handleSyncWithOverlays();
    }
  };

  const handleToggleAll = () => {
    const allEnabled = Object.values(settings).every(value => value === true);
    const newSettings = Object.keys(settings).reduce((acc, key) => {
      acc[key as keyof TimelineSettings] = !allEnabled;
      return acc;
    }, {} as TimelineSettings);
    onChange(newSettings);
  };

  const handleSyncWithOverlays = () => {
    if (!overlaySettings) return;
    
    // Map overlay settings to timeline settings
    const syncedSettings: TimelineSettings = {
      showSubtitles: overlaySettings.subtitles || false,
      showSpeakers: overlaySettings.speakers || false,
      showScenes: overlaySettings.scenes || false,
      showMotion: overlaySettings.pose || false,
      showFaces: overlaySettings.faces || false,
      showEmotions: overlaySettings.emotions || false,
    };
    
    onChange(syncedSettings);
  };

  // Check data availability for greying out options
  const isDataAvailable = (key: keyof TimelineSettings): boolean => {
    if (!annotationData) return false;
    switch (key) {
      case 'showMotion':
        return !!annotationData.person_tracking && annotationData.person_tracking.length > 0;
      case 'showSubtitles':
        return !!annotationData.speech_recognition && annotationData.speech_recognition.length > 0;
      case 'showSpeakers':
        return !!annotationData.speaker_diarization && annotationData.speaker_diarization.length > 0;
      case 'showScenes':
        return !!annotationData.scene_detection && annotationData.scene_detection.length > 0;
      case 'showFaces':
      case 'showEmotions':
        return !!annotationData.face_analysis && annotationData.face_analysis.length > 0;
      default:
        return false;
    }
  };

  // Enhanced timeline configuration (standardized with OverlayControls naming)
  const timelineGroups = [
    {
      title: '👤 Person Tracking',
      color: 'hsl(25, 95%, 53%)', // Orange
      items: [
        {
          key: 'showMotion' as const,
          label: 'COCO Pose Timeline',
          description: 'Motion intensity and person count',
          available: isDataAvailable('showMotion')
        }
      ]
    },
    {
      title: '😊 Face Analysis',
      color: 'hsl(142, 76%, 36%)', // Green
      items: [
        {
          key: 'showFaces' as const,
          label: 'Face Detection Timeline',
          description: 'LAION face count over time',
          available: isDataAvailable('showFaces')
        },
        {
          key: 'showEmotions' as const,
          label: 'Emotion Recognition Timeline',
          description: 'Dominant facial emotions over time',
          available: isDataAvailable('showEmotions')
        }
      ]
    },
    {
      title: '🔵 Speech Recognition',
      color: 'hsl(221, 83%, 53%)', // Blue
      items: [
        {
          key: 'showSubtitles' as const,
          label: 'Speech Transcription Timeline',
          description: 'WebVTT subtitle segments',
          available: isDataAvailable('showSubtitles')
        }
      ]
    },
    {
      title: '🟣 Speaker Diarization',
      color: 'hsl(271, 81%, 56%)', // Purple
      items: [
        {
          key: 'showSpeakers' as const,
          label: 'Speaker Segments Timeline',
          description: 'RTTM speaker identification',
          available: isDataAvailable('showSpeakers')
        }
      ]
    },
    {
      title: '🎬 Scene Detection',
      color: 'hsl(187, 100%, 42%)', // Teal
      items: [
        {
          key: 'showScenes' as const,
          label: 'Scene Boundaries Timeline',
          description: 'Scene transition markers',
          available: isDataAvailable('showScenes')
        }
      ]
    }
  ];

  const allEnabled = Object.values(settings).every(value => value === true);
  const anyAvailable = timelineGroups.some(group => 
    group.items.some(item => item.available)
  );

  return (
    <Card className="p-4">
      {/* Header with Toggle All and Sync Controls */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          📊 Timeline Controls
        </h3>
        <div className="flex gap-2">
          {overlaySettings && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleSyncWithOverlays}
              disabled={!anyAvailable}
              className="text-xs flex items-center gap-1"
              title="Sync with overlay settings"
            >
              <Link className="w-3 h-3" />
              Sync
            </Button>
          )}
          <Button
            variant="outline"
            size="sm"
            onClick={handleToggleAll}
            disabled={!anyAvailable || isLocked}
            className="text-xs"
          >
            {allEnabled ? '🔳 Disable All' : '☑️ Enable All'}
          </Button>
        </div>
      </div>

      <Separator className="mb-4" />

      {/* Hierarchical Timeline Groups */}
      <div className="space-y-4">
        {timelineGroups.map((group, groupIndex) => {
          const hasAvailableItems = group.items.some(item => item.available);
          
          return (
            <div key={groupIndex} className={`space-y-2 ${!hasAvailableItems ? 'opacity-50' : ''}`}>
              {/* Group Header */}
              <div className="flex items-center gap-2">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: group.color }}
                />
                <Label className="text-sm font-medium text-muted-foreground">
                  {group.title}
                </Label>
                {!hasAvailableItems && (
                  <span className="text-xs text-muted-foreground">(No data)</span>
                )}
              </div>

              {/* Group Items */}
              <div className="pl-5 space-y-2">
                {group.items.map((item) => (
                  <div key={item.key} className="flex items-center justify-between">
                    <div className="flex-1">
                      <Label 
                        htmlFor={item.key} 
                        className={`text-sm cursor-pointer ${
                          item.available ? 'text-foreground' : 'text-muted-foreground'
                        }`}
                      >
                        {item.label}
                      </Label>
                      <div className="text-xs text-muted-foreground">
                        {item.description}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      {item.available && annotationData && (
                        <FileViewer
                          annotationData={annotationData}
                          defaultTab={getViewerTabForTimelineKey(item.key)}
                          trigger={
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-6 w-6 p-0 text-xs"
                              title="View JSON data"
                            >
                              📄
                            </Button>
                          }
                        />
                      )}
                      <Switch
                        id={item.key}
                        checked={settings[item.key]}
                        onCheckedChange={() => handleToggle(item.key)}
                        disabled={!item.available || isLocked}
                        className="scale-75"
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {!anyAvailable && (
        <div className="mt-4 p-3 bg-muted rounded-lg text-center">
          <p className="text-sm text-muted-foreground">
            🔍 No timeline data available
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            Timeline tracks will appear when annotation data is loaded
          </p>
        </div>
      )}

      {/* Lock to Overlay Option */}
      {overlaySettings && anyAvailable && (
        <div className="mt-4 pt-4 border-t border-border">
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-sm font-medium">🔗 Lock to Overlays</Label>
              <p className="text-xs text-muted-foreground">
                Auto-sync timeline with overlay controls
              </p>
            </div>
            <Switch
              checked={isLocked}
              onCheckedChange={handleLockToggle}
              className="scale-75"
            />
          </div>
        </div>
      )}
    </Card>
  );
};