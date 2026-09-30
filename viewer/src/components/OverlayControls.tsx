import { OverlaySettings, StandardAnnotationData } from '@/types/annotations';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { FileViewer } from './FileViewer';

interface OverlayControlsProps {
  settings: OverlaySettings;
  onChange: (settings: OverlaySettings) => void;
  annotationData?: StandardAnnotationData | null;
}

export const OverlayControls = ({ settings, onChange, annotationData }: OverlayControlsProps) => {
  const handleToggle = (key: keyof OverlaySettings) => {
    onChange({
      ...settings,
      [key]: !settings[key]
    });
  };

  const handleToggleAll = () => {
    const allEnabled = Object.values(settings).every(value => value === true);
    const newSettings = Object.keys(settings).reduce((acc, key) => {
      acc[key as keyof OverlaySettings] = !allEnabled;
      return acc;
    }, {} as OverlaySettings);
    onChange(newSettings);
  };

  const getViewerTabForKey = (key: keyof OverlaySettings): string => {
    switch (key) {
      case 'pose':
        return 'person';
      case 'faces':
      case 'emotions':
        return 'face';
      case 'subtitles':
        return 'speech';
      case 'speakers':
        return 'speakers';
      case 'scenes':
        return 'scenes';
      default:
        return 'summary';
    }
  };

  // Check data availability for greying out options
  const isDataAvailable = (key: keyof OverlaySettings): boolean => {
    if (!annotationData) return false;
    switch (key) {
      case 'pose':
        return !!annotationData.person_tracking && annotationData.person_tracking.length > 0;
      case 'subtitles':
        return !!annotationData.speech_recognition && annotationData.speech_recognition.length > 0;
      case 'speakers':
        return !!annotationData.speaker_diarization && annotationData.speaker_diarization.length > 0;
      case 'scenes':
        return !!annotationData.scene_detection && annotationData.scene_detection.length > 0;
      case 'faces':
      case 'emotions':
        return !!annotationData.face_analysis && annotationData.face_analysis.length > 0;
      default:
        return false;
    }
  };

  // Enhanced overlay configuration with emojis and color coding per v0.2.0 spec
  const overlayGroups = [
    {
      title: '👤 Person Tracking',
      color: 'hsl(25, 95%, 53%)', // Orange
      items: [
        {
          key: 'pose' as const,
          label: 'COCO Pose Keypoints',
          description: 'Show 17-point skeleton overlay',
          available: isDataAvailable('pose')
        }
      ]
    },
    {
      title: '😊 Face Analysis',
      color: 'hsl(142, 76%, 36%)', // Green
      items: [
        {
          key: 'faces' as const,
          label: 'Face Detection Boxes',
          description: 'LAION face bounding boxes',
          available: isDataAvailable('faces')
        },
        {
          key: 'emotions' as const,
          label: 'Emotion Recognition',
          description: 'Facial emotion labels',
          available: isDataAvailable('emotions')
        }
      ]
    },
    {
      title: '🔵 Speech Recognition',
      color: 'hsl(221, 83%, 53%)', // Blue
      items: [
        {
          key: 'subtitles' as const,
          label: 'Speech Transcription',
          description: 'WebVTT subtitle overlay',
          available: isDataAvailable('subtitles')
        }
      ]
    },
    {
      title: '🟣 Speaker Diarization',
      color: 'hsl(271, 81%, 56%)', // Purple
      items: [
        {
          key: 'speakers' as const,
          label: 'Speaker Segments',
          description: 'RTTM speaker identification',
          available: isDataAvailable('speakers')
        }
      ]
    },
    {
      title: '🎬 Scene Detection',
      color: 'hsl(187, 100%, 42%)', // Teal
      items: [
        {
          key: 'scenes' as const,
          label: 'Scene Boundaries',
          description: 'Scene transition markers',
          available: isDataAvailable('scenes')
        }
      ]
    }
  ];

  const allEnabled = Object.values(settings).every(value => value === true);
  const anyAvailable = overlayGroups.some(group => 
    group.items.some(item => item.available)
  );

  return (
    <Card className="p-4">
      {/* Header with Toggle All */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          📹 Video Overlays
        </h3>
        <Button
          variant="outline"
          size="sm"
          onClick={handleToggleAll}
          disabled={!anyAvailable}
          className="text-xs"
        >
          {allEnabled ? '🔲 Disable All' : '☑️ Enable All'}
        </Button>
      </div>

      <Separator className="mb-4" />

      {/* Hierarchical Overlay Groups */}
      <div className="space-y-4">
        {overlayGroups.map((group, groupIndex) => {
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
                          defaultTab={getViewerTabForKey(item.key)}
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
                        disabled={!item.available}
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
            🔍 No annotation data loaded
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            Upload video and annotation files to enable overlays
          </p>
        </div>
      )}
    </Card>
  );
};