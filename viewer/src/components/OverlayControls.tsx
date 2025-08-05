import { OverlaySettings } from '@/types/annotations';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';

interface OverlayControlsProps {
  settings: OverlaySettings;
  onChange: (settings: OverlaySettings) => void;
}

export const OverlayControls = ({ settings, onChange }: OverlayControlsProps) => {
  const handleToggle = (key: keyof OverlaySettings) => {
    onChange({
      ...settings,
      [key]: !settings[key]
    });
  };

  const overlayOptions = [
    { key: 'pose' as const, label: 'Person Tracking (COCO)', color: 'annotation-pose' },
    { key: 'subtitles' as const, label: 'Speech Recognition (WebVTT)', color: 'annotation-subtitle' },
    { key: 'speakers' as const, label: 'Speaker Diarization (RTTM)', color: 'annotation-audio' },
    { key: 'scenes' as const, label: 'Scene Detection', color: 'annotation-event' },
    { key: 'faces' as const, label: 'Face Detection (LAION)', color: 'annotation-face' },
    { key: 'emotions' as const, label: 'Emotion Recognition', color: 'annotation-emotion' },
  ];

  return (
    <div className="space-y-4">
      {overlayOptions.map(({ key, label, color }) => (
        <div key={key} className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div
              className={`w-3 h-3 rounded-full bg-${color}`}
              style={{ backgroundColor: `hsl(var(--${color.replace('-', '-')}))` }}
            />
            <Label htmlFor={key} className="text-sm font-medium">
              {label}
            </Label>
          </div>
          <Switch
            id={key}
            checked={settings[key]}
            onCheckedChange={() => handleToggle(key)}
          />
        </div>
      ))}
    </div>
  );
};