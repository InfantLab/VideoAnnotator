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
    { key: 'pose' as const, label: 'Pose Detection', color: 'annotation-pose' },
    { key: 'faceEmotion' as const, label: 'Face Emotions', color: 'annotation-face' },
    { key: 'audioEmotion' as const, label: 'Audio Emotions', color: 'annotation-audio' },
    { key: 'subtitles' as const, label: 'Subtitles', color: 'annotation-subtitle' },
    { key: 'events' as const, label: 'Events', color: 'annotation-event' },
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