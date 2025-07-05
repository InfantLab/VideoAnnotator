import { Play, Pause, SkipBack, SkipForward, Volume2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

interface VideoControlsProps {
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  playbackRate: number;
  onPlayPause: () => void;
  onSeek: (time: number) => void;
  onFrameStep: (direction: 'forward' | 'backward') => void;
  onPlaybackRateChange: (rate: number) => void;
}

export const VideoControls = ({
  isPlaying,
  currentTime,
  duration,
  playbackRate,
  onPlayPause,
  onSeek,
  onFrameStep,
  onPlaybackRateChange
}: VideoControlsProps) => {
  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  const handleSliderChange = (value: number[]) => {
    onSeek(value[0]);
  };

  const playbackRates = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2];

  return (
    <div className="bg-card rounded-lg p-4 space-y-4">
      {/* Main Controls */}
      <div className="flex items-center gap-4">
        {/* Frame Step Back */}
        <Button
          variant="outline"
          size="sm"
          onClick={() => onFrameStep('backward')}
          className="p-2"
        >
          <SkipBack className="w-4 h-4" />
        </Button>

        {/* Play/Pause */}
        <Button
          variant="default"
          size="sm"
          onClick={onPlayPause}
          className="p-2"
        >
          {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
        </Button>

        {/* Frame Step Forward */}
        <Button
          variant="outline"
          size="sm"
          onClick={() => onFrameStep('forward')}
          className="p-2"
        >
          <SkipForward className="w-4 h-4" />
        </Button>

        {/* Time Display */}
        <div className="flex items-center gap-2 text-sm font-mono">
          <span>{formatTime(currentTime)}</span>
          <span className="text-muted-foreground">/</span>
          <span className="text-muted-foreground">{formatTime(duration)}</span>
        </div>

        {/* Playback Rate */}
        <Select
          value={playbackRate.toString()}
          onValueChange={(value) => onPlaybackRateChange(parseFloat(value))}
        >
          <SelectTrigger className="w-20">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {playbackRates.map((rate) => (
              <SelectItem key={rate} value={rate.toString()}>
                {rate}×
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        {/* Volume (placeholder) */}
        <Button variant="ghost" size="sm" className="p-2">
          <Volume2 className="w-4 h-4" />
        </Button>
      </div>

      {/* Seek Slider */}
      <div className="space-y-2">
        <Slider
          value={[currentTime]}
          max={duration}
          step={0.1}
          onValueChange={handleSliderChange}
          className="w-full"
        />
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>0:00</span>
          <span>{formatTime(duration)}</span>
        </div>
      </div>
    </div>
  );
};