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
  frameRate?: number;
}

export const VideoControls = ({
  isPlaying,
  currentTime,
  duration,
  playbackRate,
  onPlayPause,
  onSeek,
  onFrameStep,
  onPlaybackRateChange,
  frameRate = 30
}: VideoControlsProps) => {
  // Enhanced time formatting with 100ths precision
  const formatTime = (time: number, showPrecision = false) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    const hundredths = Math.floor((time % 1) * 100);
    
    if (showPrecision) {
      return `${minutes}:${seconds.toString().padStart(2, '0')}.${hundredths.toString().padStart(2, '0')}`;
    }
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  // Calculate frame numbers
  const getCurrentFrame = () => Math.floor(currentTime * frameRate) + 1;
  const getTotalFrames = () => Math.floor(duration * frameRate);

  // Enhanced seek with frame precision
  const handlePrecisionSeek = (direction: 'forward' | 'backward', precision: 'frame' | 'second' | 'tenth' | 'percent') => {
    let step: number;
    switch (precision) {
      case 'frame':
        step = 1 / frameRate;
        break;
      case 'tenth':
        step = 0.1;
        break;
      case 'second':
        step = 1;
        break;
      case 'percent':
        step = duration * 0.05; // 5% of total duration
        break;
      default:
        step = 1 / frameRate;
    }
    
    const newTime = direction === 'forward' 
      ? Math.min(currentTime + step, duration)
      : Math.max(currentTime - step, 0);
    onSeek(newTime);
  };

  const handleKeyDown = (event: React.KeyboardEvent) => {
    switch (event.key) {
      case ' ':
        event.preventDefault();
        onPlayPause();
        break;
      case 'ArrowLeft':
        event.preventDefault();
        if (event.shiftKey) {
          handlePrecisionSeek('backward', 'second');
        } else {
          onFrameStep('backward');
        }
        break;
      case 'ArrowRight':
        event.preventDefault();
        if (event.shiftKey) {
          handlePrecisionSeek('forward', 'second');
        } else {
          onFrameStep('forward');
        }
        break;
      case 'ArrowDown':
        event.preventDefault();
        handlePrecisionSeek('backward', 'tenth');
        break;
      case 'ArrowUp':
        event.preventDefault();
        handlePrecisionSeek('forward', 'tenth');
        break;
    }
  };

  const playbackRates = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2];

  return (
    <div className="bg-card rounded-lg p-3 space-y-2" tabIndex={0} onKeyDown={handleKeyDown}>
      {/* Compact Time Display */}
      <div className="flex items-center justify-between text-sm">
        <div className="font-mono font-medium text-primary">
          {formatTime(currentTime, true)} / {formatTime(duration, true)}
        </div>
        <div className="text-xs text-muted-foreground">
          Frame {getCurrentFrame().toLocaleString()} / {getTotalFrames().toLocaleString()} @ {frameRate}fps
        </div>
        <div className="text-xs text-muted-foreground">
          {((currentTime / duration) * 100).toFixed(1)}%
        </div>
      </div>

      {/* Compact Main Controls - Centered Layout */}
      <div className="flex items-center justify-center gap-2">
        {/* Left: Large jumps */}
        <div className="flex items-center gap-1">
          <Button
            variant="outline"
            size="sm"
            onClick={() => handlePrecisionSeek('backward', 'percent')}
            className="h-7 px-2 text-xs"
            title="Jump back 5%"
          >
            ⏪5%
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => handlePrecisionSeek('backward', 'second')}
            className="h-7 px-2 text-xs"
            title="Step back 1 second (Shift+←)"
          >
            ⏪1s
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => handlePrecisionSeek('backward', 'tenth')}
            className="h-7 px-2 text-xs"
            title="Step back 0.1 second (↓)"
          >
            ⏪.1
          </Button>
        </div>

        {/* Center: Frame Step + Play/Pause */}
        <div className="flex items-center gap-1">
          <Button
            variant="outline"
            size="sm"
            onClick={() => onFrameStep('backward')}
            className="h-8 px-2"
            title="Previous frame (←)"
          >
            <SkipBack className="w-3 h-3" />
          </Button>

          <Button
            variant="default"
            size="sm"
            onClick={onPlayPause}
            className="h-8 px-3"
            title="Play/Pause (Space)"
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={() => onFrameStep('forward')}
            className="h-8 px-2"
            title="Next frame (→)"
          >
            <SkipForward className="w-3 h-3" />
          </Button>
        </div>

        {/* Right: Precision controls + extras */}
        <div className="flex items-center gap-1">
          <Button
            variant="outline"
            size="sm"
            onClick={() => handlePrecisionSeek('forward', 'tenth')}
            className="h-7 px-2 text-xs"
            title="Step forward 0.1 second (↑)"
          >
            .1⏩
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => handlePrecisionSeek('forward', 'second')}
            className="h-7 px-2 text-xs"
            title="Step forward 1 second (Shift+→)"
          >
            1s⏩
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => handlePrecisionSeek('forward', 'percent')}
            className="h-7 px-2 text-xs"
            title="Jump forward 5%"
          >
            5%⏩
          </Button>
        </div>

        {/* Far Right: Speed + Volume */}
        <div className="flex items-center gap-1 ml-2">
          <Select
            value={playbackRate.toString()}
            onValueChange={(value) => onPlaybackRateChange(parseFloat(value))}
          >
            <SelectTrigger className="w-16 h-7 text-xs">
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

          <Button variant="ghost" size="sm" className="h-7 px-2">
            <Volume2 className="w-3 h-3" />
          </Button>
        </div>
      </div>

      {/* Keyboard Shortcuts Help - Updated */}
      <div className="text-xs text-muted-foreground text-center opacity-70">
        Space: Play/Pause | ←→: Frame | Shift+←→: 1s | ↑↓: 0.1s | 5%: Large jumps
      </div>
    </div>
  );
};