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
  const handlePrecisionSeek = (direction: 'forward' | 'backward', precision: 'frame' | 'second' | 'tenth') => {
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
      default:
        step = 1 / frameRate;
    }
    
    const newTime = direction === 'forward' 
      ? Math.min(currentTime + step, duration)
      : Math.max(currentTime - step, 0);
    onSeek(newTime);
  };

  const handleSliderChange = (value: number[]) => {
    onSeek(value[0]);
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
    <div className="bg-card rounded-lg p-4 space-y-4" tabIndex={0} onKeyDown={handleKeyDown}>
      {/* Enhanced Time Display with Frame Info */}
      <div className="flex items-center justify-between mb-2">
        <div className="text-sm space-y-1">
          <div className="font-mono font-medium">
            ⏱️ {formatTime(currentTime, true)} / {formatTime(duration, true)}
          </div>
          <div className="text-xs text-muted-foreground">
            🎥 Frame {getCurrentFrame().toLocaleString()} / {getTotalFrames().toLocaleString()} @ {frameRate}fps
          </div>
        </div>
        <div className="text-xs text-muted-foreground text-right">
          <div>Space: Play/Pause</div>
          <div>←→: Frame step | Shift+←→: Second step</div>
        </div>
      </div>

      {/* Main Controls */}
      <div className="flex items-center gap-4">
        {/* Precision Step Controls */}
        <div className="flex gap-1">
          <Button
            variant="outline"
            size="sm"
            onClick={() => handlePrecisionSeek('backward', 'second')}
            className="p-1 text-xs"
            title="Step back 1 second"
          >
            ⏪1s
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => handlePrecisionSeek('backward', 'tenth')}
            className="p-1 text-xs"
            title="Step back 0.1 second"
          >
            ⏪.1
          </Button>
        </div>

        {/* Frame Step Back */}
        <Button
          variant="outline"
          size="sm"
          onClick={() => onFrameStep('backward')}
          className="p-2"
          title="Previous frame"
        >
          <SkipBack className="w-4 h-4" />
        </Button>

        {/* Play/Pause */}
        <Button
          variant="default"
          size="lg"
          onClick={onPlayPause}
          className="p-3"
          title="Space: Play/Pause"
        >
          {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
        </Button>

        {/* Frame Step Forward */}
        <Button
          variant="outline"
          size="sm"
          onClick={() => onFrameStep('forward')}
          className="p-2"
          title="Next frame"
        >
          <SkipForward className="w-4 h-4" />
        </Button>

        {/* Precision Step Controls */}
        <div className="flex gap-1">
          <Button
            variant="outline"
            size="sm"
            onClick={() => handlePrecisionSeek('forward', 'tenth')}
            className="p-1 text-xs"
            title="Step forward 0.1 second"
          >
            .1⏩
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => handlePrecisionSeek('forward', 'second')}
            className="p-1 text-xs"
            title="Step forward 1 second"
          >
            1s⏩
          </Button>
        </div>

        <div className="flex-1" />

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

      {/* Enhanced Seek Slider with Frame Precision */}
      <div className="space-y-2">
        <div className="relative">
          <Slider
            value={[currentTime]}
            max={duration}
            step={1 / frameRate} // Frame-precise seeking
            onValueChange={handleSliderChange}
            className="w-full"
          />
          {/* Frame markers (every 10 seconds for readability) */}
          <div className="flex justify-between text-xs text-muted-foreground mt-1">
            <span>🎥 0</span>
            <span className="text-primary font-mono">
              Frame {getCurrentFrame()}
            </span>
            <span>🎥 {getTotalFrames()}</span>
          </div>
        </div>
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>0:00.00</span>
          <span className="text-primary">
            Progress: {((currentTime / duration) * 100).toFixed(1)}%
          </span>
          <span>{formatTime(duration, true)}</span>
        </div>
      </div>
    </div>
  );
};