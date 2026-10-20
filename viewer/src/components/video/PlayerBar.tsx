import { Pause, Play, SkipBack, SkipForward } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { formatClock, formatClockTenths } from '@/lib/lanes';

const RATES = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2];

interface PlayerBarProps {
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  playbackRate: number;
  /** From the results (frame over time); the frame number is left out when unknown. */
  frameRate: number | null;
  /** Name of the lane ⏮ ⏭ step through; they're off without one. */
  selectedLane: string | null;
  onPlayPause: () => void;
  onPreviousEvent: () => void;
  onNextEvent: () => void;
  onPlaybackRateChange: (rate: number) => void;
  onSeek: (time: number) => void;
}

/** Under the video: play, the readout, a seek bar, event stepping and speed, in one row (spec 028 FR-006, T020). */
export function PlayerBar({
  isPlaying,
  currentTime,
  duration,
  playbackRate,
  frameRate,
  selectedLane,
  onPlayPause,
  onPreviousEvent,
  onNextEvent,
  onPlaybackRateChange,
  onSeek,
}: PlayerBarProps) {
  const eventTitle = (which: string, key: string) =>
    selectedLane ? `${which} event in ${selectedLane} (${key})` : 'Select a track to step through its events';
  return (
    <div className="flex w-full items-center gap-1" data-testid="player-bar">
      <Button size="sm" className="h-7 w-8 px-0" onClick={onPlayPause} aria-label={isPlaying ? 'Pause' : 'Play'} title="Play or pause (Space)">
        {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
      </Button>
      <span className="whitespace-nowrap px-1 font-mono text-xs" data-testid="readout">
        {formatClockTenths(currentTime)} / {formatClock(duration)}
      </span>
      {frameRate !== null && (
        <span className="whitespace-nowrap px-1 font-mono text-xs text-muted-foreground" data-testid="frame-number" title={`${frameRate} frames per second, from the results`}>
          frame {Math.round(currentTime * frameRate).toLocaleString()}
        </span>
      )}
      <Slider
        className="mx-2 min-w-24 flex-1"
        min={0}
        max={duration > 0 ? duration : 1}
        step={0.1}
        value={[Math.min(currentTime, duration > 0 ? duration : 1)]}
        onValueChange={([t]) => onSeek(t)}
        disabled={!(duration > 0)}
        aria-label="Seek"
      />
      <Button
        variant="ghost"
        size="icon"
        className="h-7 w-7"
        onClick={onPreviousEvent}
        disabled={!selectedLane}
        aria-label="Previous event"
        title={eventTitle('Previous', '[')}
      >
        <SkipBack className="h-3.5 w-3.5" />
      </Button>
      <Button
        variant="ghost"
        size="icon"
        className="h-7 w-7"
        onClick={onNextEvent}
        disabled={!selectedLane}
        aria-label="Next event"
        title={eventTitle('Next', ']')}
      >
        <SkipForward className="h-3.5 w-3.5" />
      </Button>
      <Select value={String(playbackRate)} onValueChange={(v) => onPlaybackRateChange(Number(v))}>
        <SelectTrigger className="h-7 w-16 text-xs" aria-label="Playback speed">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {RATES.map((rate) => (
            <SelectItem key={rate} value={String(rate)}>
              {rate}×
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
