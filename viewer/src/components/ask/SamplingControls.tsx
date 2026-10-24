import { useState } from 'react';
import { AlertTriangle } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { SAMPLING_DOCS_URL } from '@/lib/ask';
import { formatEstimate, type Estimate } from '@/lib/estimate';
import { formatClock } from '@/lib/lanes';
import { parseMoments } from '@/lib/sampling';
import type { AskSampling } from '@/types/ask';

interface SamplingControlsProps {
  sampling: AskSampling;
  onChange: (sampling: AskSampling) => void;
  count: number;
  estimate: Estimate;
  /** Whether "A short clip" can be chosen, and why not (spec 033 FR-027). */
  video?: { offered: boolean; reason?: string };
}

const DEFAULT_CLIP = { length: 2, fps: 4 };

const clockText = (t: number | null | undefined) => (t == null ? '' : formatClock(t));

/**
 * Where the run will look, drawn on the timeline as you change it (FR-010,
 * FR-011). Text fields start from `sampling`: give a new `key` to show another
 * entry's.
 */
export function SamplingControls({ sampling, onChange, count, estimate, video }: SamplingControlsProps) {
  const listed = sampling.moments != null;
  const frames = sampling.frames ?? 1;
  const clip = sampling.clip ?? null;
  const [momentsText, setMomentsText] = useState(() => (sampling.moments ?? []).map(formatClock).join(', '));
  const [fromText, setFromText] = useState(() => clockText(sampling.start));
  const [toText, setToText] = useState(() => clockText(sampling.end));

  const set = (patch: Partial<AskSampling>) => onChange({ ...sampling, ...patch });

  return (
    <div className="space-y-3" data-testid="sampling-controls">
      <div className="flex items-center justify-between gap-2">
        <Label htmlFor="ask-listed" className="text-sm">
          Only at these moments
        </Label>
        <Switch
          id="ask-listed"
          checked={listed}
          onCheckedChange={(on) => set({ moments: on ? parseMoments(momentsText) : null })}
        />
      </div>

      {listed ? (
        <Input
          aria-label="Moments"
          placeholder="0:12, 0:47, 1:30"
          value={momentsText}
          onChange={(e) => {
            setMomentsText(e.target.value);
            set({ moments: parseMoments(e.target.value) });
          }}
        />
      ) : (
        <div className="grid grid-cols-3 gap-2">
          <div className="space-y-1">
            <Label htmlFor="ask-interval" className="text-xs">
              Every (s)
            </Label>
            <Input
              id="ask-interval"
              type="number"
              min={0.1}
              step={0.5}
              value={sampling.interval ?? 5}
              onChange={(e) => {
                const v = Number(e.target.value);
                if (v > 0) set({ interval: v });
              }}
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="ask-from" className="text-xs">
              From
            </Label>
            <Input
              id="ask-from"
              placeholder="0:00"
              value={fromText}
              onChange={(e) => {
                setFromText(e.target.value);
                set({ start: parseMoments(e.target.value)[0] ?? 0 });
              }}
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="ask-to" className="text-xs">
              To
            </Label>
            <Input
              id="ask-to"
              placeholder="end"
              value={toText}
              onChange={(e) => {
                setToText(e.target.value);
                set({ end: parseMoments(e.target.value)[0] ?? null });
              }}
            />
          </div>
        </div>
      )}

      <div className="grid grid-cols-2 gap-2">
        <div className="space-y-1">
          <Label htmlFor="ask-frames" className="text-xs">
            Frames per sample
          </Label>
          <select
            id="ask-frames"
            className="h-9 w-full rounded-md border border-input bg-background px-2 text-sm"
            value={clip ? 'clip' : frames}
            onChange={(e) =>
              e.target.value === 'clip'
                ? set({ clip: DEFAULT_CLIP, frames: 1 })
                : set({ frames: Number(e.target.value), clip: null })
            }
          >
            {[1, 2, 3, 4, 5, 6, 7, 8, 9].map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
            {video && (
              <option value="clip" disabled={!video.offered}>
                {video.offered ? 'A short clip' : `A short clip (${video.reason ?? 'not available'})`}
              </option>
            )}
          </select>
        </div>
        {clip && (
          <div className="space-y-1">
            <Label htmlFor="ask-clip-length" className="text-xs">
              Clip length (s)
            </Label>
            <Input
              id="ask-clip-length"
              type="number"
              min={0.5}
              max={30}
              step={0.5}
              value={clip.length}
              onChange={(e) => {
                const v = Number(e.target.value);
                if (v > 0) set({ clip: { ...clip, length: v } });
              }}
            />
          </div>
        )}
        {!clip && frames > 1 && (
          <div className="space-y-1">
            <Label htmlFor="ask-spacing" className="text-xs">
              Seconds apart
            </Label>
            <Input
              id="ask-spacing"
              type="number"
              min={0.04}
              step={0.1}
              value={sampling.spacing ?? sampling.interval ?? 5}
              onChange={(e) => {
                const v = Number(e.target.value);
                if (v > 0) set({ spacing: v });
              }}
            />
          </div>
        )}
      </div>

      {video && !video.offered && video.reason && (
        <p className="text-xs text-muted-foreground" data-testid="clip-unavailable">
          A short clip: {video.reason}
        </p>
      )}
      <p className="text-xs text-muted-foreground">
        {clip
          ? `Each sample sends ${clip.length} s of video around it (${clip.fps} frames a second) to the model, which answers your question about that clip on its own, without seeing the others.`
          : `Each sample sends ${frames === 1 ? 'one frame' : `${frames} frames together`} to the model, which answers your question about ${frames === 1 ? 'it' : 'them'} on its own, without seeing the others.`}{' '}
        <a className="underline" href={SAMPLING_DOCS_URL} target="_blank" rel="noreferrer">
          More
        </a>
      </p>

      <p className="text-sm" data-testid="sample-count">
        {count} sample{count === 1 ? '' : 's'} · {formatEstimate(estimate.seconds)}
        {estimate.rough && <span className="text-muted-foreground"> (rough guess until you try it)</span>}
      </p>
      {estimate.overHour && (
        <p className="flex items-center gap-1 text-xs text-amber-700 dark:text-amber-400">
          <AlertTriangle className="h-3 w-3" /> That's over an hour. Fewer samples, or one frame each, would be quicker.
        </p>
      )}
    </div>
  );
}
