import { useCallback, useEffect, useLayoutEffect, useState } from 'react';
import { createPortal } from 'react-dom';

import { Button } from '@/components/ui/button';
import { TIPS, type Tip } from '@/lib/videoTips';

const WIDTH = 320;
const GAP = 8;

/**
 * Tips on the Video page, one at a time, each pointing at what it's about
 * (UAT #70). Only for what is on screen: no list tip for a video opened on
 * its own, no notes tip without a server.
 */
export function VideoTips({ active, onEnd }: { active: boolean; onEnd: () => void }) {
  const [tips, setTips] = useState<Tip[] | null>(null);
  const [index, setIndex] = useState(0);
  const [rect, setRect] = useState<DOMRect | null>(null);

  useEffect(() => {
    if (!active) {
      setTips(null);
      return;
    }
    // Lanes and the list arrive after the page: look once they've had a moment.
    const timer = setTimeout(() => {
      const shown = TIPS.filter((tip) => document.querySelector(tip.target));
      if (!shown.length) onEnd();
      else {
        setTips(shown);
        setIndex(0);
      }
    }, 600);
    return () => clearTimeout(timer);
  }, [active, onEnd]);

  const tip = tips?.[index] ?? null;

  const place = useCallback(() => {
    const el = tip ? document.querySelector(tip.target) : null;
    setRect(el ? el.getBoundingClientRect() : null);
  }, [tip]);

  useLayoutEffect(() => {
    place();
    if (!tip) return;
    window.addEventListener('resize', place);
    window.addEventListener('scroll', place, true);
    return () => {
      window.removeEventListener('resize', place);
      window.removeEventListener('scroll', place, true);
    };
  }, [tip, place]);

  useEffect(() => {
    if (!tip) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onEnd();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [tip, onEnd]);

  if (!tips || !tip) return null;
  const last = index === tips.length - 1;
  const r = rect ?? new DOMRect(window.innerWidth / 2 - WIDTH / 2, 80, 0, 0);
  const below = r.bottom + GAP + 140 < window.innerHeight;
  const left = Math.min(Math.max(GAP, r.left), Math.max(GAP, window.innerWidth - WIDTH - GAP));

  return createPortal(
    <>
      {rect && (
        <div
          aria-hidden="true"
          className="pointer-events-none fixed z-[60] rounded-md ring-2 ring-primary ring-offset-2 ring-offset-background"
          style={{ top: r.top, left: r.left, width: r.width, height: r.height }}
        />
      )}
      <div
        role="dialog"
        aria-label={`Tip ${index + 1} of ${tips.length}`}
        data-testid="video-tip"
        className="fixed z-[61] rounded-md border bg-popover p-3 text-sm text-popover-foreground shadow-lg"
        style={{ width: WIDTH, left, ...(below ? { top: r.bottom + GAP } : { bottom: window.innerHeight - r.top + GAP }) }}
      >
        <p className="text-xs text-muted-foreground">
          {index + 1} of {tips.length}
        </p>
        <p className="mt-1">{tip.text}</p>
        <div className="mt-3 flex justify-end gap-2">
          {!last && (
            <Button variant="ghost" size="sm" className="h-7 text-xs" onClick={onEnd}>
              Skip tips
            </Button>
          )}
          <Button size="sm" className="h-7 text-xs" onClick={() => (last ? onEnd() : setIndex(index + 1))} autoFocus>
            {last ? 'Done' : 'Next tip'}
          </Button>
        </div>
      </div>
    </>,
    document.body,
  );
}
