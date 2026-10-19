import { useEffect, useRef } from 'react';

export interface VideoKeyHandlers {
  playPause: () => void;
  /** Move the playhead by `seconds` (negative: back). */
  step: (seconds: number) => void;
  frameStep: (direction: 1 | -1) => void;
  previousEvent: () => void;
  nextEvent: () => void;
  zoomIn: () => void;
  zoomOut: () => void;
  fit: () => void;
  toggleDebug?: () => void;
}

/** Typing in these never moves the video (spec 028 edge case). */
export function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable || target.getAttribute('contenteditable') === 'true') return true;
  return ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName);
}

/**
 * The Video page's keys, anywhere on the page (spec 028 R8): every key the old
 * player had, plus `[`/`]` for the selected lane's previous/next event and
 * `+`/`−`/`0` to zoom.
 */
export function useVideoKeys(handlers: VideoKeyHandlers, enabled = true): void {
  const latest = useRef(handlers);
  latest.current = handlers;

  useEffect(() => {
    if (!enabled) return;
    const onKeyDown = (e: KeyboardEvent) => {
      const h = latest.current;
      if (e.ctrlKey && e.shiftKey && (e.key === 'D' || e.key === 'd')) {
        e.preventDefault();
        h.toggleDebug?.();
        return;
      }
      if (e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey || isTypingTarget(e.target)) return;
      const run = (action: () => void) => {
        e.preventDefault();
        action();
      };
      switch (e.key) {
        case ' ':
          return run(h.playPause);
        case 'ArrowLeft':
          return run(() => (e.shiftKey ? h.step(-1) : h.frameStep(-1)));
        case 'ArrowRight':
          return run(() => (e.shiftKey ? h.step(1) : h.frameStep(1)));
        case 'ArrowUp':
          return run(() => h.step(0.1));
        case 'ArrowDown':
          return run(() => h.step(-0.1));
        case '[':
          return run(h.previousEvent);
        case ']':
          return run(h.nextEvent);
        case '+':
        case '=':
          return run(h.zoomIn);
        case '-':
        case '−':
          return run(h.zoomOut);
        case '0':
          return run(h.fit);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [enabled]);
}
