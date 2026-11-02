/** Tips on the Video page (UAT #70): one at a time, only for what's on screen. */
import { act, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { VideoTips } from '@/components/video/VideoTips';
import { TIPS_SEEN_KEY, markTipsSeen, tipsSeen } from '@/lib/videoTips';

const Page = ({ list = false, notes = false }) => (
  <div>
    {list && <nav data-testid="list-navigation">peekaboo · 3 of 48</nav>}
    <button data-tip="layers">Layers</button>
    <span data-testid="lane-label">Speech</span>
    {notes && <button data-tip="notes">Note</button>}
  </div>
);

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

const start = (onEnd = vi.fn()) => {
  render(<VideoTips active onEnd={onEnd} />);
  act(() => vi.advanceTimersByTime(700));
  return onEnd;
};

describe('Video page tips', () => {
  it('only offers tips for what is on screen', () => {
    render(<Page />);
    start();
    expect(screen.getByTestId('video-tip')).toHaveTextContent('1 of 2');
    expect(screen.getByTestId('video-tip')).toHaveTextContent('Choose which results show as lanes');
  });

  it('goes through them in order, and Done ends them', () => {
    render(<Page list notes />);
    const onEnd = start();
    const texts = [];
    for (let i = 0; i < 4; i++) {
      texts.push(screen.getByTestId('video-tip').textContent);
      act(() => screen.getByRole('button', { name: i < 3 ? 'Next tip' : 'Done' }).click());
    }
    expect(texts[0]).toContain('previous or next video');
    expect(texts[3]).toContain('Press N for a note');
    expect(onEnd).toHaveBeenCalledOnce();
  });

  it('Skip tips ends them at once', () => {
    render(<Page list />);
    const onEnd = start();
    act(() => screen.getByRole('button', { name: 'Skip tips' }).click());
    expect(onEnd).toHaveBeenCalledOnce();
  });

  it('is remembered in this browser once seen', () => {
    markTipsSeen();
    expect(localStorage.setItem).toHaveBeenCalledWith(TIPS_SEEN_KEY, '1');
    vi.mocked(localStorage.getItem).mockReturnValueOnce('1');
    expect(tipsSeen()).toBe(true);
  });
});
