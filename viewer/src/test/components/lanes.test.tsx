import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { AnswerLane } from '@/components/lanes/AnswerLane';
import { SampleTicks } from '@/components/lanes/SampleTicks';
import { TimeAxis } from '@/components/lanes/TimeAxis';
import { categoryHues } from '@/lib/categoryColors';
import { axisTicks, formatClock, type LaneMarker } from '@/lib/lanes';

const markers: LaneMarker[] = [
  { t: 0, status: 'answered', label: 'TOUCH' },
  { t: 25, status: 'answered', label: 'NO_TOUCH' },
  { t: 50, status: 'error', label: null },
  { t: 75, status: 'skipped', label: null },
];

describe('lanes', () => {
  it('places markers by time', () => {
    render(<AnswerLane markers={markers} duration={100} hues={categoryHues(['TOUCH', 'NO_TOUCH'])} />);
    const xs = screen.getAllByTestId('answer-marker').map((m) => m.getAttribute('x'));
    expect(xs).toEqual(['0%', '25%', '50%', '75%']);
  });

  it('seeks to a marker when it is clicked', () => {
    const onSeek = vi.fn();
    render(<AnswerLane markers={markers} duration={100} hues={new Map()} onSeek={onSeek} />);
    fireEvent.click(screen.getAllByTestId('answer-marker')[1]);
    expect(onSeek).toHaveBeenCalledWith(25);
  });

  it('draws errors and skipped samples as their own shapes, not answers', () => {
    render(<AnswerLane markers={markers} duration={100} hues={new Map()} />);
    const [answered, , error, skipped] = screen.getAllByTestId('answer-marker');
    expect(answered.querySelector('circle')).not.toBeNull();
    expect(error.querySelector('circle')).toBeNull();
    expect(error.querySelector('path')).not.toBeNull();
    expect(skipped.querySelector('rect')).not.toBeNull();
    expect(error.getAttribute('aria-label')).toContain('the model failed');
    expect(skipped.getAttribute('aria-label')).toContain('skipped');
  });

  it('gives the same answer the same colour', () => {
    const twice: LaneMarker[] = [
      { t: 1, status: 'answered', label: 'TOUCH' },
      { t: 2, status: 'answered', label: 'TOUCH' },
      { t: 3, status: 'answered', label: 'NO_TOUCH' },
    ];
    render(<AnswerLane markers={twice} duration={10} hues={categoryHues(['TOUCH', 'NO_TOUCH'])} />);
    const fills = screen.getAllByTestId('answer-marker').map((m) => m.querySelector('circle')?.getAttribute('fill'));
    expect(fills[0]).toBe(fills[1]);
    expect(fills[0]).not.toBe(fills[2]);
  });

  it('marks the highlighted sample', () => {
    render(<SampleTicks times={[0, 5, 10]} duration={10} highlight={5} />);
    const ticks = screen.getAllByTestId('sample-tick');
    expect(ticks).toHaveLength(3);
    expect(ticks[1].getAttribute('stroke-width')).toBe('2.5');
  });

  it('labels the axis in minutes and seconds', () => {
    render(<TimeAxis duration={132} />);
    expect(screen.getByText('0:00')).toBeInTheDocument();
    expect(screen.getByText('2:00')).toBeInTheDocument();
    expect(axisTicks(132)).toEqual([0, 30, 60, 90, 120]);
    expect(formatClock(3725)).toBe('1:02:05');
  });
});
