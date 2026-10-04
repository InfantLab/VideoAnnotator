import { describe, expect, it } from 'vitest';
import { queueLabel } from '@/lib/queuePosition';

describe('queueLabel', () => {
  it.each([
    [1, 'Next in queue'],
    [2, '2nd in queue'],
    [3, '3rd in queue'],
    [4, '4th in queue'],
    [11, '11th in queue'],
    [12, '12th in queue'],
    [21, '21st in queue'],
    [102, '102nd in queue'],
    [113, '113th in queue'],
  ])('position %i reads "%s"', (position, label) => {
    expect(queueLabel({ status: 'pending', queue_position: position })).toBe(label);
  });

  it('says nothing for a job that is not pending', () => {
    expect(queueLabel({ status: 'running', queue_position: 1 })).toBeNull();
  });

  it.each([undefined, null, 0, 1.5, '2'])('says nothing for a position of %p', (position) => {
    expect(queueLabel({ status: 'pending', queue_position: position })).toBeNull();
  });
});
