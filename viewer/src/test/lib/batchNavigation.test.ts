import { describe, expect, it } from 'vitest';

import { batchPosition } from '@/lib/batchNavigation';

const job = (id: string, status = 'completed') => ({ id, status });

describe('batchPosition', () => {
  const jobs = [job('a'), job('b', 'failed'), job('c'), job('d', 'running'), job('e')];

  it('counts every video of the batch, n of N', () => {
    expect(batchPosition(jobs, 'c')).toMatchObject({ position: 3, total: 5 });
  });

  it('steps over videos with no results to open', () => {
    const position = batchPosition(jobs, 'c')!;
    expect(position.previous?.id).toBe('a');
    expect(position.next?.id).toBe('e');
  });

  it('has nothing before the first or after the last', () => {
    expect(batchPosition(jobs, 'a')!.previous).toBeNull();
    expect(batchPosition(jobs, 'e')!.next).toBeNull();
  });

  it('is null for a job that is not in the batch', () => {
    expect(batchPosition(jobs, 'z')).toBeNull();
  });
});
