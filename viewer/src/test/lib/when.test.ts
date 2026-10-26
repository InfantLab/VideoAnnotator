/** One way to say when (viewer design P8, UAT #2). */
import { describe, expect, it } from 'vitest';

import { formatDay, formatWhen } from '@/lib/when';

const now = new Date(2026, 9, 26, 21, 0);

describe('formatWhen', () => {
  it('says only the time today, drops the year this year, and keeps it otherwise', () => {
    expect(formatWhen(new Date(2026, 9, 26, 17, 50), now)).toBe('17:50');
    expect(formatWhen(new Date(2026, 9, 3, 14, 32), now)).toBe('3 Oct, 14:32');
    expect(formatWhen(new Date(2025, 9, 26, 9, 5), now)).toBe('26 Oct 2025, 09:05');
  });

  it('is empty without a date', () => {
    expect(formatWhen(null, now)).toBe('');
    expect(formatWhen('not a date', now)).toBe('');
    expect(formatDay(undefined, now)).toBe('');
  });
});
