import { describe, expect, it } from 'vitest';

import { categoryHues } from '@/lib/categoryColors';

describe('categoryHues', () => {
  it('gives values that share a first letter different hues', () => {
    const hues = categoryHues(['SPEAKER_00', 'SPEAKER_01', 'SPEAKER_00', 'SPEAKER_02']);
    expect(hues.size).toBe(3);
    expect(new Set(hues.values()).size).toBe(3);
  });

  it('orders numerically, so the same IDs get the same hues in any file', () => {
    const a = categoryHues(['SPEAKER_10', 'SPEAKER_2']);
    const b = categoryHues(['SPEAKER_2', 'SPEAKER_10', 'SPEAKER_2']);
    expect(a.get('SPEAKER_2')).toBe(0);
    expect(a).toEqual(b);
  });
});
