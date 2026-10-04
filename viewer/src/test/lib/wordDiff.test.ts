import { describe, expect, it } from 'vitest';
import { whitespaceOnly, wordDiff } from '@/lib/wordDiff';

describe('wordDiff', () => {
  it('marks the changed words and keeps the rest', () => {
    expect(wordDiff('Is the adult touching the infant?', 'Is the parent touching the baby?')).toEqual([
      { kind: 'same', text: 'Is the ' },
      { kind: 'removed', text: 'adult' },
      { kind: 'added', text: 'parent' },
      { kind: 'same', text: ' touching the ' },
      { kind: 'removed', text: 'infant?' },
      { kind: 'added', text: 'baby?' },
    ]);
  });

  it('reconstructs both texts', () => {
    const a = 'Answer TOUCH or NO_TOUCH.\nThen explain.';
    const b = 'Answer TOUCH, NEAR or NO_TOUCH.\n\nThen explain briefly.';
    const parts = wordDiff(a, b);
    expect(parts.filter((p) => p.kind !== 'added').map((p) => p.text).join('')).toBe(a);
    expect(parts.filter((p) => p.kind !== 'removed').map((p) => p.text).join('')).toBe(b);
  });

  it('names whitespace-only differences', () => {
    expect(whitespaceOnly('a b', 'a  b\n')).toBe(true);
    expect(whitespaceOnly('a b', 'a c')).toBe(false);
    expect(whitespaceOnly('a b', 'a b')).toBe(false);
  });
});
