/** Ask AI answer colours (UAT #23, #50; Caspar 2026-10-27). */
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { answerColours, answerKey, assign, PALETTE_SIZE, readJson, similarity } from '@/lib/answerColours';

const answered = (label: string) => ({ label, fields: null, status: 'answered' as const, fit: null });

// The shared setup stubs localStorage out; these need one that remembers.
beforeEach(() => {
  const store = new Map<string, string>();
  vi.stubGlobal('localStorage', {
    getItem: (k: string) => store.get(k) ?? null,
    setItem: (k: string, v: string) => void store.set(k, v),
    removeItem: (k: string) => void store.delete(k),
    clear: () => store.clear(),
  });
});

describe('answer colours', () => {
  it('keeps an answer’s colour across tries and runs of one question', () => {
    const first = answerColours('Is the child smiling?', [answered('yes'), answered('no')]);
    const later = answerColours('Is the child smiling?', [answered('no'), answered('maybe'), answered('no'), answered('yes')]);
    expect(later.colour(answered('yes'))).toBe(first.colour(answered('yes')));
    expect(later.colour(answered('no'))).toBe(first.colour(answered('no')));
  });

  it('gives 16 answers 16 different colours, none red', () => {
    const labels = Array.from({ length: PALETTE_SIZE }, (_, i) => `answer ${String.fromCharCode(97 + i)}`);
    const { colour } = answerColours('q', labels.map(answered));
    const colours = labels.map((l) => colour(answered(l)));
    expect(new Set(colours).size).toBe(PALETTE_SIZE);
    for (const c of colours) {
      const hue = Number(/hsl\((\d+)/.exec(c)![1]);
      expect(hue > 20 && hue < 340).toBe(true);
    }
  });

  it('past 16, gives a new answer a shade of the colour of the one it reads most like', () => {
    const base = Object.fromEntries(
      ['playing peekaboo', 'reading a book', 'eating', 'crying', 'sleeping', 'waving', 'clapping', 'laughing', 'pointing', 'walking', 'sitting', 'standing', 'jumping', 'drawing', 'singing', 'dancing'].map((k, slot) => [k, { slot }]),
    );
    const next = assign(base, ['playing peek-a-boo']);
    expect(next['playing peek-a-boo'].slot).toBe(base['playing peekaboo'].slot);
    expect(next['playing peek-a-boo'].shade).toBe(0);
  });

  it('compares answers loosely: case, spaces, underscores and a final full stop', () => {
    expect(answerKey(answered('Not_touching.'))).toBe(answerKey(answered('not touching')));
    expect(similarity('happy', 'happy')).toBeCloseTo(1);
    expect(similarity('happy', 'sad')).toBeLessThan(0.3);
  });
});

describe('a JSON reply (UAT #27)', () => {
  it('is read as fields, in ```json fences or not', () => {
    expect(readJson('```json\n{"expression": "happy"}\n```')).toEqual({ expression: 'happy' });
    expect(readJson('{"adults": 2}')).toEqual({ adults: 2 });
    expect(readJson('happy')).toBeNull();
    expect(readJson('[1, 2]')).toBeNull();
  });

  it('is coloured by its first value', () => {
    expect(answerKey(answered('```json\n{"expression": "Happy"}\n```'))).toBe('happy');
  });
});
