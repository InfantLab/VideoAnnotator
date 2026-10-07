import { describe, expect, it } from 'vitest';
import { hasDifferences, manifestFrom, matchDataset, type Candidate } from '@/lib/datasetMatch';
import type { ManifestEntry } from '@/types/datasets';

const c = (name: string, size: number, relativePath: string | null = null): Candidate<string> => ({
  name,
  size,
  relativePath,
  item: relativePath ?? name,
});
const e = (filename: string, size_bytes: number, relative_path: string | null = null): ManifestEntry => ({
  filename,
  size_bytes,
  relative_path,
});

describe('matchDataset', () => {
  it('matches by relative path when recorded, and reports every kind of difference', () => {
    const manifest = [e('a.mp4', 1, 'p1/a.mp4'), e('a.mp4', 2, 'p2/a.mp4'), e('b.mp4', 3, 'b.mp4'), e('c.mp4', 4, 'c.mp4')];
    const found = [c('a.mp4', 1, 'p1/a.mp4'), c('a.mp4', 9, 'p2/a.mp4'), c('c.mp4', 4, 'c.mp4'), c('d.mp4', 5, 'd.mp4')];
    const m = matchDataset(manifest, found);
    expect(m.matched.map((x) => x.item)).toEqual(['p1/a.mp4', 'c.mp4']);
    expect(m.changed.map((x) => x.item)).toEqual(['p2/a.mp4']);
    expect(m.missing.map((x) => x.relative_path)).toEqual(['b.mp4']);
    expect(m.added).toEqual(['d.mp4']);
    expect(hasDifferences(m)).toBe(true);
  });

  it('falls back to names for datasets saved without paths', () => {
    const m = matchDataset([e('a.mp4', 1), e('b.mp4', 2)], [c('b.mp4', 2, 'sub/b.mp4'), c('a.mp4', 1, 'a.mp4')]);
    expect(m.matched).toHaveLength(2);
    expect(hasDifferences(m)).toBe(false);
  });

  it('does not guess between same-named files without a path', () => {
    const m = matchDataset([e('a.mp4', 1)], [c('a.mp4', 1, 'x/a.mp4'), c('a.mp4', 1, 'y/a.mp4')]);
    expect(m.ambiguous).toHaveLength(1);
    expect(m.matched).toHaveLength(0);
  });

  it('uses size to tell same-named files apart when that is enough', () => {
    const m = matchDataset([e('a.mp4', 1)], [c('a.mp4', 1, 'x/a.mp4'), c('a.mp4', 2, 'y/a.mp4')]);
    expect(m.matched.map((x) => x.item)).toEqual(['x/a.mp4']);
    expect(m.added).toEqual(['y/a.mp4']);
  });

  it('round-trips through manifestFrom', () => {
    const files = [c('a.mp4', 1, 'p/a.mp4'), c('b.mp4', 2, 'b.mp4')];
    expect(hasDifferences(matchDataset(manifestFrom(files), files))).toBe(false);
  });

  it('matches 5,000 files quickly', () => {
    const files = Array.from({ length: 5000 }, (_, i) => c(`v${i}.mp4`, i, `s${i % 50}/v${i}.mp4`));
    const start = performance.now();
    const m = matchDataset(manifestFrom(files), files);
    expect(m.matched).toHaveLength(5000);
    expect(performance.now() - start).toBeLessThan(2000);
  });
});
