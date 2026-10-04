/**
 * Comparing a saved dataset's list of videos with the files available now
 * (spec 018). A dataset remembers names, sizes and paths, never the videos, so
 * reusing one means finding the files again; anything that differs is shown
 * before a run starts, never skipped silently.
 */
import type { ManifestEntry, ScannedVideo } from '@/types/datasets';

/** A candidate file, from the browser or a server folder scan. */
export interface Candidate<T> {
  name: string;
  size: number;
  relativePath: string | null;
  item: T;
}

export interface DatasetMatch<T> {
  matched: Array<{ entry: ManifestEntry; item: T }>;
  /** In the dataset, not found. */
  missing: ManifestEntry[];
  /** Found, not in the dataset. */
  added: T[];
  /** Same path (or name), different size. */
  changed: Array<{ entry: ManifestEntry; item: T }>;
  /** No path recorded and several files share the name: not guessed. */
  ambiguous: ManifestEntry[];
}

/** Where a browser file sits in the picked folder, without the folder's own name. */
export function relativePathOf(file: File): string | null {
  const path = (file as File & { webkitRelativePath?: string }).webkitRelativePath;
  if (!path) return null;
  const slash = path.indexOf('/');
  return slash >= 0 ? path.slice(slash + 1) : path;
}

export function fileCandidates(files: File[]): Candidate<File>[] {
  return files.map((file) => ({ name: file.name, size: file.size, relativePath: relativePathOf(file), item: file }));
}

export function scanCandidates(videos: ScannedVideo[]): Candidate<ScannedVideo>[] {
  return videos.map((v) => ({ name: v.name, size: v.size_bytes, relativePath: v.relative_path, item: v }));
}

export function manifestFrom<T>(candidates: Candidate<T>[]): ManifestEntry[] {
  return candidates.map((c) => ({ filename: c.name, size_bytes: c.size, relative_path: c.relativePath }));
}

export function matchDataset<T>(manifest: ManifestEntry[], candidates: Candidate<T>[]): DatasetMatch<T> {
  const result: DatasetMatch<T> = { matched: [], missing: [], added: [], changed: [], ambiguous: [] };
  const byPath = new Map<string, Candidate<T>>();
  const byName = new Map<string, Candidate<T>[]>();
  for (const c of candidates) {
    if (c.relativePath) byPath.set(c.relativePath, c);
    byName.set(c.name, [...(byName.get(c.name) ?? []), c]);
  }

  const used = new Set<Candidate<T>>();
  for (const entry of manifest) {
    let found: Candidate<T> | undefined;
    if (entry.relative_path) {
      found = byPath.get(entry.relative_path);
    } else {
      const sameName = (byName.get(entry.filename) ?? []).filter((c) => !used.has(c));
      if (sameName.length > 1) {
        const sameSize = sameName.filter((c) => c.size === entry.size_bytes);
        if (sameSize.length !== 1) {
          result.ambiguous.push(entry);
          continue;
        }
        found = sameSize[0];
      } else {
        found = sameName[0];
      }
    }
    if (!found || used.has(found)) {
      result.missing.push(entry);
      continue;
    }
    used.add(found);
    (found.size === entry.size_bytes ? result.matched : result.changed).push({ entry, item: found.item });
  }
  result.added = candidates.filter((c) => !used.has(c)).map((c) => c.item);
  return result;
}

export function hasDifferences(match: DatasetMatch<unknown>): boolean {
  return match.missing.length + match.added.length + match.changed.length + match.ambiguous.length > 0;
}
