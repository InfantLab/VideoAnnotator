/**
 * Re-finding a saved dataset's files in this browser (spec 018). Where the
 * browser supports it, the folder a dataset was saved from is remembered here
 * (never sent to the server), so reusing the dataset only asks for permission
 * again; otherwise the user picks the folder again.
 */
import { idbDel, idbGet, idbSet } from '@/lib/persistence/idbKv';
import { ensurePermission } from '@/lib/localLibrary/libraryStore';
import type { Candidate } from '@/lib/datasetMatch';

const key = (datasetId: string) => `datasets.handle.${datasetId}`;

const VIDEO_EXTENSIONS = ['.mp4', '.webm', '.avi', '.mov', '.mkv', '.m4v'];

export function supportsFolderHandles(): boolean {
  return typeof window !== 'undefined' && 'showDirectoryPicker' in window;
}

export async function rememberFolder(datasetId: string, handle: FileSystemDirectoryHandle): Promise<void> {
  try {
    await idbSet(key(datasetId), handle);
  } catch {
    // Not remembered: the user re-picks the folder next time.
  }
}

export async function forgetFolder(datasetId: string): Promise<void> {
  try {
    await idbDel(key(datasetId));
  } catch {
    // Nothing to forget.
  }
}

/** The dataset's folder, if remembered and readable. Prompts, so call from a click. */
export async function rememberedFolder(datasetId: string): Promise<FileSystemDirectoryHandle | null> {
  try {
    const handle = await idbGet<FileSystemDirectoryHandle>(key(datasetId));
    if (handle && (await ensurePermission(handle, 'read'))) return handle;
  } catch {
    // Fall through to re-picking.
  }
  return null;
}

type IterableDirectory = FileSystemDirectoryHandle & {
  entries(): AsyncIterable<[string, FileSystemHandle]>;
};

/** Every video under `handle`, with its path within it. */
export async function videosIn(handle: FileSystemDirectoryHandle, prefix = ''): Promise<Candidate<File>[]> {
  const found: Candidate<File>[] = [];
  for await (const [name, entry] of (handle as IterableDirectory).entries()) {
    const path = prefix ? `${prefix}/${name}` : name;
    if (entry.kind === 'directory') {
      found.push(...(await videosIn(entry as FileSystemDirectoryHandle, path)));
    } else if (VIDEO_EXTENSIONS.some((ext) => name.toLowerCase().endsWith(ext))) {
      const file = await (entry as FileSystemFileHandle).getFile();
      found.push({ name, size: file.size, relativePath: path, item: file });
    }
  }
  return found;
}
