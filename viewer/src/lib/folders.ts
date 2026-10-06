import type { FolderRef } from '@/types/ingest';

/** `path` as the researcher's own machine shows it (differs under Docker). */
export function displayPathOf(path: string, folders: FolderRef[]): string {
  for (const folder of folders) {
    if (path === folder.path || path.startsWith(`${folder.path}/`)) {
      return folder.display_path + path.slice(folder.path.length);
    }
  }
  return path;
}
