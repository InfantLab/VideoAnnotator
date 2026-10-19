/**
 * Where a video was opened from, and its neighbours there (spec 028 R12):
 * `from=run:<batch id>`, `from=project:<dataset id>` or `from=folder:<path>`.
 */

export type ListKind = 'run' | 'project' | 'folder';

export interface ListContext {
  kind: ListKind;
  id: string;
}

export function parseFrom(from: string | null | undefined): ListContext | null {
  const match = /^(run|project|folder):(.+)$/.exec(from ?? '');
  return match ? { kind: match[1] as ListKind, id: match[2] } : null;
}

export const fromParam = (context: ListContext) => `${context.kind}:${context.id}`;

/** One video in the list, in the list's order. */
export interface ListItem {
  name: string;
  /** Its server path, when known without asking. */
  path: string | null;
  /** The job to open it on (a run's own job). */
  run?: string;
  /** Asks the server for the path (an uploaded project video's stored copy). */
  resolve?: () => Promise<string | null>;
  /** False for a video that can't be opened (no results in a run, no stored copy). */
  usable: boolean;
}

export interface ListPosition {
  /** 1-based. */
  position: number;
  total: number;
  previous: ListItem | null;
  next: ListItem | null;
}

const base = (path: string) => path.split(/[\\/]/).pop() ?? path;

/** This video's place in `items`, by path, else by file name; neighbours skip videos that can't open. */
export function positionIn(items: ListItem[], path: string, run?: string | null): ListPosition | null {
  let index = run ? items.findIndex((i) => i.run === run) : -1;
  if (index < 0) index = items.findIndex((i) => i.path === path);
  if (index < 0) index = items.findIndex((i) => i.name === base(path));
  if (index < 0) return null;
  return {
    position: index + 1,
    total: items.length,
    previous: items.slice(0, index).reverse().find((i) => i.usable) ?? null,
    next: items.slice(index + 1).find((i) => i.usable) ?? null,
  };
}

/** The address of `item` in the same list. */
export function neighbourLink(item: ListItem, path: string, context: ListContext): string {
  const params = new URLSearchParams({ path, from: fromParam(context) });
  if (item.run) params.set('run', item.run);
  return `/video?${params}`;
}

/** A dataset's videos on the server: in its folder, by manifest entry. */
export function joinPath(folder: string, relative: string): string {
  const sep = folder.includes('\\') && !folder.includes('/') ? '\\' : '/';
  return `${folder.replace(/[\\/]+$/, '')}${sep}${relative.replace(/^[\\/]+/, '')}`;
}
