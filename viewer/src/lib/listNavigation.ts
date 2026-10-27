/**
 * Where a video was opened from, and its neighbours there (spec 028 R12):
 * `from=run:<batch id>`, `from=project:<dataset id>`, `from=folder:<path>`, and
 * (spec 029) `from=all` and `from=unassigned` for the Videos page's lists.
 */

import type { VideosQuery } from '@/types/videos';

export type ListKind = 'run' | 'project' | 'folder' | 'all' | 'unassigned';

export interface ListContext {
  kind: ListKind;
  id: string;
}

export function parseFrom(from: string | null | undefined): ListContext | null {
  if (from === 'all' || from === 'unassigned') return { kind: from, id: '' };
  const match = /^(run|project|folder):(.+)$/.exec(from ?? '');
  return match ? { kind: match[1] as ListKind, id: match[2] } : null;
}

export const fromParam = (context: ListContext) =>
  context.kind === 'all' || context.kind === 'unassigned' ? context.kind : `${context.kind}:${context.id}`;

/** The Videos page's query for a list. */
export function videosQueryOf(context: ListContext): VideosQuery {
  switch (context.kind) {
    case 'project':
      return { project: context.id };
    case 'run':
      return { run: context.id };
    case 'folder':
      return { folder: context.id };
    case 'unassigned':
      return { notInProject: true };
    default:
      return {};
  }
}

/** The Videos page showing a list. */
export function videosHref(context: ListContext): string {
  const query = videosQueryOf(context);
  if (query.run) return `/runs/${encodeURIComponent(query.run)}`;
  const params = new URLSearchParams();
  if (query.project) params.set('project', query.project);
  if (query.folder) params.set('folder', query.folder);
  if (query.notInProject) params.set('not_in_project', '1');
  const search = params.toString();
  return search ? `/videos?${search}` : '/videos';
}

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
