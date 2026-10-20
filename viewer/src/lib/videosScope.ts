/**
 * The Videos page's address (spec 029 contracts/viewer.md):
 * `/videos[?project=<id> | folder=<path>[&recursive=1] | not_in_project=1 | run=<id>][&q=…]`.
 */
import { fromParam, type ListContext } from '@/lib/listNavigation';
import type { VideoItem, VideosQuery, VideoScope } from '@/types/videos';

export function queryOf(params: URLSearchParams): VideosQuery {
  const project = params.get('project');
  if (project) return { project };
  const run = params.get('run');
  if (run) return { run };
  const folder = params.get('folder');
  if (folder) return { folder, recursive: params.get('recursive') === '1' || params.get('recursive') === 'true' };
  if (params.get('not_in_project')) return { notInProject: true };
  return {};
}

export function paramsOf(query: VideosQuery): URLSearchParams {
  const params = new URLSearchParams();
  if (query.project) params.set('project', query.project);
  else if (query.run) params.set('run', query.run);
  else if (query.folder) {
    params.set('folder', query.folder);
    if (query.recursive) params.set('recursive', '1');
  } else if (query.notInProject) params.set('not_in_project', '1');
  return params;
}

export function hrefOf(query: VideosQuery): string {
  const search = paramsOf(query).toString();
  return search ? `/videos?${search}` : '/videos';
}

/** The list a row's Video page moves through (spec 028's `from=`). */
export function listContextOf(query: VideosQuery): ListContext {
  if (query.project) return { kind: 'project', id: query.project };
  if (query.run) return { kind: 'run', id: query.run };
  if (query.folder) return { kind: 'folder', id: query.folder };
  if (query.notInProject) return { kind: 'unassigned', id: '' };
  return { kind: 'all', id: '' };
}

export function sameScope(a: VideosQuery, b: VideosQuery): boolean {
  return paramsOf(a).toString() === paramsOf(b).toString();
}

/** What the header calls the scope. */
export function scopeTitle(scope: VideoScope | undefined, query: VideosQuery): string {
  if (query.notInProject) return 'Not in a project';
  if (query.folder === 'uploads') return 'Uploaded videos';
  if (scope?.name) return scope.name;
  if (query.folder) return query.folder;
  if (query.project) return 'Project';
  if (query.run) return 'Run';
  return 'All videos';
}

/** The pipelines with a column: every one that ran on a shown video, in a stable order. */
export function pipelinesOf(videos: VideoItem[], order: string[] = []): string[] {
  const seen = new Set<string>();
  for (const video of videos) for (const name of Object.keys(video.results)) seen.add(name);
  const known = order.filter((name) => seen.has(name));
  const rest = [...seen].filter((name) => !order.includes(name)).sort();
  return [...known, ...rest];
}

export const folderLabel = (folder: string) => (folder === 'uploads' ? 'Uploaded videos' : folder);

/** The last two parts of a long path: "…/pk/site_a". */
export function shortFolder(folder: string): string {
  if (folder === 'uploads') return 'Uploaded videos';
  const parts = folder.replace(/[\\/]+$/, '').split(/[\\/]/);
  return parts.length > 3 ? `…/${parts.slice(-2).join('/')}` : folder;
}

/** A row's Video page, moving through the same list. */
export function videoLink(video: VideoItem, context: ListContext): string {
  const params = new URLSearchParams({ path: video.path, from: fromParam(context) });
  if (video.run_job_id) params.set('run', video.run_job_id);
  return `/video?${params}`;
}
