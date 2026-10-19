import { useQuery } from '@tanstack/react-query';

import { apiClient } from '@/api/client';
import { joinPath, type ListContext, type ListItem } from '@/lib/listNavigation';

export interface ListInfo {
  /** "Peekaboo study", the folder's name, the run's name. */
  name: string;
  /** The list's own page, if it has one. */
  href: string | null;
  items: ListItem[];
}

const folderName = (path: string) => path.replace(/[\\/]+$/, '').split(/[\\/]/).pop() || path;

async function load(context: ListContext): Promise<ListInfo> {
  if (context.kind === 'run') {
    const [batch, jobs] = await Promise.all([
      apiClient.getBatch(context.id).catch(() => null),
      // The whole run, not a page of it, so "n of N" and the neighbours are right.
      apiClient.getJobs(1, 1000, { batchId: context.id }),
    ]);
    const name = batch?.batch_name || 'Run';
    return {
      name,
      href: `/batches/${encodeURIComponent(context.id)}`,
      items: jobs.jobs.map((job) => ({
        name: job.video_filename || (job.video_path ?? job.id).split(/[\\/]/).pop() || job.id,
        path: job.video_path ?? null,
        run: job.id,
        usable: job.status === 'completed' && !!job.video_path,
      })),
    };
  }
  if (context.kind === 'project') {
    const dataset = await apiClient.getDataset(context.id);
    const folder = dataset.server_folder;
    if (folder) {
      return {
        name: dataset.name,
        href: '/datasets',
        items: dataset.video_manifest.map((e) => ({
          name: e.filename,
          path: joinPath(folder, e.relative_path || e.filename),
          usable: true,
        })),
      };
    }
    const stored = await apiClient.getStoredVideos(context.id);
    return {
      name: dataset.name,
      href: '/datasets',
      items: stored.videos.map((v) => ({
        name: v.filename,
        path: null,
        // An uploaded video is wherever the job that stored it put it: asked for only when opened.
        resolve: v.job_id ? async () => (await apiClient.getJob(v.job_id!)).video_path ?? null : undefined,
        usable: !!v.job_id,
      })),
    };
  }
  const listing = await apiClient.browseServerFolders(context.id);
  return {
    name: folderName(context.id),
    href: null,
    items: [...listing.videos]
      .sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }))
      .map((v) => ({ name: v.name, path: v.path, usable: true })),
  };
}

/** The list a video was opened from (spec 028 US7). */
export function useListContext(context: ListContext | null) {
  return useQuery({
    queryKey: ['videoList', context?.kind, context?.id],
    queryFn: () => load(context!),
    enabled: !!context,
    staleTime: 60_000,
    retry: false,
  });
}
