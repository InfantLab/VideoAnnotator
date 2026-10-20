import { useQuery } from '@tanstack/react-query';

import { apiClient } from '@/api/client';
import { videosHref, videosQueryOf, type ListContext, type ListItem } from '@/lib/listNavigation';

export interface ListInfo {
  /** "Peekaboo study", the folder's name, the run's name. */
  name: string;
  /** The list's own page, if it has one. */
  href: string | null;
  items: ListItem[];
}

const folderName = (path: string) => path.replace(/[\\/]+$/, '').split(/[\\/]/).pop() || path;

function listName(context: ListContext, scopeName: string | null | undefined): string {
  if (context.kind === 'all') return 'All videos';
  if (context.kind === 'unassigned') return 'Not in a project';
  if (context.kind === 'folder') return context.id === 'uploads' ? 'Uploaded videos' : folderName(context.id);
  return scopeName || (context.kind === 'run' ? 'Run' : 'Project');
}

/** Every list is a scope of the Videos page (spec 029), in its order. */
async function load(context: ListContext): Promise<ListInfo> {
  const listing = await apiClient.listVideos(videosQueryOf(context));
  return {
    name: listName(context, listing.scope.name),
    href: videosHref(context),
    items: listing.videos.map((video) => ({
      name: video.name,
      path: video.path,
      run: video.run_job_id ?? undefined,
      // A missing file with results still opens (spec 028); one with neither has nothing to show.
      usable: video.available || Object.keys(video.results).length > 0,
    })),
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
