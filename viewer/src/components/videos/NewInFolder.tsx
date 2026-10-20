import { Button } from '@/components/ui/button';
import { useProjects } from '@/hooks/useProjects';
import { videosWord } from '@/lib/runs';

/** "3 new videos in /data/pk · Add them": a selection's folder has grown (spec 029 FR-007). */
export function NewInFolder({ project, folder, videos }: { project: { id: string; name: string }; folder: string; videos: { path: string; name: string }[] }) {
  const { add } = useProjects();
  if (!videos.length) return null;
  return (
    <p className="flex flex-wrap items-center gap-2 rounded-md border border-dashed px-3 py-2 text-sm" data-testid="new-in-folder">
      <span>
        <strong>{videos.length === 1 ? '1 new video' : `${videos.length} new videos`}</strong> in {folder}
      </span>
      <span className="text-muted-foreground" title={videos.map((v) => v.name).join(', ')}>
        ({videos.slice(0, 3).map((v) => v.name).join(', ')}
        {videos.length > 3 ? ', …' : ''})
      </span>
      <Button size="sm" variant="secondary" onClick={() => add.mutate({ project, paths: videos.map((v) => v.path) })} disabled={add.isPending}>
        Add {videos.length === 1 ? 'it' : 'them'}
      </Button>
      <span className="sr-only">{videosWord(videos.length)}</span>
    </p>
  );
}
