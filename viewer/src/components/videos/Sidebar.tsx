import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { Folder, FolderInput, FolderKanban, Layers, Plus, Upload, VideoOff } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { hrefOf, sameScope, shortFolder } from '@/lib/videosScope';
import type { VideoCounts, VideosQuery } from '@/types/videos';

function Entry({
  to,
  current,
  icon,
  label,
  count,
  title,
}: {
  to: VideosQuery;
  current: VideosQuery;
  icon: ReactNode;
  label: string;
  /** Unknown until the list has loaded: then nothing, never a 0 that isn't true. */
  count?: number;
  title?: string;
}) {
  const active = sameScope(to, current);
  return (
    <Link
      to={hrefOf(to)}
      aria-current={active ? 'page' : undefined}
      title={title ?? label}
      className={cn(
        'flex items-center gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-muted',
        active && 'bg-muted font-medium',
      )}
    >
      <span className="shrink-0 text-muted-foreground">{icon}</span>
      <span className="min-w-0 flex-1 truncate">{label}</span>
      <span className="shrink-0 text-xs tabular-nums text-muted-foreground">{count}</span>
    </Link>
  );
}

function Heading({ children, action }: { children: ReactNode; action?: ReactNode }) {
  return (
    <div className="mt-4 flex items-center justify-between px-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
      <span>{children}</span>
      {action}
    </div>
  );
}

/**
 * All videos, Not in a project, Projects and Folders, each with its count
 * (spec 029 FR-001). Runs are not here: a run is a filter, reached from its
 * project, the activity indicator or a message.
 */
export function Sidebar({
  counts,
  current,
  onNewProject,
}: {
  counts: VideoCounts | undefined;
  current: VideosQuery;
  onNewProject?: () => void;
}) {
  const projects = counts?.projects ?? [];
  const clashing = new Set(
    projects.map((p) => p.name).filter((name, i, names) => names.indexOf(name) !== i),
  );
  return (
    <nav aria-label="Videos by place" className="space-y-0.5">
      <Entry to={{}} current={current} icon={<Layers className="h-4 w-4" />} label="All videos" count={counts?.all} />
      <Entry
        to={{ notInProject: true }}
        current={current}
        icon={<VideoOff className="h-4 w-4" />}
        label="Not in a project"
        count={counts?.not_in_project}
      />

      <Heading
        action={
          onNewProject && (
            <Button variant="ghost" size="sm" className="h-6 px-1.5 text-xs normal-case" onClick={onNewProject}>
              <Plus className="mr-1 h-3 w-3" />
              New project
            </Button>
          )
        }
      >
        Projects
      </Heading>
      {counts && projects.length === 0 && <p className="px-2 py-1 text-xs text-muted-foreground">None yet.</p>}
      {projects.map((project) => (
        <Entry
          key={project.id}
          to={{ project: project.id }}
          current={current}
          icon={<FolderKanban className="h-4 w-4" />}
          label={clashing.has(project.name) && project.owner_name ? `${project.name} (${project.owner_name})` : project.name}
          count={project.count}
        />
      ))}

      <Heading>Folders</Heading>
      {counts && counts.folders.length === 0 && (
        <p className="px-2 py-1 text-xs text-muted-foreground">No shared folders with videos yet.</p>
      )}
      {(counts?.folders ?? []).map((folder) => (
        <Entry
          key={folder.path}
          to={{ folder: folder.path, recursive: folder.recursive }}
          current={current}
          icon={folder.uploads ? <Upload className="h-4 w-4" /> : folder.recursive ? <FolderInput className="h-4 w-4" /> : <Folder className="h-4 w-4" />}
          label={folder.uploads ? 'Uploaded videos' : shortFolder(folder.label)}
          title={folder.uploads ? 'Videos uploaded from a browser' : folder.label}
          count={folder.count}
        />
      ))}
    </nav>
  );
}
