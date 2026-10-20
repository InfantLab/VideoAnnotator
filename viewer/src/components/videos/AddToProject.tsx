import { useState } from 'react';
import { ChevronDown, FolderPlus, Plus } from 'lucide-react';

import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { NewProjectDialog } from '@/components/videos/NewProjectDialog';
import { useProjects } from '@/hooks/useProjects';
import type { ProjectCount } from '@/types/videos';

/** Add to project ▾: each project, or New project… (spec 029 FR-005). */
export function AddToProject({ paths, projects, onDone }: { paths: string[]; projects: ProjectCount[]; onDone?: () => void }) {
  const { add, create } = useProjects();
  const [naming, setNaming] = useState(false);
  const clashing = new Set(projects.map((p) => p.name).filter((n, i, all) => all.indexOf(n) !== i));
  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="outline" size="sm">
            <FolderPlus className="mr-1 h-4 w-4" />
            Add to project
            <ChevronDown className="ml-1 h-4 w-4" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start">
          {projects.map((project) => (
            <DropdownMenuItem
              key={project.id}
              onSelect={() => add.mutate({ project, paths }, { onSuccess: onDone })}
            >
              {project.name}
              {clashing.has(project.name) && project.owner_name ? ` (${project.owner_name})` : ''}
            </DropdownMenuItem>
          ))}
          {projects.length > 0 && <DropdownMenuSeparator />}
          <DropdownMenuItem onSelect={() => setNaming(true)}>
            <Plus className="mr-2 h-4 w-4" />
            New project…
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <NewProjectDialog
        open={naming}
        onOpenChange={setNaming}
        count={paths.length}
        busy={create.isPending}
        onCreate={(name) =>
          create.mutate(
            { name, paths },
            {
              onSuccess: () => {
                setNaming(false);
                onDone?.();
              },
            },
          )
        }
      />
    </>
  );
}
