import { ChevronDown, Workflow as WorkflowIcon } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuTrigger } from '@/components/ui/dropdown-menu';
import { useProjects } from '@/hooks/useProjects';
import { useWorkflows } from '@/hooks/useWorkflows';
import { notify } from '@/lib/notify';
import { applyBody } from '@/lib/workflows';
import type { Preset } from '@/types/presets';

/** Apply workflow ▾ on a project (spec 029 FR-014). */
export function ApplyWorkflow({ project }: { project: { id: string; name: string } }) {
  const { data: workflows = [] } = useWorkflows();
  const { update } = useProjects();
  if (!workflows.length) return null;
  const apply = (workflow: Preset) =>
    update.mutate(
      { id: project.id, body: applyBody(workflow) },
      {
        onSuccess: () =>
          notify({
            title: `${project.name} now uses ${workflow.name}'s settings`,
            body: 'A copy: changing the workflow later leaves the project as it is.',
            tone: 'success',
          }),
      },
    );
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="sm">
          <WorkflowIcon className="mr-1 h-4 w-4" />
          Apply workflow
          <ChevronDown className="ml-1 h-4 w-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuLabel>Copy its settings into {project.name}</DropdownMenuLabel>
        {workflows.map((w) => (
          <DropdownMenuItem key={w.id} onSelect={() => apply(w)}>
            {w.name}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
