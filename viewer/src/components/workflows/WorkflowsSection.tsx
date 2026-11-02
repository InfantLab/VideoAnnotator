import { useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQueries, useQuery } from '@tanstack/react-query';
import { ChevronDown, Download, Pencil, Play, Plus, Trash2, Upload } from 'lucide-react';

import { apiClient } from '@/api/client';
import { WorkflowEditor } from '@/components/workflows/WorkflowEditor';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu';
import { pipelineSchemaQueryOptions } from '@/hooks/usePipelineCatalog';
import { usePipelineNames } from '@/hooks/usePipelineNames';
import { useProjects } from '@/hooks/useProjects';
import { useWorkflowActions, useWorkflows } from '@/hooks/useWorkflows';
import { saveBlob } from '@/lib/exports';
import { notify } from '@/lib/notify';
import { settingsSummary } from '@/lib/pipelineSettings';
import { runState } from '@/lib/runTarget';
import { userMessage } from '@/lib/userMessage';
import { applyBody, workflowFile } from '@/lib/workflows';
import type { PipelineDescriptor } from '@/types/pipelines';
import type { Preset, PresetCreateRequest } from '@/types/presets';
import { formatDay } from '@/lib/when';

function origin(workflow: Preset): string {
  const source = workflow.source;
  if (source?.kind === 'project') return `from the project ${source.name ?? ''}`;
  if (source?.kind === 'run') return `from the run ${source.name ?? ''}`;
  return `saved ${formatDay(workflow.created_at)}`;
}

const fileName = (name: string) => `${name.replace(/[^\w\- ]+/g, '').trim() || 'workflow'}.workflow.json`;

function WorkflowCard({ workflow, projects }: { workflow: Preset; projects: { id: string; name: string }[] }) {
  const navigate = useNavigate();
  const { nameOf } = usePipelineNames();
  const { remove } = useWorkflowActions();
  const { update } = useProjects();
  const [editing, setEditing] = useState(false);
  const [removing, setRemoving] = useState(false);
  // Each pipeline with what differs from its defaults: "DeepFace (Age on)" (UAT #71, P6).
  const schemas = useQueries({ queries: workflow.selected_pipelines.map((id) => pipelineSchemaQueryOptions(id)) });
  const pipelines = workflow.selected_pipelines.map((id, i) => {
    const parameters = schemas[i]?.data?.parameters;
    const cue = parameters ? settingsSummary({ id, name: id, parameters } as PipelineDescriptor, workflow.config) : '';
    return cue && cue !== 'defaults' ? `${nameOf(id)} (${cue})` : nameOf(id);
  });

  const exportFile = () => {
    saveBlob(new Blob([JSON.stringify(workflowFile(workflow), null, 2)], { type: 'application/json' }), fileName(workflow.name));
  };
  const apply = (project: { id: string; name: string }) =>
    update.mutate(
      { id: project.id, body: applyBody(workflow) },
      { onSuccess: () => notify({ title: `${project.name} now uses ${workflow.name}'s settings`, tone: 'success' }) },
    );

  return (
    <Card data-testid="workflow">
      <CardContent className="space-y-3 pt-4">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h3 className="font-semibold">{workflow.name}</h3>
          <span className="text-xs text-muted-foreground">{origin(workflow)}</span>
        </div>
        <p className="text-sm">{pipelines.join(', ') || 'No pipelines'}</p>
        {workflow.unavailable_pipelines?.length ? (
          <p className="text-xs text-amber-700">Not on this server now: {workflow.unavailable_pipelines.map(nameOf).join(', ')}</p>
        ) : null}
        <div className="flex flex-wrap gap-2">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button size="sm" variant="outline" disabled={!projects.length}>
                Apply to project
                <ChevronDown className="ml-1 h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent>
              {projects.map((p) => (
                <DropdownMenuItem key={p.id} onSelect={() => apply(p)}>
                  {p.name}
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
          <Button size="sm" variant="outline" onClick={() => navigate('/run', { state: runState({ target: null, workflowId: workflow.id }) })}>
            <Play className="mr-1 h-4 w-4" />
            Use in a new run
          </Button>
          <Button size="sm" variant="outline" onClick={exportFile}>
            <Download className="mr-1 h-4 w-4" />
            Export
          </Button>
          <Button size="sm" variant="ghost" onClick={() => setEditing(true)}>
            <Pencil className="mr-1 h-4 w-4" />
            Edit
          </Button>
          <Button size="sm" variant="ghost" onClick={() => setRemoving(true)}>
            <Trash2 className="mr-1 h-4 w-4" />
            Delete
          </Button>
        </div>
      </CardContent>
      {editing && <WorkflowEditor workflow={workflow} open onOpenChange={setEditing} />}
      <AlertDialog open={removing} onOpenChange={setRemoving}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete the workflow {workflow.name}?</AlertDialogTitle>
            <AlertDialogDescription>Projects that took its settings keep them; runs aren&apos;t touched.</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Keep it</AlertDialogCancel>
            <AlertDialogAction
              onClick={() =>
                remove.mutate(workflow, {
                  onSuccess: () => notify({ title: `Workflow ${workflow.name} deleted` }),
                  onError: (e) => notify({ title: "Couldn't delete it", body: userMessage(e), tone: 'error' }),
                })
              }
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </Card>
  );
}

/** Pipelines → Workflows (spec 029 FR-016, UAT #71): saved settings, made, applied, used, shared. */
export function WorkflowsSection() {
  const { data: workflows = [], isLoading } = useWorkflows();
  const { create } = useWorkflowActions();
  const projects = useQuery({ queryKey: ['videos', 'projects-list'], queryFn: () => apiClient.listDatasets(), retry: false });
  const input = useRef<HTMLInputElement>(null);
  const [making, setMaking] = useState(false);

  const importFile = async (file: File) => {
    try {
      const body = JSON.parse(await file.text()) as PresetCreateRequest;
      const workflow = await create.mutateAsync(body);
      notify({ title: `Workflow ${workflow.name} imported`, tone: 'success' });
    } catch (error) {
      notify({
        title: "Couldn't import the workflow",
        body: error instanceof SyntaxError ? "That file isn't a workflow file." : userMessage(error),
        tone: 'error',
      });
    }
  };

  return (
    <section id="workflows" className="space-y-4" data-testid="workflows">
      <p className="text-sm">
        A workflow is a set of pipelines and their settings saved under a name: to run the same way again, to give a
        project the same settings, or to send to a colleague as a file. Nothing uses one until you choose it, in New run
        or on a project.
      </p>
      <div className="flex flex-wrap gap-2">
        <Button size="sm" onClick={() => setMaking(true)}>
          <Plus className="mr-1 h-4 w-4" />
          New workflow
        </Button>
        <input
          ref={input}
          type="file"
          accept="application/json,.json"
          className="hidden"
          data-testid="workflow-import"
          onChange={(e) => {
            const file = e.target.files?.[0];
            e.target.value = '';
            if (file) void importFile(file);
          }}
        />
        <Button size="sm" variant="outline" onClick={() => input.current?.click()}>
          <Upload className="mr-1 h-4 w-4" />
          Import a file
        </Button>
      </div>
      {making && <WorkflowEditor open onOpenChange={setMaking} />}
      {!isLoading && workflows.length === 0 && (
        <p className="text-sm text-muted-foreground">None yet. Make one here, or save one from a project or a run with Save as workflow.</p>
      )}
      <div className="grid gap-3 md:grid-cols-2">
        {workflows.map((w) => (
          <WorkflowCard key={w.id} workflow={w} projects={(projects.data?.datasets ?? []).map((d) => ({ id: d.id, name: d.name }))} />
        ))}
      </div>
    </section>
  );
}
