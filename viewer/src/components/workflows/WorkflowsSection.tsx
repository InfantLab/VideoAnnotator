import { useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { ChevronDown, Download, Pencil, Play, Trash2, Upload } from 'lucide-react';

import { apiClient } from '@/api/client';
import { PageHelp } from '@/components/PageHelp';
import { Term } from '@/components/Term';
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
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu';
import { Input } from '@/components/ui/input';
import { usePipelineTitles } from '@/hooks/usePipelineTitles';
import { useProjects } from '@/hooks/useProjects';
import { useWorkflowActions, useWorkflows } from '@/hooks/useWorkflows';
import { saveBlob } from '@/lib/exports';
import { notify } from '@/lib/notify';
import { runState } from '@/lib/runTarget';
import { userMessage } from '@/lib/userMessage';
import { applyBody, workflowFile } from '@/lib/workflows';
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
  const { titleOf } = usePipelineTitles();
  const { update: rename, remove } = useWorkflowActions();
  const { update } = useProjects();
  const [naming, setNaming] = useState(false);
  const [name, setName] = useState(workflow.name);
  const [removing, setRemoving] = useState(false);

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
        <p className="text-sm">{workflow.selected_pipelines.map(titleOf).join(', ') || 'No pipelines'}</p>
        {workflow.unavailable_pipelines?.length ? (
          <p className="text-xs text-amber-700">Not on this server now: {workflow.unavailable_pipelines.map(titleOf).join(', ')}</p>
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
          <Button size="sm" variant="ghost" onClick={() => setNaming(true)}>
            <Pencil className="mr-1 h-4 w-4" />
            Rename
          </Button>
          <Button size="sm" variant="ghost" onClick={() => setRemoving(true)}>
            <Trash2 className="mr-1 h-4 w-4" />
            Delete
          </Button>
        </div>
      </CardContent>
      <Dialog open={naming} onOpenChange={setNaming}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Rename {workflow.name}</DialogTitle>
          </DialogHeader>
          <Input value={name} onChange={(e) => setName(e.target.value)} aria-label="Workflow name" autoFocus />
          <DialogFooter>
            <Button
              disabled={!name.trim()}
              onClick={() =>
                rename.mutate(
                  { id: workflow.id, body: { name: name.trim() } },
                  {
                    onSuccess: () => setNaming(false),
                    onError: (e) => notify({ title: "Couldn't rename it", body: userMessage(e), tone: 'error' }),
                  },
                )
              }
            >
              Rename
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
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

/** Pipelines → Workflows (spec 029 FR-016): saved settings, applied, used, shared. */
export function WorkflowsSection() {
  const { data: workflows = [], isLoading } = useWorkflows();
  const { create } = useWorkflowActions();
  const projects = useQuery({ queryKey: ['videos', 'projects-list'], queryFn: () => apiClient.listDatasets(), retry: false });
  const input = useRef<HTMLInputElement>(null);

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
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-2xl font-bold">
          <Term word="workflow">Workflows</Term>
        </h2>
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
          Import workflow
        </Button>
      </div>
      <PageHelp page="workflows">Pipelines and settings saved under a name, used only when you choose them.</PageHelp>
      {!isLoading && workflows.length === 0 && (
        <p className="text-sm text-muted-foreground">None yet. Save one from a project or a run: Save as workflow.</p>
      )}
      <div className="grid gap-3 md:grid-cols-2">
        {workflows.map((w) => (
          <WorkflowCard key={w.id} workflow={w} projects={(projects.data?.datasets ?? []).map((d) => ({ id: d.id, name: d.name }))} />
        ))}
      </div>
    </section>
  );
}
