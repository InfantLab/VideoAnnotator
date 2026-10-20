import { useCallback, useState, type ReactNode } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Pencil, Trash2 } from 'lucide-react';

import { WhatToFind } from '@/components/run/WhatToFind';
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
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from '@/components/ui/sheet';
import { ScopeHeader } from '@/components/videos/ScopeHeader';
import { NewInFolder } from '@/components/videos/NewInFolder';
import { WhatChanged } from '@/components/videos/WhatChanged';
import { usePipelineCatalog } from '@/hooks/usePipelineCatalog';
import { usePipelineTitles } from '@/hooks/usePipelineTitles';
import { useProject, useProjects } from '@/hooks/useProjects';
import { useRunSettings } from '@/hooks/useRunSettings';
import { notify } from '@/lib/notify';
import { withDefaults } from '@/lib/pipelineDefaults';
import { runStatus } from '@/lib/runs';
import { hrefOf } from '@/lib/videosScope';
import type { RunSettings, RunSummary, VideoItem, VideosResponse } from '@/types/videos';

const day = (iso: string | null | undefined) =>
  iso ? new Date(iso).toLocaleDateString(undefined, { day: 'numeric', month: 'short' }) : null;

/** "from the run First faces, 12 Oct" / "edited 14 Oct" / "from the workflow Faces 40". */
function settingsOrigin(settings: RunSettings): string {
  const from = settings.source.from as { kind?: string; name?: string | null } | null | undefined;
  const when = day(settings.source.changed_at);
  const tail = when ? `, ${when}` : '';
  if (from?.kind === 'workflow') return `from the workflow ${from.name ?? ''}${tail}`;
  if (from?.kind === 'edit') return `edited${tail}`;
  if (from?.kind === 'run') return `from the run ${from.name ?? ''}${tail}`;
  return 'the defaults until it runs';
}

function question(config: Record<string, unknown>): string | null {
  const vlm = config.vlm_annotation as { prompt?: unknown } | undefined;
  return typeof vlm?.prompt === 'string' && vlm.prompt.trim() ? vlm.prompt.trim() : null;
}

/** The project's settings in the Edit sheet: What to find, saved without running (spec 029 FR-010). */
function EditSettings({ project, settings, open, onOpenChange }: { project: { id: string; name: string }; settings: RunSettings; open: boolean; onOpenChange: (open: boolean) => void }) {
  const catalog = usePipelineCatalog({ enabled: open });
  const pipelines = catalog.data?.catalog.pipelines ?? [];
  const [selected, setSelected] = useState<string[]>(settings.selected_pipelines);
  const [config, setConfig] = useState<Record<string, unknown>>(settings.config);
  const [valid, setValid] = useState(true);
  const { update } = useProjects();
  const onConfig = useCallback((updater: (prev: Record<string, unknown>) => Record<string, unknown>) => setConfig(updater), []);

  const save = () =>
    update.mutate(
      {
        id: project.id,
        body: {
          current_settings: {
            selected_pipelines: selected,
            config: Object.fromEntries(Object.entries(withDefaults(pipelines, config)).filter(([id]) => selected.includes(id))),
            source: { kind: 'edit' },
          },
        },
      },
      {
        onSuccess: () => {
          notify({ title: `Settings saved for ${project.name}`, body: 'The next run starts from them.', tone: 'success' });
          onOpenChange(false);
        },
      },
    );

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="w-full overflow-y-auto sm:max-w-3xl">
        <SheetHeader>
          <SheetTitle>Settings for {project.name}</SheetTitle>
          <SheetDescription>What the next run on this project finds, and how. Saving doesn&apos;t start a run.</SheetDescription>
        </SheetHeader>
        <div className="mt-4 space-y-4">
          <WhatToFind
            pipelines={pipelines}
            isLoading={catalog.isLoading}
            error={catalog.error}
            restartRequired={catalog.data?.restartRequired ?? false}
            selected={selected}
            onSelect={setSelected}
            config={config}
            onConfig={onConfig}
            source={settingsOrigin(settings)}
            onValidity={setValid}
          />
          <div className="flex gap-2">
            <Button onClick={save} disabled={update.isPending || !valid || !selected.length}>
              Save settings
            </Button>
            <Button variant="ghost" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
          </div>
        </div>
      </SheetContent>
    </Sheet>
  );
}

function RunsLine({ runs }: { runs: RunSummary[] }) {
  if (!runs.length) return null;
  return (
    <p className="flex flex-wrap gap-x-3 text-sm" data-testid="project-runs">
      <span className="text-muted-foreground">
        <Term word="run">Runs</Term>:
      </span>
      {runs.slice(0, 5).map((run) => (
        <Link key={run.id} to={hrefOf({ run: run.id })} className="hover:underline" title={run.name}>
          {run.active ? <span className="text-blue-600">●</span> : '✓'} {runStatus(run)}
        </Link>
      ))}
      {runs.length > 5 && <span className="text-muted-foreground">and {runs.length - 5} more</span>}
    </p>
  );
}

interface ProjectHeaderProps {
  id: string;
  data: VideosResponse;
  /** Save as workflow, Apply workflow (spec 029 US4). */
  workflows?: ReactNode;
}

/** A project's header (spec 029 US2, US3): its settings, runs, and what can be done to it. */
export function ProjectHeader({ id, data, workflows }: ProjectHeaderProps) {
  const navigate = useNavigate();
  const project = useProject(id);
  const settings = useRunSettings({ project: id });
  const { titleOf } = usePipelineTitles();
  const { update, destroy } = useProjects();
  const [editing, setEditing] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [removing, setRemoving] = useState(false);
  const [name, setName] = useState('');
  const dataset = project.data;
  const title = dataset?.name ?? data.scope.name ?? 'Project';
  const ref = { id, name: title };
  const follows = !!dataset?.server_folder && !dataset.server_selection;
  const videos: VideoItem[] = data.videos;

  const rename = () =>
    update.mutate(
      { id, body: { name: name.trim() } },
      { onSuccess: () => { setRenaming(false); notify({ title: `Renamed to ${name.trim()}` }); } },
    );

  return (
    <>
      <ScopeHeader
        title={
          <span className="flex items-center gap-2">
            {title}
            <Button variant="ghost" size="icon" className="h-7 w-7" aria-label="Rename" onClick={() => { setName(title); setRenaming(true); }}>
              <Pencil className="h-4 w-4" />
            </Button>
          </span>
        }
        count={data.total}
        subtitle={
          <>
            {dataset?.owner_name && <>made by {dataset.owner_name}</>}
            {follows && <> · All videos in {dataset!.server_folder}, including new ones</>}
          </>
        }
        runTarget={{ kind: 'project', id, name: title }}
        exportScope={{ kind: 'project', id }}
        runs={data.runs}
        videos={videos}
        below={
          <>
            {settings.data && (
              <p className="flex flex-wrap items-center gap-x-2 text-sm" data-testid="project-settings">
                <span className="text-muted-foreground">Settings:</span>{' '}
                <span>
                  {settings.data.selected_pipelines.length
                    ? settings.data.selected_pipelines.map(titleOf).join(', ')
                    : 'none yet'}
                </span>{' '}
                {question(settings.data.config) && <span className="italic">“{question(settings.data.config)}”</span>}
                <span className="text-xs text-muted-foreground">({settingsOrigin(settings.data)})</span>
                <Button variant="link" size="sm" className="h-auto p-0" onClick={() => setEditing(true)}>
                  Edit
                </Button>
              </p>
            )}
            <RunsLine runs={data.runs} />
            {dataset?.server_folder && <NewInFolder project={ref} folder={dataset.server_folder} videos={data.new_in_folder} />}
          </>
        }
      >
        <WhatChanged runs={data.runs} />
        {workflows}
        <Button variant="outline" size="sm" onClick={() => setRemoving(true)}>
          <Trash2 className="mr-1 h-4 w-4" />
          Remove project
        </Button>
      </ScopeHeader>

      {settings.data && editing && <EditSettings project={ref} settings={settings.data} open={editing} onOpenChange={setEditing} />}

      <Dialog open={renaming} onOpenChange={setRenaming}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Rename {title}</DialogTitle>
          </DialogHeader>
          <Input autoFocus value={name} onChange={(e) => setName(e.target.value)} aria-label="Project name" />
          <DialogFooter>
            <Button onClick={rename} disabled={!name.trim() || update.isPending}>
              Rename
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <AlertDialog open={removing} onOpenChange={setRemoving}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Remove the project {title}?</AlertDialogTitle>
            <AlertDialogDescription>
              Only the project goes. Its videos and all their results stay, in their folders and in other projects.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Keep it</AlertDialogCancel>
            <AlertDialogAction onClick={() => destroy.mutate(ref, { onSuccess: () => navigate('/videos') })}>
              Remove project
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
