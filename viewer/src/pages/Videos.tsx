import { useEffect, useMemo, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { Loader2 } from 'lucide-react';

import type { ExportScope } from '@/api/client';
import { APIError } from '@/api/handleError';
import { NeedsServer } from '@/components/NeedsServer';
import { PageHelp } from '@/components/PageHelp';
import { Button } from '@/components/ui/button';
import { Term } from '@/components/Term';
import { AddToProject } from '@/components/videos/AddToProject';
import { NewProjectDialog } from '@/components/videos/NewProjectDialog';
import { ProjectHeader } from '@/components/videos/ProjectHeader';
import { RunHeader } from '@/components/videos/RunHeader';
import { RunWhatChanged } from '@/components/videos/RunWhatChanged';
import { ScopeHeader } from '@/components/videos/ScopeHeader';
import { SelectionBar } from '@/components/videos/SelectionBar';
import { Sidebar } from '@/components/videos/Sidebar';
import { VideosTable } from '@/components/videos/VideosTable';
import { usePipelineTitles } from '@/hooks/usePipelineTitles';
import { useProject, useProjects } from '@/hooks/useProjects';
import { useWorkflows } from '@/hooks/useWorkflows';
import { ApplyWorkflow } from '@/components/workflows/ApplyWorkflow';
import { SaveAsWorkflow } from '@/components/workflows/SaveAsWorkflow';
import { projectSettingsOf, runSettingsOf } from '@/lib/workflows';
import { useVideos } from '@/hooks/useVideos';
import type { RunTarget } from '@/lib/runTarget';
import { videosWord } from '@/lib/runs';
import { userMessage } from '@/lib/userMessage';
import { hrefOf, listContextOf, paramsOf, pipelinesOf, queryOf, scopeTitle } from '@/lib/videosScope';
import type { VideoItem, VideosQuery } from '@/types/videos';

function runTargetOf(query: VideosQuery, title: string, videos: VideoItem[]): RunTarget | null {
  if (query.project) return { kind: 'project', id: query.project, name: title };
  if (query.folder && query.folder !== 'uploads') return { kind: 'folder', path: query.folder, recursive: !!query.recursive };
  if (query.run) return null;
  const paths = videos.filter((v) => v.available).map((v) => v.path);
  return paths.length ? { kind: 'videos', paths, label: `${videosWord(paths.length)} in ${title}` } : null;
}

function exportScopeOf(query: VideosQuery, videos: VideoItem[]): ExportScope | null {
  if (query.project) return { kind: 'project', id: query.project };
  if (query.run) return { kind: 'run', id: query.run };
  return videos.length ? { kind: 'videos', paths: videos.map((v) => v.path) } : null;
}

const isNoServer = (error: unknown) => error instanceof APIError && error.status === 0;

/**
 * Every video the server knows, by place: all, not in a project, a project,
 * a folder, or a run (spec 029 US1, US2, US5).
 */
export default function Videos() {
  const [params, setParams] = useSearchParams();
  const query = useMemo(() => queryOf(params), [params]);
  const scopeKey = paramsOf(query).toString();
  const filter = params.get('q') ?? '';
  const { data, error, isLoading } = useVideos(query);
  const { titleOf, order } = usePipelineTitles(!error);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [naming, setNaming] = useState(false);
  const projects = useProjects();

  useEffect(() => setSelected(new Set()), [scopeKey]);

  const videos = useMemo(() => {
    const all = data?.videos ?? [];
    const needle = filter.trim().toLowerCase();
    return needle ? all.filter((v) => v.name.toLowerCase().includes(needle)) : all;
  }, [data, filter]);
  const pipelines = useMemo(() => pipelinesOf(data?.videos ?? [], order), [data, order]);
  const title = scopeTitle(data?.scope, query);
  const context = listContextOf(query);
  const selectedPaths = (data?.videos ?? []).map((v) => v.path).filter((p) => selected.has(p));
  const project = query.project ? { id: query.project, name: title } : null;
  const run = query.run ? data?.runs.find((r) => r.id === query.run) : undefined;
  // A run of a project whose settings came from a workflow offers its results as the workflow says (T050).
  const runProject = useProject(run?.project?.id);
  const workflows = useWorkflows(!!runProject.data?.current_settings?.source);
  const fromWorkflow = runProject.data?.current_settings?.source;
  const runExportChoices =
    fromWorkflow?.kind === 'workflow' ? workflows.data?.find((w) => w.id === fromWorkflow.id)?.export_choices ?? null : null;

  const setFilter = (text: string) => {
    const next = new URLSearchParams(params);
    if (text) next.set('q', text);
    else next.delete('q');
    setParams(next, { replace: true });
  };
  const select = (paths: string[], on: boolean) =>
    setSelected((current) => {
      const next = new Set(current);
      for (const path of paths) {
        if (on) next.add(path);
        else next.delete(path);
      }
      return next;
    });

  if (isNoServer(error)) {
    return (
      <div className="container mx-auto space-y-4 px-6 py-8">
        <PageHelp page="noServer">Videos lists the videos on a VideoAnnotator server and the results each has.</PageHelp>
        <NeedsServer what="Videos lists the videos a VideoAnnotator server has, with the results each one has, and runs pipelines on them." />
      </div>
    );
  }

  return (
    <div className="container mx-auto grid gap-6 px-6 py-6 md:grid-cols-[15rem_1fr]">
      <aside className="md:sticky md:top-20 md:self-start">
        <Sidebar counts={data?.counts} current={query} onNewProject={() => setNaming(true)} />
        <NewProjectDialog
          open={naming}
          onOpenChange={setNaming}
          count={0}
          busy={projects.create.isPending}
          onCreate={(name) => projects.create.mutate({ name }, { onSuccess: () => setNaming(false) })}
        />
      </aside>
      <main className="min-w-0 space-y-4">
        <PageHelp page="videos">
          Every <Term word="video" /> the server knows, by <Term word="folder" /> and <Term word="project" />, with
          the results each has. Pick some, then <strong>Run…</strong> or <strong>Export</strong>.
        </PageHelp>

        {error && !data && (
          <p className="text-sm text-destructive" role="alert">
            {userMessage(error)}
          </p>
        )}

        {run ? (
          <RunHeader run={run} firstVideoPath={data?.videos[0]?.path ?? null} exportChoices={runExportChoices}>
            <RunWhatChanged run={run} />
            <SaveAsWorkflow source={{ kind: 'run', id: run.id, name: run.name }} settings={() => runSettingsOf(run.job_ids[0])} />
          </RunHeader>
        ) : query.project && data ? (
          <ProjectHeader
            id={query.project}
            data={data}
            workflows={
              <>
                <SaveAsWorkflow source={{ kind: 'project', id: query.project, name: title }} settings={() => projectSettingsOf(query.project!)} />
                <ApplyWorkflow project={{ id: query.project, name: title }} />
              </>
            }
          />
        ) : (
          data && (
            <ScopeHeader
              title={title}
              count={data.total}
              runTarget={runTargetOf(query, title, data.videos)}
              exportScope={exportScopeOf(query, data.videos)}
              runs={data.runs}
              videos={data.videos}
              below={
                data.subfolders.length > 0 && (
                  <p className="flex flex-wrap items-center gap-2 text-sm" data-testid="subfolders">
                    <span className="text-muted-foreground">Folders inside:</span>
                    {data.subfolders.map((f) => (
                      <Link key={f.path} to={hrefOf({ folder: f.path })} className="rounded-md border px-2 py-0.5 hover:bg-muted">
                        {f.name}
                      </Link>
                    ))}
                  </p>
                )
              }
            />
          )
        )}

        {selectedPaths.length > 0 && (
          <SelectionBar paths={selectedPaths} project={project} onClear={() => setSelected(new Set())}>
            <AddToProject paths={selectedPaths} projects={data?.counts.projects ?? []} onDone={() => setSelected(new Set())} />
            {project && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => projects.remove.mutate({ project, paths: selectedPaths }, { onSuccess: () => setSelected(new Set()) })}
              >
                Remove from project
              </Button>
            )}
          </SelectionBar>
        )}

        {isLoading && !data ? (
          <p className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" /> Loading videos…
          </p>
        ) : data && data.total === 0 ? (
          <EmptyScope query={query} />
        ) : (
          data && (
            <VideosTable
              videos={videos}
              pipelines={pipelines}
              titleOf={titleOf}
              selected={selected}
              onSelect={select}
              context={context}
              filter={filter}
              onFilter={setFilter}
            />
          )
        )}
      </main>
    </div>
  );
}

function EmptyScope({ query }: { query: VideosQuery }) {
  const text = query.project
    ? 'This project has no videos yet. Select videos anywhere and choose Add to project.'
    : query.folder
      ? 'No videos in this folder.'
      : query.notInProject
        ? 'Every video is in a project.'
        : 'No videos yet. Pick a folder on the left, or start a run from Home to upload videos from this computer.';
  return (
    <p className="rounded-md border border-dashed p-8 text-center text-sm text-muted-foreground" data-testid="videos-empty">
      {text}
    </p>
  );
}
