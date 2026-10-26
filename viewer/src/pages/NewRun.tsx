import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useLocation, useNavigate } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';

import { apiClient } from '@/api/client';
import { APIError } from '@/api/handleError';
import { NeedsServer } from '@/components/NeedsServer';
import { PageHelp } from '@/components/PageHelp';
import { ChooseVideos, type Chosen } from '@/components/run/ChooseVideos';
import { StartRun } from '@/components/run/StartRun';
import { defaultRunName, uploadRun } from '@/components/run/uploadRun';
import { WhatToFind } from '@/components/run/WhatToFind';
import { WorkflowPicker } from '@/components/workflows/WorkflowPicker';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { usePipelineCatalog } from '@/hooks/usePipelineCatalog';
import { useRunSettings } from '@/hooks/useRunSettings';
import { notify } from '@/lib/notify';
import { configFor, withDefaults } from '@/lib/pipelineDefaults';
import { isPipelineSelectable, partitionSelection } from '@/lib/pipelineExtras';
import { projectOf, runStartOf, targetLabel, type RunTarget } from '@/lib/runTarget';
import { videosWord } from '@/lib/runs';
import { rememberRunSetup, weightsNotesFor } from '@/lib/runSetup';
import { userMessage } from '@/lib/userMessage';
import { hrefOf } from '@/lib/videosScope';
import type { RunSettings } from '@/types/videos';

const fileName = (path: string) => path.split(/[\\/]/).pop() ?? path;
const day = (iso: string | null | undefined) =>
  iso ? new Date(iso).toLocaleDateString(undefined, { day: 'numeric', month: 'short' }) : null;

interface Settings {
  key: string;
  selected: string[];
  config: Record<string, unknown>;
  source: ReactNode;
}

/** The server's prefill, said in words (spec 029 FR-012). */
function fromServer(found: RunSettings): Settings {
  const { source } = found;
  const when = day(source.changed_at);
  const words =
    source.kind === 'workflow'
      ? `the workflow “${source.name}”`
      : source.kind === 'project'
        ? `this project's settings${when ? `, last changed ${when}` : ''}`
        : source.kind === 'last_used'
          ? `your last run${source.name ? ` (${source.name})` : ''}${when ? `, ${when}` : ''}`
          : 'the defaults (nothing ticked until you choose)';
  return {
    key: `server:${source.kind}:${source.id ?? ''}:${source.changed_at ?? ''}`,
    selected: found.selected_pipelines,
    config: found.config,
    source: words,
  };
}

/** The videos of a folder or project target, asked of the server; others are known already. */
function useTargetVideos(target: RunTarget | null) {
  const query =
    target?.kind === 'project' ? { project: target.id } : target?.kind === 'folder' ? { folder: target.path, recursive: target.recursive } : null;
  return useQuery({
    queryKey: ['videos', 'run-target', query],
    queryFn: () => apiClient.listVideos(query!),
    enabled: query !== null,
    select: (data) => data.videos.filter((v) => v.available).map((v) => v.path),
    retry: false,
  });
}

/** A run being repeated: its own settings, to start from. */
function useRerunSettings(target: RunTarget | null) {
  return useQuery({
    queryKey: ['run-settings-of', target?.kind === 'rerun' ? target.runId : null],
    enabled: target?.kind === 'rerun',
    retry: false,
    queryFn: async () => {
      const rerun = target as Extract<RunTarget, { kind: 'rerun' }>;
      const job = rerun.single ? await apiClient.getJob(rerun.runId) : (await apiClient.getJobs(1, 1, { batchId: rerun.runId })).jobs[0];
      return { selected: job?.selected_pipelines ?? [], config: (job?.config ?? {}) as Record<string, unknown> };
    },
  });
}

/**
 * New run (spec 029 R9): opened on a target, two parts, What to find and
 * Start; pressing Start is the only confirmation (FR-017). Without a target,
 * it first asks for the videos.
 */
export default function NewRun() {
  const location = useLocation();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [start] = useState(() => runStartOf(location.state));
  const [target, setTarget] = useState<RunTarget | null>(start.target);
  const [files, setFiles] = useState<File[] | null>(null);
  const [workflowId, setWorkflowId] = useState<string | null>(start.workflowId ?? null);
  const [selected, setSelected] = useState<string[]>([]);
  const [config, setConfig] = useState<Record<string, unknown>>({});
  const [applied, setApplied] = useState<string | null>(null);
  const [notices, setNotices] = useState<string[]>([]);
  const [valid, setValid] = useState(true);
  const [name, setName] = useState('');
  const [starting, setStarting] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  const catalog = usePipelineCatalog();
  const pipelines = useMemo(() => catalog.data?.catalog.pipelines ?? [], [catalog.data]);
  const chosen = target !== null || files !== null;
  const own = start.settings && !workflowId ? start.settings : null;
  const rerunSettings = useRerunSettings(own ? null : target);
  const usesServer = !own && !(target?.kind === 'rerun' && !workflowId);
  const server = useRunSettings({ project: projectOf(target), workflow: workflowId }, chosen && usesServer);
  const targetVideos = useTargetVideos(target);

  const settings: Settings | null = useMemo(() => {
    if (own) {
      return {
        key: 'start',
        selected: own.selectedPipelines ?? [],
        config: own.config ?? {},
        source: own.label ?? 'the settings you came with',
      };
    }
    if (target?.kind === 'rerun' && !workflowId) {
      return rerunSettings.data
        ? { key: `rerun:${target.runId}`, ...rerunSettings.data, source: `the run “${target.label}”` }
        : null;
    }
    return server.data ? fromServer(server.data) : null;
  }, [own, target, workflowId, rerunSettings.data, server.data]);

  useEffect(() => {
    if (!settings || !pipelines.length || applied === settings.key) return;
    // Nothing ticked that the user, a project, a workflow or a last run didn't choose (UAT #42, P6).
    const wanted = settings.selected;
    const known = wanted.filter((id) => pipelines.some((p) => p.id === id));
    const { kept, leftOut } = partitionSelection(known, pipelines);
    const unknown = wanted.filter((id) => !known.includes(id)).map((id) => `${id}: not on this server`);
    setSelected(kept);
    setNotices([...leftOut, ...unknown]);
    setConfig(withDefaults(pipelines, settings.config));
    setApplied(settings.key);
  }, [settings, pipelines, applied]);

  const onConfig = useCallback((updater: (prev: Record<string, unknown>) => Record<string, unknown>) => setConfig(updater), []);

  const paths = target?.kind === 'videos' ? target.paths : target?.kind === 'project' || target?.kind === 'folder' ? targetVideos.data ?? null : null;
  const count = files ? files.length : target?.kind === 'rerun' ? (target.single ? 1 : target.videoCount ?? null) : paths ? paths.length : null;
  const names = files ? files.map((f) => f.name) : paths ? paths.map(fileName) : [];
  const where = files
    ? 'from this computer, uploaded first'
    : target?.kind === 'folder'
      ? `in ${target.path}`
      : target?.kind === 'project'
        ? `in ${target.name}`
        : target?.kind === 'rerun'
          ? `of ${target.label}, used again`
          : 'on the server';
  const projectName = target?.kind === 'project' ? target.name : target?.kind === 'videos' ? target.projectName ?? null : null;
  const defaultName = files ? defaultRunName(files) : target ? (target.kind === 'rerun' ? `${target.label} (rerun)` : targetLabel(target)) : '';
  const blocked =
    selected.length === 0
      ? 'Choose at least one thing to find.'
      : pipelines.some((p) => selected.includes(p.id) && !isPipelineSelectable(p))
        ? "Some of the chosen pipelines can't run right now; take them out, or set them up on Pipelines."
        : !valid
          ? 'Fix the settings marked above first.'
          : null;

  const startRun = async () => {
    setStarting(true);
    setProblem(null);
    const runConfig = configFor(selected, config);
    try {
      let runId: string;
      let started: number;
      let skipped: string[] = [];
      if (files) {
        const result = await uploadRun(files, selected, runConfig, { name, projectId: projectOf(target) });
        if (!result.created.length) throw new Error(result.failed.map((f) => `${f.name}: ${f.reason}`).join('; '));
        runId = result.runId;
        started = result.created.length;
        skipped = result.failed.map((f) => `${f.name}: ${f.reason}`);
      } else if (target?.kind === 'rerun') {
        const overrides = { selected_pipelines: selected, config: runConfig ?? {} };
        if (target.single) {
          runId = (await apiClient.rerunJob(target.runId, overrides)).id;
          started = 1;
        } else {
          const result = await apiClient.rerunBatch(target.runId, overrides);
          runId = result.batch_id;
          started = result.created.length;
          skipped = result.skipped.map((s) => s.reason);
        }
      } else {
        const result = await apiClient.runVideos({
          paths: paths ?? [],
          selected_pipelines: selected,
          config: runConfig,
          name: name.trim() || undefined,
          project_id: projectOf(target),
        });
        runId = result.run_id;
        started = result.created.length;
        skipped = result.skipped.map((s) => `${fileName(s.path)}: ${s.reason}`);
      }
      rememberRunSetup(runId, weightsNotesFor(pipelines, selected));
      queryClient.invalidateQueries({ queryKey: ['videos'] });
      notify({
        title: `Run started on ${videosWord(started)}`,
        body: `You can close this tab; results will be here.${skipped.length ? ` Not run: ${skipped.join('; ')}` : ''}`,
        action: { label: 'Show run', to: hrefOf({ run: runId }) },
        tone: 'success',
      });
      navigate(hrefOf({ run: runId }));
    } catch (error) {
      setProblem(error instanceof APIError || error instanceof Error ? userMessage(error) : String(error));
    } finally {
      setStarting(false);
    }
  };

  const choose = (choice: Chosen) => {
    if ('files' in choice) setFiles(choice.files);
    else setTarget(choice.target);
  };

  if (catalog.error instanceof APIError && catalog.error.status === 0) {
    return (
      <div className="container mx-auto max-w-3xl space-y-4 px-6 py-8">
        <NeedsServer what="A run finds things in videos on a VideoAnnotator server: speech, faces, people, scenes, and answers to your questions." />
      </div>
    );
  }

  return (
    <div className="container mx-auto max-w-5xl space-y-6 px-6 py-8">
      <div className="space-y-2">
        <Button variant="ghost" size="sm" onClick={() => (window.history.length > 1 ? navigate(-1) : navigate('/videos'))}>
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back
        </Button>
        <h1 className="text-3xl font-bold">New run</h1>
        <PageHelp page="newRun">
          Choose what to find in {chosen ? names.length === 1 ? names[0] : `these ${videosWord(count ?? 0)}` : 'some videos'}, then start.
        </PageHelp>
      </div>

      {!chosen && (
        <Card>
          <CardHeader>
            <CardTitle>Which videos</CardTitle>
          </CardHeader>
          <CardContent>
            <ChooseVideos onChoose={choose} />
          </CardContent>
        </Card>
      )}

      {chosen && (
        <>
          <Card>
            <CardHeader>
              <CardTitle>1. What to find</CardTitle>
            </CardHeader>
            <CardContent>
              <WhatToFind
                pipelines={pipelines}
                isLoading={catalog.isLoading}
                error={catalog.error}
                restartRequired={catalog.data?.restartRequired ?? false}
                selected={selected}
                onSelect={setSelected}
                config={config}
                onConfig={onConfig}
                source={settings?.source ?? '…'}
                workflows={<WorkflowPicker value={workflowId} onChange={setWorkflowId} />}
                notices={notices}
                onDismissNotices={() => setNotices([])}
                onValidity={setValid}
              />
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>2. Start</CardTitle>
            </CardHeader>
            <CardContent>
              <StartRun
                count={count}
                names={names}
                where={where}
                pipelineTitles={selected.map((id) => {
                  const p = pipelines.find((x) => x.id === id);
                  return p?.title || p?.name || id;
                })}
                setupNotes={weightsNotesFor(pipelines, selected)}
                name={name}
                onName={setName}
                defaultName={defaultName}
                project={projectName}
                starting={starting}
                blocked={blocked}
                onStart={startRun}
              />
              {problem && (
                <p className="mt-3 text-sm text-destructive" role="alert">
                  {problem}
                </p>
              )}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
