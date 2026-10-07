import { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import {
  BookOpen,
  Check,
  Circle,
  Database,
  ExternalLink,
  Eye,
  Loader2,
  MessageSquareText,
  MonitorPlay,
  Pencil,
  Play,
  RotateCcw,
  Server,
  Star,
} from 'lucide-react';

import { apiClient } from '@/api/client';
import type { JobResponse } from '@/api/client';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { useServerCapabilitiesContext } from '@/contexts/ServerCapabilitiesContext';
import { pipelineCatalogQueryOptions } from '@/hooks/usePipelineCatalog';
import { parseApiError } from '@/lib/errorHandling';
import { getJobDatasetIndex, getRootDirHandle } from '@/lib/localLibrary/libraryStore';
import { settingsOf, wizardState } from '@/lib/wizardStart';
import type { PipelineDescriptor } from '@/types/pipelines';
import { APP_NAME, GITHUB_URL, VERSION } from '@/utils/version';

const SETUP_HIDDEN_KEY = 'vav.home.setupHidden';

const videoName = (job: JobResponse) =>
  job.video_filename || job.video_path?.split(/[\\/]/).pop() || job.id.slice(0, 8);

const ago = (iso?: string | null) => {
  if (!iso) return '';
  const minutes = Math.round((Date.now() - new Date(iso).getTime()) / 60000);
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} h ago`;
  return new Date(iso).toLocaleDateString();
};

// Older servers send no readiness; then `available` is all there is.
const isReady = (p: PipelineDescriptor) =>
  p.readiness ? p.readiness.state === 'ready' : p.available !== false;

const STATUS_STYLE: Record<string, string> = {
  completed: 'bg-green-100 text-green-800 border-green-200 dark:bg-green-950 dark:text-green-200 dark:border-green-900',
  failed: 'bg-red-100 text-red-800 border-red-200 dark:bg-red-950 dark:text-red-200 dark:border-red-900',
  running: 'bg-blue-100 text-blue-800 border-blue-200 dark:bg-blue-950 dark:text-blue-200 dark:border-blue-900',
  pending: 'bg-yellow-100 text-yellow-800 border-yellow-200 dark:bg-yellow-950 dark:text-yellow-200 dark:border-yellow-900',
};

const Step = ({ done, children }: { done: boolean; children: React.ReactNode }) => (
  <li className="flex items-start gap-2 text-sm">
    {done ? (
      <Check className="h-4 w-4 mt-0.5 text-green-600 shrink-0" />
    ) : (
      <Circle className="h-4 w-4 mt-0.5 text-muted-foreground shrink-0" />
    )}
    <span className={done ? 'text-muted-foreground line-through' : ''}>{children}</span>
  </li>
);

function RecentRun({ job }: { job: JobResponse }) {
  const navigate = useNavigate();
  const [problem, setProblem] = useState<string | null>(null);
  const label = videoName(job);
  const pipelines = job.selected_pipelines?.join(', ') || 'default pipelines';

  const runAgain = async () => {
    setProblem(null);
    try {
      const next = await apiClient.rerunJob(job.id);
      navigate(`/jobs/${next.id}`);
    } catch (e) {
      setProblem(parseApiError(e).message);
    }
  };
  const fixAndRerun = () =>
    navigate('/jobs/new', { state: wizardState({ mode: 'rerun', jobId: job.id, label, ...settingsOf(job) }) });

  return (
    <li className="p-3 space-y-1">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <div className="min-w-0 flex-1 basis-48">
          <Link to={`/jobs/${job.id}`} className="font-medium hover:underline truncate block">
            {label}
          </Link>
          <div className="text-xs text-muted-foreground truncate">
            {pipelines} · {ago(job.completed_at ?? job.created_at)}
            {job.batch_name ? ` · batch ${job.batch_name}` : ''}
          </div>
        </div>
        <Badge variant="outline" className={STATUS_STYLE[job.status] ?? ''}>
          {job.status === 'running' ? `running ${Math.round(job.progress_percentage)}%` : job.status}
        </Badge>
        <div className="flex gap-1 shrink-0">
          {job.status === 'completed' && (
            <>
              <Button size="sm" asChild>
                <Link to={`/view/${job.id}`}>
                  <Eye className="h-4 w-4 mr-1" /> View
                </Link>
              </Button>
              <Button size="sm" variant="ghost" onClick={() => void runAgain()} title="Run again with the same settings">
                <RotateCcw className="h-4 w-4" />
                <span className="sr-only">Run again</span>
              </Button>
            </>
          )}
          {(job.status === 'failed' || job.status === 'cancelled') && (
            <Button size="sm" variant="outline" onClick={fixAndRerun}>
              <Pencil className="h-4 w-4 mr-1" /> Fix &amp; rerun
            </Button>
          )}
          {(job.status === 'pending' || job.status === 'running') && (
            <Button size="sm" variant="outline" asChild>
              <Link to={`/jobs/${job.id}`}>Open</Link>
            </Button>
          )}
        </div>
      </div>
      {job.status === 'failed' && job.error_message && (
        <p className="text-xs text-destructive line-clamp-2">{job.error_message}</p>
      )}
      {problem && <p className="text-xs text-destructive">{problem}</p>}
    </li>
  );
}

/**
 * The start page: is everything working, what happened lately, what to do next.
 * Results (what jobs produced, on this computer) and Datasets (videos to run,
 * on the server) are kept apart here as in the nav.
 */
export default function Home() {
  const navigate = useNavigate();
  const { capabilities, isLoading: connecting } = useServerCapabilitiesContext();
  const connected = !!capabilities;

  const jobs = useQuery({
    queryKey: ['dashboardJobs', 'recent'],
    queryFn: () => apiClient.getJobs(1, 20),
    enabled: connected,
    refetchInterval: 15000,
  });
  const catalog = useQuery({ ...pipelineCatalogQueryOptions(), enabled: connected });
  const datasets = useQuery({ queryKey: ['datasets'], queryFn: () => apiClient.listDatasets(), enabled: connected });
  const prompts = useQuery({ queryKey: ['prompts', '', false], queryFn: () => apiClient.listPrompts(), enabled: connected });

  const [resultsFolder, setResultsFolder] = useState<string | null>(null);
  const [savedResults, setSavedResults] = useState(0);
  useEffect(() => {
    void (async () => {
      setResultsFolder((await getRootDirHandle())?.name ?? null);
      setSavedResults(Object.keys(await getJobDatasetIndex()).length);
    })();
  }, []);

  const allJobs = jobs.data?.jobs ?? [];
  const recent = allJobs.slice(0, 6);
  const latestDone = allJobs.find((j) => j.status === 'completed');
  const pipelines = catalog.data?.catalog.pipelines ?? [];
  const ready = pipelines.filter(isReady).length;
  const recentDatasets = [...(datasets.data?.datasets ?? [])]
    .sort((a, b) => (b.last_used_at ?? b.created_at).localeCompare(a.last_used_at ?? a.created_at))
    .slice(0, 4);
  const starred = (prompts.data?.prompts ?? []).filter((p) => p.starred).slice(0, 4);

  const steps = {
    connected,
    pipelines: ready > 0,
    ranJob: allJobs.length > 0,
    sawResults: savedResults > 0,
  };
  const [setupHidden, setSetupHidden] = useState(() => {
    try {
      return localStorage.getItem(SETUP_HIDDEN_KEY) === '1';
    } catch {
      return false;
    }
  });
  const hideSetup = () => {
    setSetupHidden(true);
    try {
      localStorage.setItem(SETUP_HIDDEN_KEY, '1');
    } catch {
      /* private window: hidden for this visit only */
    }
  };
  const showSetup = !setupHidden && !jobs.isLoading && !Object.values(steps).every(Boolean);

  return (
    <div className="container mx-auto px-6 py-8 max-w-6xl space-y-6">
      <div className="flex items-baseline justify-between gap-4 flex-wrap">
        <h1 className="text-3xl font-bold">{APP_NAME}</h1>
        <div className="text-sm text-muted-foreground flex items-center gap-3">
          <span className="font-mono">viewer v{VERSION}</span>
          <a
            href={`${GITHUB_URL}/blob/master/CHANGELOG.md`}
            target="_blank"
            rel="noopener noreferrer"
            className="underline underline-offset-2 hover:text-foreground"
          >
            Release notes
          </a>
        </div>
      </div>

      {/* Status strip */}
      <Card className="p-3 flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
        <Link to="/settings" className="flex items-center gap-2 hover:underline">
          <Server className={`h-4 w-4 ${connected ? 'text-green-600' : 'text-muted-foreground'}`} />
          {connecting ? 'Connecting to the server…' : connected ? 'Server connected' : 'No server connected'}
          {capabilities?.version && <span className="font-mono text-muted-foreground">v{capabilities.version}</span>}
        </Link>
        {connected && catalog.data && (
          <span className="flex items-center gap-2">
            {ready} of {pipelines.length} pipelines ready
            {ready < pipelines.length && (
              <Link to="/jobs/new" className="underline underline-offset-2 text-primary">
                set up the others
              </Link>
            )}
          </span>
        )}
        <Link to="/results" className="flex items-center gap-2 hover:underline">
          <BookOpen className="h-4 w-4" />
          {resultsFolder
            ? `${savedResults} result${savedResults === 1 ? '' : 's'} saved on this computer`
            : 'No results folder chosen'}
        </Link>
        {!connected && !connecting && (
          <Button size="sm" variant="outline" asChild className="ml-auto">
            <Link to="/settings">Connect to a server</Link>
          </Button>
        )}
      </Card>

      {/* Primary actions */}
      <div className="grid gap-3 grid-cols-1 sm:grid-cols-3">
        <Button size="lg" className="h-auto py-4 justify-start gap-3" disabled={!connected} onClick={() => navigate('/jobs/new')}>
          <Play className="h-5 w-5" />
          <span className="text-left">
            <span className="block font-semibold">Run pipelines on videos</span>
            <span className="block text-xs opacity-80 font-normal">Start a new job on the server</span>
          </span>
        </Button>
        <Button
          size="lg"
          variant="outline"
          className="h-auto py-4 justify-start gap-3"
          disabled={!latestDone}
          onClick={() => latestDone && navigate(`/view/${latestDone.id}`)}
        >
          <Eye className="h-5 w-5" />
          <span className="text-left min-w-0">
            <span className="block font-semibold">Open latest results</span>
            <span className="block text-xs text-muted-foreground font-normal truncate">
              {latestDone ? videoName(latestDone) : 'No finished job yet'}
            </span>
          </span>
        </Button>
        <Button size="lg" variant="outline" className="h-auto py-4 justify-start gap-3" onClick={() => navigate('/viewer')}>
          <MonitorPlay className="h-5 w-5" />
          <span className="text-left">
            <span className="block font-semibold">View files from this computer</span>
            <span className="block text-xs text-muted-foreground font-normal">A video and its annotation files</span>
          </span>
        </Button>
      </div>

      {showSetup && (
        <Card className="p-4 border-primary/30 bg-primary/5">
          <div className="flex items-center justify-between mb-2">
            <div className="font-semibold">Getting set up</div>
            <Button size="sm" variant="ghost" onClick={hideSetup}>
              Hide
            </Button>
          </div>
          <ol className="space-y-1.5">
            <Step done={steps.connected}>
              Connect to a VideoAnnotator server (<Link className="underline" to="/settings">Settings</Link>)
            </Step>
            <Step done={steps.pipelines}>
              Install at least one pipeline (offered when you <Link className="underline" to="/jobs/new">start a job</Link>)
            </Step>
            <Step done={steps.ranJob}>Run your first job on a video</Step>
            <Step done={steps.sawResults}>
              Open its results; they are kept under <Link className="underline" to="/results">Results</Link> on this
              computer
            </Step>
          </ol>
          <p className="text-xs text-muted-foreground mt-3">
            No server yet? <Link className="underline" to="/results">Try the demos</Link> or read{' '}
            <Link className="underline" to="/getting-started">Getting started</Link>.
          </p>
        </Card>
      )}

      <div className="grid gap-6 grid-cols-1 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <div className="flex items-center justify-between p-4 pb-2">
            <div className="font-semibold">Recent runs</div>
            <Link to="/jobs" className="text-sm underline underline-offset-2">
              All jobs
            </Link>
          </div>
          {!connected ? (
            <p className="p-4 text-sm text-muted-foreground">Connect to a server to see its jobs.</p>
          ) : jobs.isLoading ? (
            <p className="p-4 text-sm text-muted-foreground flex items-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin" /> Loading…
            </p>
          ) : jobs.error ? (
            <p className="p-4 text-sm text-destructive">Couldn't load jobs: {parseApiError(jobs.error).message}</p>
          ) : recent.length === 0 ? (
            <p className="p-4 text-sm text-muted-foreground">
              No jobs yet. <Link className="underline" to="/jobs/new">Run pipelines on a video</Link> to start.
            </p>
          ) : (
            <ul className="divide-y">
              {recent.map((job) => (
                <RecentRun key={job.id} job={job} />
              ))}
            </ul>
          )}
        </Card>

        <div className="space-y-6">
          <Card className="p-4 space-y-2">
            <div className="flex items-center justify-between">
              <div className="font-semibold flex items-center gap-2">
                <Database className="h-4 w-4" /> Datasets
              </div>
              <Link to="/datasets" className="text-sm underline underline-offset-2">
                All
              </Link>
            </div>
            <p className="text-xs text-muted-foreground">Named lists of videos to run jobs on, shared on the server.</p>
            {recentDatasets.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                None yet. <Link className="underline" to="/datasets">Make one</Link> from a folder of videos.
              </p>
            ) : (
              <ul className="space-y-1">
                {recentDatasets.map((d) => (
                  <li key={d.id} className="flex items-center gap-2 text-sm">
                    <span className="truncate flex-1" title={d.name}>
                      {d.name}
                    </span>
                    <span className="text-xs text-muted-foreground shrink-0">
                      {d.video_manifest.length} video{d.video_manifest.length === 1 ? '' : 's'}
                    </span>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-7 px-2"
                      title={`Run pipelines on ${d.name}`}
                      onClick={() =>
                        navigate('/jobs/new', { state: wizardState({ mode: 'dataset', datasetId: d.id, name: d.name }) })
                      }
                    >
                      <Play className="h-3.5 w-3.5" />
                      <span className="sr-only">Run {d.name}</span>
                    </Button>
                  </li>
                ))}
              </ul>
            )}
          </Card>

          <Card className="p-4 space-y-2">
            <div className="flex items-center justify-between">
              <div className="font-semibold flex items-center gap-2">
                <MessageSquareText className="h-4 w-4" /> Starred prompts
              </div>
              <Link to="/prompts" className="text-sm underline underline-offset-2">
                All
              </Link>
            </div>
            {starred.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                Star the VLM prompts you reuse on the <Link className="underline" to="/prompts">Prompts</Link> page;
                try new ones in the <Link className="underline" to="/workbench">workbench</Link>.
              </p>
            ) : (
              <ul className="space-y-1">
                {starred.map((p) => (
                  <li key={p.sha256} className="flex items-center gap-2 text-sm">
                    <Star className="h-3.5 w-3.5 fill-yellow-400 text-yellow-500 shrink-0" />
                    <span className="truncate flex-1" title={p.text}>
                      {p.name || p.text.trim().split('\n')[0]}
                    </span>
                    <span className="text-xs text-muted-foreground shrink-0">{p.use_count}×</span>
                  </li>
                ))}
              </ul>
            )}
          </Card>

          <Card className="p-4 text-sm space-y-1">
            <div className="font-semibold flex items-center gap-2">
              <BookOpen className="h-4 w-4" /> Results on this computer
            </div>
            <p className="text-muted-foreground">
              {resultsFolder
                ? `${savedResults} job${savedResults === 1 ? '' : 's'} kept in “${resultsFolder}”, opening without a download.`
                : 'Opening a finished job keeps its results in a folder you choose.'}
            </p>
            <div className="flex gap-3">
              <Link className="underline underline-offset-2" to="/results">
                Open Results
              </Link>
              <Link className="underline underline-offset-2" to="/results">
                Demos
              </Link>
              <a
                className="underline underline-offset-2 inline-flex items-center gap-1"
                href={`${GITHUB_URL}#readme`}
                target="_blank"
                rel="noopener noreferrer"
              >
                Docs <ExternalLink className="h-3 w-3" />
              </a>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
