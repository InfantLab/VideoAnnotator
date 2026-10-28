import { useState, type ReactNode } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { Check, ChevronRight, Circle, FolderKanban, FolderOpen, KeyRound, Play, Server, Sparkles } from 'lucide-react';

import { apiClient } from '@/api/client';
import { KindIcons } from '@/components/runs/KindIcons';
import { RunList } from '@/components/runs/RunRow';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { useConnection } from '@/hooks/useConnection';
import { pipelineCatalogQueryOptions } from '@/hooks/usePipelineCatalog';
import { useRuns } from '@/hooks/useRuns';
import { DEMOS } from '@/lib/demos';
import { hrefOf, videoLink } from '@/lib/videosScope';
import { formatDay, formatWhen } from '@/lib/when';
import type { PipelineDescriptor } from '@/types/pipelines';
import type { RunListItem, VideoItem } from '@/types/videos';
import { APP_NAME, GITHUB_URL, VERSION } from '@/utils/version';

const SETUP_HIDDEN_KEY = 'vav.home.setupHidden';

// "1.7.0-dev.0" (viewer) and "1.7.0.dev0" (server) are the same version (UAT #13).
const sameVersion = (a: string, b: string) => a.replace(/[^a-z0-9]/gi, '') === b.replace(/[^a-z0-9]/gi, '');

// Older servers send no readiness; then `available` is all there is.
const isReady = (p: PipelineDescriptor) => (p.readiness ? p.readiness.state === 'ready' : p.available !== false);

const Step = ({ done, children }: { done: boolean; children: ReactNode }) => (
  <li className="flex items-start gap-2 text-sm">
    {done ? <Check className="mt-0.5 h-4 w-4 shrink-0 text-green-600" /> : <Circle className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />}
    <span className={done ? 'text-muted-foreground line-through' : ''}>{children}</span>
  </li>
);

function StartPoint({ icon, title, about, onClick, disabled }: { icon: ReactNode; title: string; about: string; onClick: () => void; disabled?: boolean }) {
  return (
    <Button size="lg" variant="outline" className="h-auto min-w-0 justify-start gap-3 whitespace-normal py-5" onClick={onClick} disabled={disabled}>
      {icon}
      <span className="min-w-0 text-left">
        <span className="block font-semibold">{title}</span>
        <span className="block text-xs font-normal text-muted-foreground">{about}</span>
      </span>
    </Button>
  );
}

const DAY_MS = 24 * 60 * 60 * 1000;
const RECENT_VIDEOS = 5;
const SHOWN_RUNS = 5;
const SHOWN_PROJECTS = 6;

/** Those going on and those finished in the last day; else the last one (viewer design: Home). */
function runsToShow(runs: RunListItem[], now = Date.now()): { runs: RunListItem[]; quiet: boolean } {
  const lately = runs.filter((r) => r.active || (r.finished_at && now - new Date(r.finished_at).getTime() < DAY_MS));
  if (lately.length) return { runs: lately.slice(0, SHOWN_RUNS), quiet: false };
  return { runs: runs.slice(0, 1), quiet: true };
}

function Section({ title, more, children }: { title: string; more?: { label: string; to: string }; children: ReactNode }) {
  return (
    <section className="space-y-2">
      <div className="flex items-baseline justify-between gap-4">
        <h2 className="text-lg font-semibold">{title}</h2>
        {more && (
          <Link to={more.to} className="flex items-center gap-0.5 text-sm text-muted-foreground hover:text-foreground hover:underline">
            {more.label} <ChevronRight className="h-3.5 w-3.5" aria-hidden="true" />
          </Link>
        )}
      </div>
      {children}
    </section>
  );
}

function RecentVideo({ video }: { video: VideoItem }) {
  const found = Object.entries(video.results)
    .filter(([, r]) => r.state === 'done')
    .map(([name]) => name);
  return (
    <li>
      <Link to={videoLink(video, { kind: 'all', id: '' })} className="flex min-w-0 items-center gap-3 px-3 py-2 hover:bg-muted/50" data-testid="recent-video">
        <span className="min-w-0 flex-1 truncate font-medium" title={video.path}>
          {video.name}
        </span>
        <KindIcons pipelines={found} />
        <span className="w-32 shrink-0 text-right text-xs tabular-nums text-muted-foreground">ran {formatWhen(video.last_run_at)}</span>
      </Link>
    </li>
  );
}

/** What's going on and where to start (viewer design: Home, D3): runs, recent videos, projects, start points, demos. */
export default function Home() {
  const navigate = useNavigate();
  const connection = useConnection();
  const connected = connection.state === 'connected';
  const serverVersion = connection.version;
  const catalog = useQuery({ ...pipelineCatalogQueryOptions(), enabled: connected });
  const known = useQuery({
    queryKey: ['videos', 'home'],
    queryFn: () => apiClient.listVideos({ sort: 'recent', limit: RECENT_VIDEOS }),
    enabled: connected,
    retry: false,
  });
  const runList = useRuns(20, connected);
  const pipelines = catalog.data?.catalog.pipelines ?? [];
  const ready = pipelines.filter(isReady).length;
  const videos = known.data?.counts.all ?? 0;

  const steps = { connected, pipelines: ready > 0, ran: videos > 0 };
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
  const showSetup = !setupHidden && !known.isLoading && !Object.values(steps).every(Boolean);
  // Only when something needs doing (viewer design: Home).
  const showStatus = !connected || (catalog.data !== undefined && ready === 0);
  const shown = runsToShow(runList.data?.runs ?? []);
  const lastRunOf = (project: string) => runList.data?.runs.find((r) => r.project?.id === project)?.started_at;
  const projects = (known.data?.counts.projects ?? []).slice(0, SHOWN_PROJECTS);

  return (
    <div className="container mx-auto max-w-5xl space-y-6 px-6 py-8">
      <div className="flex flex-wrap items-baseline justify-between gap-4">
        <h1 className="text-3xl font-bold">{APP_NAME}</h1>
        <div className="flex items-center gap-3 text-sm text-muted-foreground">
          <span className="font-mono">v{VERSION}</span>
          <a href={`${GITHUB_URL}/blob/master/CHANGELOG.md`} target="_blank" rel="noopener noreferrer" className="underline underline-offset-2 hover:text-foreground">
            Release notes
          </a>
        </div>
      </div>
      {showStatus && <Card className="flex flex-wrap items-center gap-x-6 gap-y-2 p-3 text-sm">
        <Link to="/settings" className="flex items-center gap-2 hover:underline">
          {connection.state === 'needs-key' || connection.state === 'key-refused' ? (
            <KeyRound className="h-4 w-4 text-amber-500" />
          ) : (
            <Server className={`h-4 w-4 ${connected ? 'text-green-600' : 'text-muted-foreground'}`} />
          )}
          {
            {
              checking: 'Connecting to the server…',
              connected: connection.user ? `Connected as ${connection.user.name}` : 'Server connected',
              'needs-key': 'The server needs your key',
              'key-refused': "The server didn't accept your key",
              'no-server': 'No server connected',
            }[connection.state]
          }
          {serverVersion && !sameVersion(serverVersion, VERSION) && (
            <span className="font-mono text-muted-foreground">server v{serverVersion}</span>
          )}
        </Link>
        {connected && catalog.data && (
          <span className="flex items-center gap-2">
            {ready} of {pipelines.length} pipelines ready
            {ready < pipelines.length && (
              <Link to="/pipelines" className="text-primary underline underline-offset-2">
                set up the others
              </Link>
            )}
          </span>
        )}
        {!connected && connection.state !== 'checking' && (
          <Button size="sm" variant="outline" asChild className="ml-auto">
            <Link to="/settings">{connection.state === 'no-server' ? 'Connect to a server' : 'Connect with your key'}</Link>
          </Button>
        )}
      </Card>}

      <div className="grid grid-cols-1 gap-3 md:grid-cols-3" data-testid="start-points">
        <StartPoint icon={<Sparkles className="h-5 w-5" />} title="Ask AI about a video" about="Try a question on a few moments, then the whole video" onClick={() => navigate('/ask')} disabled={!connected} />
        <StartPoint icon={<Play className="h-5 w-5" />} title="Run pipelines on videos" about="Speech, faces, people, scenes: one table for a folder" onClick={() => navigate('/run')} disabled={!connected} />
        <StartPoint icon={<FolderOpen className="h-5 w-5" />} title="Open files from your computer" about="A video and its result files, no server needed" onClick={() => navigate('/viewer')} />
      </div>

      {showSetup && (
        <Card className="border-primary/30 bg-primary/5 p-4" data-testid="setup-checklist">
          <div className="mb-2 flex items-center justify-between">
            <div className="font-semibold">Getting set up</div>
            <Button size="sm" variant="ghost" onClick={hideSetup}>
              Hide
            </Button>
          </div>
          <ol className="space-y-1.5">
            <Step done={steps.connected}>
              Connect to a VideoAnnotator server with your key (<Link className="underline" to="/settings">Settings</Link>)
            </Step>
            <Step done={steps.pipelines}>
              Install at least one pipeline (on the <Link className="underline" to="/pipelines">Pipelines</Link> page)
            </Step>
            <Step done={steps.ran}>
              Run pipelines on your first videos; they appear on <Link className="underline" to="/videos">Videos</Link>
            </Step>
          </ol>
          <p className="mt-3 text-xs text-muted-foreground">
            New here? <Link className="underline" to="/help">Help</Link> explains the words and where things are.
          </p>
        </Card>
      )}

      {connected && shown.runs.length > 0 && (
        <Section title="Runs" more={{ label: 'All runs', to: '/runs' }}>
          {shown.quiet && <p className="text-sm text-muted-foreground">Nothing running. Your last run:</p>}
          <RunList runs={shown.runs} />
        </Section>
      )}

      {connected && (known.data?.videos.length ?? 0) > 0 && (
        <Section title="Recent videos" more={{ label: `All ${videos} videos`, to: '/videos' }}>
          <ul className="divide-y rounded-md border">
            {known.data!.videos.map((video) => (
              <RecentVideo key={video.path} video={video} />
            ))}
          </ul>
        </Section>
      )}

      {connected && projects.length > 0 && (
        <Section title="Projects">
          <ul className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {projects.map((project) => {
              const last = lastRunOf(project.id);
              return (
                <li key={project.id}>
                  <Link to={hrefOf({ project: project.id })} className="flex min-w-0 items-center gap-2 rounded-md border px-3 py-2 hover:bg-muted/50" data-testid="home-project">
                    <FolderKanban className="h-4 w-4 shrink-0 text-muted-foreground" aria-hidden="true" />
                    <span className="truncate font-medium">{project.name}</span>
                    <span className="ml-auto shrink-0 text-xs text-muted-foreground">
                      {project.count} {project.count === 1 ? 'video' : 'videos'}
                      {last ? ` · last run ${formatDay(last)}` : ''}
                    </span>
                  </Link>
                </li>
              );
            })}
          </ul>
        </Section>
      )}

      <section className="space-y-2 border-t pt-6">
        <h2 className="text-lg font-semibold">Try it on a demo clip</h2>
        <p className="text-sm text-muted-foreground">Real results on a short clip, opened on the Video page. No server needed.</p>
        <div className="flex flex-wrap gap-2">
          {DEMOS.map((d) => (
            <Button key={d.key} variant="secondary" className="h-auto whitespace-normal py-2 text-left" asChild>
              <Link to={`/viewer?demo=${d.key}`}>
                <span>
                  <span className="font-semibold">{d.title}</span>
                  <span className="font-normal text-muted-foreground">: {d.description}</span>
                </span>
              </Link>
            </Button>
          ))}
        </div>
      </section>
    </div>
  );
}
