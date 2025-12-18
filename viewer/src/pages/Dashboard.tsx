import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';

import { apiClient } from '@/api/client';
import type { JobListResponse, JobResponse } from '@/api/client';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Footer } from '@/components/Footer';
import { TokenStatusIndicator } from '@/components/TokenStatusIndicator';
import { ConnectionErrorBanner } from '@/components/ConnectionErrorBanner';
import { ErrorDisplay } from '@/components/ErrorDisplay';
import { useToast } from '@/hooks/use-toast';
import { useServerCapabilitiesContext } from '@/contexts/ServerCapabilitiesContext';
import { parseApiError } from '@/lib/errorHandling';
import { VERSION, GITHUB_URL, APP_NAME } from '@/utils/version';
import {
  getRootDirHandle,
  setRootDirHandle
} from '@/lib/localLibrary/libraryStore';
import { installBundledDemoDataset } from '@/lib/localLibrary/installDemoDataset';
import CreateJobs from '@/pages/CreateJobs';

type JobsMode = 'compact' | 'full';
const DASHBOARD_JOBS_MODE_KEY = 'vav.dashboard.jobsMode';

function isCorsOrNetworkError(error: Error): boolean {
  const message = error.message.toLowerCase();

  const isAuthError =
    message.includes('401') ||
    message.includes('403') ||
    message.includes('404') ||
    message.includes('unauthorized') ||
    message.includes('forbidden') ||
    message.includes('not found') ||
    message.includes('auth') ||
    message.includes('permission');

  if (isAuthError) return false;

  return (
    message.includes('cors') ||
    message.includes('fetch') ||
    message.includes('network') ||
    message.includes('failed to fetch') ||
    message.includes('networkerror') ||
    message.includes('timeout') ||
    message.includes('timed out') ||
    message.includes('access-control-allow-origin')
  );
}

function StatusBadge({ status }: { status: string }) {
  const className =
    status === 'completed'
      ? 'bg-green-100 text-green-800 border-green-200'
      : status === 'failed'
        ? 'bg-red-100 text-red-800 border-red-200'
        : status === 'running'
          ? 'bg-blue-100 text-blue-800 border-blue-200'
          : status === 'pending'
            ? 'bg-yellow-100 text-yellow-800 border-yellow-200'
            : 'bg-gray-100 text-gray-800 border-gray-200';

  return (
    <Badge variant="outline" className={className}>
      {status.toUpperCase()}
    </Badge>
  );
}

function RecentJobs() {
  const { data, isLoading, error, refetch, dataUpdatedAt } = useQuery<JobListResponse>({
    queryKey: ['dashboardJobs', 'recent'],
    queryFn: () => apiClient.getJobs(1, 5),
    refetchInterval: 30000,
    refetchOnWindowFocus: false
  });

  const lastCheckedText = useMemo(() => {
    if (!dataUpdatedAt) return 'Not checked yet';
    return `Last checked: ${new Date(dataUpdatedAt).toLocaleTimeString()}`;
  }, [dataUpdatedAt]);

  if (error) {
    return <ErrorDisplay error={parseApiError(error)} />;
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div className="text-xs text-muted-foreground">{lastCheckedText}</div>
        <Button variant="outline" size="sm" onClick={() => void refetch()} disabled={isLoading}>
          Refresh
        </Button>
      </div>

      {isLoading && !data ? (
        <div className="text-sm text-muted-foreground">Loading jobs…</div>
      ) : data?.jobs?.length ? (
        <div className="space-y-2">
          {data.jobs.map((job: JobResponse) => {
            // Defensive: server variants may return different field names.
            const jobRecord = job as unknown as Record<string, unknown>;
            const byKey = (key: string) => jobRecord[key];

            const videoFileNameCandidate = byKey('video_filename');
            const filenameCandidate = byKey('filename');
            const videoNameCandidate = byKey('video_name');
            const videoPathCandidate = byKey('video_path');

            const videoName =
              (typeof videoFileNameCandidate === 'string' && videoFileNameCandidate) ||
              (typeof filenameCandidate === 'string' && filenameCandidate) ||
              (typeof videoNameCandidate === 'string' && videoNameCandidate) ||
              (typeof videoPathCandidate === 'string' && (videoPathCandidate.split('/').pop() || videoPathCandidate)) ||
              'N/A';

            return (
              <div key={job.id} className="flex items-center justify-between gap-4 border rounded-md p-3">
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <div className="font-mono text-sm">{job.id.slice(0, 8)}…</div>
                    <StatusBadge status={job.status} />
                  </div>
                  <div className="text-sm text-muted-foreground truncate">{videoName}</div>
                </div>
                <div className="flex gap-2 shrink-0">
                  <Link to={`/view/${job.id}`}>
                    <Button variant="outline" size="sm">
                      View
                    </Button>
                  </Link>
                  <Link to={`/create/jobs/${job.id}`}>
                    <Button size="sm">Job</Button>
                  </Link>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="text-sm text-muted-foreground">No jobs found.</div>
      )}
    </div>
  );
}

export default function Dashboard() {
  const { toast } = useToast();
  const { error, refresh, capabilities, isLoading } = useServerCapabilitiesContext();

  const [libraryRootName, setLibraryRootName] = useState<string | null>(null);
  const [isPickingFolder, setIsPickingFolder] = useState(false);
  const [isInstallingDemo, setIsInstallingDemo] = useState(false);

  const [jobsMode, setJobsMode] = useState<JobsMode>(() => {
    const raw = localStorage.getItem(DASHBOARD_JOBS_MODE_KEY);
    return raw === 'full' ? 'full' : 'compact';
  });

  useEffect(() => {
    const run = async () => {
      const root = await getRootDirHandle();
      setLibraryRootName(root?.name ?? null);
    };
    void run();
  }, []);

  const releaseNotesUrl = `${GITHUB_URL}/blob/master/CHANGELOG.md`;

  const apiUrl =
    localStorage.getItem('videoannotator_api_url') ||
    import.meta.env.VITE_API_BASE_URL ||
    '';

  const showConnectionError = error && !isLoading && !capabilities && isCorsOrNetworkError(error);

  const setMode = (next: JobsMode) => {
    setJobsMode(next);
    localStorage.setItem(DASHBOARD_JOBS_MODE_KEY, next);
  };

  const pickLibraryFolder = async () => {
    if (!('showDirectoryPicker' in window)) {
      toast({
        title: 'Folder picker not supported',
        description: 'Your browser does not support the File System Access API required for the Local Library.',
        variant: 'destructive'
      });
      return;
    }

    setIsPickingFolder(true);
    try {
      // @ts-expect-error - File System Access API is not always present in TS lib.dom
      const handle: FileSystemDirectoryHandle = await window.showDirectoryPicker({ mode: 'readwrite' });
      await setRootDirHandle(handle);
      setLibraryRootName(handle.name);
    } catch (error) {
      // User cancelled or browser threw.
      console.warn('Folder pick cancelled/failed:', error);
    } finally {
      setIsPickingFolder(false);
    }
  };

  const installDemo = async () => {
    setIsInstallingDemo(true);
    try {
      const root = await getRootDirHandle();
      if (!root) {
        toast({
          title: 'Choose a library folder first',
          description: 'Pick a folder where demo data should be saved.',
          variant: 'destructive'
        });
        return;
      }

      const result = await installBundledDemoDataset({ rootDir: root });
      toast({
        title: 'Demo dataset installed',
        description: `Saved under “${result.entry.folderName}”.`,
        duration: 5000
      });
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : String(error);
      toast({
        title: 'Demo install failed',
        description: message,
        variant: 'destructive',
        duration: 8000
      });
    } finally {
      setIsInstallingDemo(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <div className="flex-1">
        <div className="container mx-auto px-4 py-6 space-y-6">
          <div className="flex items-start justify-between gap-4 flex-wrap">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">{APP_NAME}</h1>
              <div className="mt-1 flex items-center gap-3 text-sm text-muted-foreground flex-wrap">
                <span className="font-mono px-2 py-0.5 rounded bg-primary/10 text-primary">v{VERSION}</span>
                <a
                  href={releaseNotesUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline underline-offset-2"
                >
                  Release notes
                </a>
              </div>
              <p className="mt-3 text-gray-700 max-w-3xl">
                View local video + annotation datasets, and connect to VideoAnnotator to create jobs and download results into your library.
              </p>
            </div>

            <div className="flex items-center gap-2">
              <Link to="/viewer">
                <Button>Open Viewer</Button>
              </Link>
              <Link to="/create/jobs">
                <Button variant="outline">Control Panel</Button>
              </Link>
              <Link to="/library">
                <Button variant="outline">Local Library</Button>
              </Link>
            </div>
          </div>

          {showConnectionError && (
            <ConnectionErrorBanner error={error} apiUrl={apiUrl} onRetry={refresh} />
          )}

          <div className="grid gap-6 lg:grid-cols-2">
            <Card className="p-4 space-y-3">
              <div className="flex items-center justify-between gap-4 flex-wrap">
                <div>
                  <div className="font-semibold">Server connection</div>
                  <div className="text-sm text-muted-foreground">Configure API URL / token and validate access.</div>
                </div>
                <TokenStatusIndicator showDetails={true} />
              </div>
              <div className="text-xs text-muted-foreground">
                API URL: <span className="font-mono break-all">{apiUrl || '(not set)'}</span>
              </div>
              <div className="flex gap-2 flex-wrap">
                <Link to="/create/settings">
                  <Button variant="outline" size="sm">Settings</Button>
                </Link>
                <Button variant="outline" size="sm" onClick={() => refresh()}>
                  Refresh status
                </Button>
              </div>
            </Card>

            <Card className="p-4 space-y-3">
              <div className="flex items-center justify-between gap-4 flex-wrap">
                <div>
                  <div className="font-semibold">Local library</div>
                  <div className="text-sm text-muted-foreground">
                    Choose a folder where downloaded job results (and optional demos) are stored.
                  </div>
                </div>
                <div className="text-sm text-muted-foreground">
                  {libraryRootName ? (
                    <span>
                      Folder: <span className="font-medium">{libraryRootName}</span>
                    </span>
                  ) : (
                    <span>No folder selected</span>
                  )}
                </div>
              </div>

              <div className="flex gap-2 flex-wrap">
                <Button onClick={() => void pickLibraryFolder()} disabled={isPickingFolder}>
                  {isPickingFolder ? 'Choosing…' : 'Choose folder'}
                </Button>
                <Button
                  variant="outline"
                  onClick={() => void installDemo()}
                  disabled={!libraryRootName || isInstallingDemo}
                  title={!libraryRootName ? 'Choose a library folder first' : undefined}
                >
                  {isInstallingDemo ? 'Installing demo…' : 'Install demo dataset'}
                </Button>
              </div>

              <div className="text-xs text-muted-foreground">
                Demo installs a bundled sample dataset into your selected folder so you can open it locally without a server.
              </div>
            </Card>
          </div>

          <Card className="p-4 space-y-4">
            <div className="flex items-start justify-between gap-4 flex-wrap">
              <div>
                <div className="font-semibold">Jobs</div>
                <div className="text-sm text-muted-foreground">Recent jobs and job creation shortcuts.</div>
              </div>
              <div className="flex items-center gap-2">
                <Link to="/create/new">
                  <Button size="sm">New Job</Button>
                </Link>
                <Link to="/create/jobs">
                  <Button variant="outline" size="sm">All Jobs</Button>
                </Link>
              </div>
            </div>

            <div className="flex items-center justify-between gap-3 flex-wrap">
              <Tabs value={jobsMode} onValueChange={(value) => setMode(value as JobsMode)}>
                <TabsList>
                  <TabsTrigger value="compact">Compact</TabsTrigger>
                  <TabsTrigger value="full">Full table</TabsTrigger>
                </TabsList>
              </Tabs>
            </div>

            {jobsMode === 'compact' ? (
              <RecentJobs />
            ) : (
              <CreateJobs embedded />
            )}
          </Card>
        </div>
      </div>
      <Footer />
    </div>
  );
}
