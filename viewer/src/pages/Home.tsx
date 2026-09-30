import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';

import { apiClient } from '@/api/client';
import type { JobListResponse, JobResponse } from '@/api/client';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ErrorDisplay } from '@/components/ErrorDisplay';
import { useServerCapabilitiesContext } from '@/contexts/ServerCapabilitiesContext';
import { parseApiError } from '@/lib/errorHandling';
import { VERSION, GITHUB_URL, APP_NAME } from '@/utils/version';
import {
  getRootDirHandle,
  getJobDatasetIndex,
} from '@/lib/localLibrary/libraryStore';
import {
  Eye, Play, FileText, Upload, BookOpen, Briefcase,
  MonitorPlay, Server, ArrowRight, Sparkles,
} from 'lucide-react';
import viewerPreview from '@/assets/VideoAnnotationViewer.png';
import Jobs from '@/pages/Jobs';

/* ------------------------------------------------------------------ */
/*  Small sub-components                                               */
/* ------------------------------------------------------------------ */

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
    refetchOnWindowFocus: false,
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
                    <Button variant="outline" size="sm">View</Button>
                  </Link>
                  <Link to={`/jobs/${job.id}`}>
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

type JobsMode = 'compact' | 'full';
const DASHBOARD_JOBS_MODE_KEY = 'vav.dashboard.jobsMode';

/* ------------------------------------------------------------------ */
/*  Features data                                                      */
/* ------------------------------------------------------------------ */

const features = [
  {
    icon: <Eye className="w-6 h-6" />,
    title: 'Multimodal Overlays',
    description: 'View pose detection, facial emotions, audio sentiment, and events overlaid on video',
  },
  {
    icon: <Play className="w-6 h-6" />,
    title: 'Synchronized Timeline',
    description: 'Navigate through rich timeline with waveforms, motion graphs, and event markers',
  },
  {
    icon: <FileText className="w-6 h-6" />,
    title: 'JSON Annotations',
    description: 'Load analysis results from any computer vision or audio processing pipeline',
  },
];

/* ------------------------------------------------------------------ */
/*  Home Page                                                          */
/* ------------------------------------------------------------------ */

export default function Home() {
  const { capabilities } = useServerCapabilitiesContext();

  const [libraryRootName, setLibraryRootName] = useState<string | null>(null);
  const [datasetCount, setDatasetCount] = useState(0);

  const [jobsMode, setJobsMode] = useState<JobsMode>(() => {
    const raw = localStorage.getItem(DASHBOARD_JOBS_MODE_KEY);
    return raw === 'full' ? 'full' : 'compact';
  });

  const releaseNotesUrl = `${GITHUB_URL}/blob/master/CHANGELOG.md`;

  useEffect(() => {
    const run = async () => {
      const root = await getRootDirHandle();
      setLibraryRootName(root?.name ?? null);
      const index = await getJobDatasetIndex();
      setDatasetCount(Object.keys(index).length);
    };
    void run();
  }, []);

  const setMode = (next: JobsMode) => {
    setJobsMode(next);
    localStorage.setItem(DASHBOARD_JOBS_MODE_KEY, next);
  };

  const serverConnected = !!capabilities;

  return (
    <>
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-transparent to-accent/10" />
        <div className="container mx-auto px-6 py-12 relative">
          <div className="text-center max-w-4xl mx-auto mb-10">
            <div className="flex items-center justify-center gap-3 mb-4">
              <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                {APP_NAME}
              </h1>
            </div>
            <div className="flex items-center justify-center gap-3 mb-6">
              <span className="font-mono px-2 py-0.5 rounded bg-primary/10 text-primary text-sm">
                v{VERSION}
              </span>
              <a
                href={releaseNotesUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-muted-foreground underline underline-offset-2 hover:text-foreground"
              >
                Release notes
              </a>
            </div>
            <p className="text-lg text-muted-foreground mb-8 leading-relaxed max-w-2xl mx-auto">
              Review video annotations with pose detection, emotion recognition, audio analysis,
              and interactive timeline visualization.
            </p>

            {/* Primary CTAs */}
            <div className="flex flex-col sm:flex-row gap-3 justify-center">
              <Link to="/library">
                <Button size="lg" className="text-lg px-6 gap-2">
                  <BookOpen className="w-5 h-5" />
                  Open Library
                </Button>
              </Link>
              <Link to="/viewer">
                <Button size="lg" variant="outline" className="text-lg px-6 gap-2">
                  <Upload className="w-5 h-5" />
                  View Files
                </Button>
              </Link>
              {serverConnected && (
                <Link to="/jobs/new">
                  <Button size="lg" variant="outline" className="text-lg px-6 gap-2">
                    <Briefcase className="w-5 h-5" />
                    New Job
                  </Button>
                </Link>
              )}
            </div>
          </div>

          {/* Interface Preview */}
          <div className="max-w-5xl mx-auto">
            <Card className="overflow-hidden border-0 shadow-2xl">
              <img
                src={viewerPreview}
                alt="Video Annotation Viewer Interface"
                className="w-full h-auto"
              />
            </Card>
          </div>
        </div>
      </div>

      {/* Quick Status Cards */}
      <div className="container mx-auto px-6 py-8">
        <div className="grid gap-4 md:grid-cols-3 mb-8">
          {/* Server Status */}
          <Link to="/settings" className="block">
            <Card className="p-4 hover:shadow-md transition-shadow h-full">
              <div className="flex items-center gap-3 mb-2">
                <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${serverConnected ? 'bg-green-100 text-green-600' : 'bg-gray-100 text-gray-400'}`}>
                  <Server className="w-5 h-5" />
                </div>
                <div>
                  <div className="font-semibold text-sm">Server</div>
                  <div className={`text-xs ${serverConnected ? 'text-green-600' : 'text-muted-foreground'}`}>
                    {serverConnected ? 'Connected' : 'Not connected'}
                  </div>
                </div>
              </div>
            </Card>
          </Link>

          {/* Library Status */}
          <Link to="/library" className="block">
            <Card className="p-4 hover:shadow-md transition-shadow h-full">
              <div className="flex items-center gap-3 mb-2">
                <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${libraryRootName ? 'bg-blue-100 text-blue-600' : 'bg-gray-100 text-gray-400'}`}>
                  <BookOpen className="w-5 h-5" />
                </div>
                <div>
                  <div className="font-semibold text-sm">Library</div>
                  <div className="text-xs text-muted-foreground">
                    {libraryRootName
                      ? `${datasetCount} dataset${datasetCount !== 1 ? 's' : ''} in ${libraryRootName}`
                      : 'No folder selected'}
                  </div>
                </div>
              </div>
            </Card>
          </Link>

          {/* Getting Started */}
          <Link to="/getting-started" className="block">
            <Card className="p-4 hover:shadow-md transition-shadow h-full border-primary/20 bg-primary/5">
              <div className="flex items-center gap-3 mb-2">
                <div className="w-10 h-10 rounded-lg flex items-center justify-center bg-primary/10 text-primary">
                  <Sparkles className="w-5 h-5" />
                </div>
                <div>
                  <div className="font-semibold text-sm">Getting Started</div>
                  <div className="text-xs text-muted-foreground">
                    New here? Learn how to use VAV
                  </div>
                </div>
              </div>
            </Card>
          </Link>
        </div>

        {/* Features Grid */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-center mb-6">Powerful Analysis Features</h2>
          <div className="grid md:grid-cols-3 gap-6 max-w-4xl mx-auto">
            {features.map((feature, index) => (
              <Card key={index} className="p-5 text-center hover:shadow-md transition-shadow">
                <div className="w-12 h-12 mx-auto mb-3 bg-primary/10 rounded-lg flex items-center justify-center text-primary">
                  {feature.icon}
                </div>
                <h3 className="font-semibold mb-2">{feature.title}</h3>
                <p className="text-sm text-muted-foreground">{feature.description}</p>
              </Card>
            ))}
          </div>
        </div>

        {/* Recent Jobs Section */}
        <Card className="p-4 space-y-4">
          <div className="flex items-start justify-between gap-4 flex-wrap">
            <div>
              <div className="font-semibold">Recent Jobs</div>
              <div className="text-sm text-muted-foreground">Latest annotation jobs from your server.</div>
            </div>
            <div className="flex items-center gap-2">
              <Link to="/jobs/new">
                <Button size="sm">New Job</Button>
              </Link>
              <Link to="/jobs">
                <Button variant="outline" size="sm">All Jobs</Button>
              </Link>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant={jobsMode === 'compact' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setMode('compact')}
            >
              Compact
            </Button>
            <Button
              variant={jobsMode === 'full' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setMode('full')}
            >
              Full table
            </Button>
          </div>

          {jobsMode === 'compact' ? (
            <RecentJobs />
          ) : (
            <Jobs embedded />
          )}
        </Card>
      </div>
    </>
  );
}
