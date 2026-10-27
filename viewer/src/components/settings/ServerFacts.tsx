// Settings → This server (UAT #16): the facts someone running the server asks
// about (what it runs on, how full it is, where it keeps things), in sentences.
import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { Check, Copy } from 'lucide-react';
import { apiClient } from '@/api/client';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { useSystemHealth } from '@/hooks/useSystemHealth';
import { formatMemory, formatUptime } from '@/lib/formatters';
import type { PipelineDescriptor } from '@/types/pipelines';
import type { SystemHealthResponse } from '@/types/system';

const size = (bytes: number) => formatMemory(bytes, bytes).used;

function Fact({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="min-w-0">
      <dt className="text-sm font-medium">{label}</dt>
      <dd className="text-sm text-muted-foreground break-words">{children}</dd>
    </div>
  );
}

/** "Linux", "Windows", "macOS" from Python's platform.platform(). */
const systemName = (platform: string) => {
  const first = platform.split('-')[0];
  return first === 'Darwin' || first === 'macOS' ? 'macOS' : first;
};

function gpuText(health: SystemHealthResponse): string {
  const gpu = health.gpu;
  if (gpu?.available && gpu.device_name) {
    return gpu.device_count && gpu.device_count > 1 ? `${gpu.device_name} (and ${gpu.device_count - 1} more)` : gpu.device_name;
  }
  return 'None found: models run on the processor, which is slower';
}

function runningText(health: SystemHealthResponse): string {
  const w = health.workers;
  if (!w) return 'Unknown';
  if (w.status !== 'running') return 'The worker that runs pipelines is stopped';
  const parts = [w.active_jobs > 0 && `${w.active_jobs} running`, w.queued_jobs > 0 && `${w.queued_jobs} waiting`].filter(Boolean);
  return parts.length ? `${parts.join(', ')} (${w.max_concurrent_workers} at a time)` : `Nothing (${w.max_concurrent_workers} at a time)`;
}

interface ServerFactsProps {
  address: string;
  version?: string;
  pipelines?: PipelineDescriptor[];
}

export function ServerFacts({ address, version, pipelines }: ServerFactsProps) {
  const { data: health, isLoading } = useSystemHealth({ enabled: true });
  const ready = pipelines?.filter((p) => (p.readiness ? p.readiness.state === 'ready' : p.available !== false)).length;

  return (
    <Card data-testid="server-facts">
      <CardHeader>
        <CardTitle className="text-base">This server</CardTitle>
        <CardDescription>What it runs on and what it's doing.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <dl className="grid gap-4 sm:grid-cols-2">
          <Fact label="Address">{address}</Fact>
          <Fact label="Version">{version ? `VideoAnnotator ${version}` : 'Unknown'}</Fact>
          <Fact label="Pipelines">
            {pipelines ? (
              <>
                {ready} of {pipelines.length} ready ·{' '}
                <Link to="/pipelines" className="underline">
                  see Pipelines
                </Link>
              </>
            ) : (
              <Skeleton className="mt-1 h-4 w-24" />
            )}
          </Fact>
          {isLoading && !health ? (
            <Skeleton className="h-10 w-full" />
          ) : health?.system ? (
            <>
              <Fact label="Running now">
                {runningText(health)} ·{' '}
                <Link to="/runs" className="underline">
                  see Runs
                </Link>
              </Fact>
              <Fact label="Computer">
                {systemName(health.system.platform)} · {health.system.cpu_count} processor cores ·{' '}
                {size(health.system.memory.total)} memory ({size(health.system.memory.available)} free)
              </Fact>
              <Fact label="Graphics card (GPU)">{gpuText(health)}</Fact>
              <Fact label="Disk">
                {size(health.system.disk.free)} free of {size(health.system.disk.total)}
              </Fact>
              <Fact label="Running for">{formatUptime(health.uptime_seconds)}</Fact>
            </>
          ) : null}
        </dl>
        {health?.gpu?.compatibility_warning && (
          <p className="text-sm text-amber-700 dark:text-amber-400">{health.gpu.compatibility_warning}</p>
        )}
      </CardContent>
    </Card>
  );
}

function CopyPath({ label, path }: { label: string; path: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="min-w-0">
      <dt className="text-sm font-medium">{label}</dt>
      <dd className="flex items-center gap-1">
        <span className="truncate font-mono text-xs text-muted-foreground" title={path}>
          {path}
        </span>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="h-6 w-6 shrink-0 p-0"
          aria-label={`Copy the ${label.toLowerCase()} location`}
          onClick={() =>
            navigator.clipboard.writeText(path).then(
              () => {
                setCopied(true);
                setTimeout(() => setCopied(false), 1500);
              },
              () => undefined,
            )
          }
        >
          {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
        </Button>
      </dd>
    </div>
  );
}

/** Where models, logs and the database are; results have their own card. */
export function ServerLocationsCard() {
  const locations = useQuery({
    queryKey: ['videoannotator', 'system', 'locations'],
    queryFn: () => apiClient.getServerLocations(),
    retry: false,
  });
  if (!locations.data) return null;
  const l = locations.data;
  return (
    <Card data-testid="server-locations">
      <CardHeader>
        <CardTitle className="text-base">Where the server keeps its own files</CardTitle>
        <CardDescription>The downloaded models, its log (to send when asking for help) and its database of runs.</CardDescription>
      </CardHeader>
      <CardContent>
        <dl className="grid gap-3">
          <CopyPath label="Models" path={l.models.display_path} />
          <CopyPath label="Logs" path={l.logs.display_path} />
          <CopyPath label="Database" path={l.database.display_path} />
        </dl>
      </CardContent>
    </Card>
  );
}
