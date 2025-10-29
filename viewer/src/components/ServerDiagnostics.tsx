/**
 * T057-T065: ServerDiagnostics component
 * Displays comprehensive server health information with auto-refresh
 */

import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/api/client';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  Server,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Cpu,
  HardDrive,
  Database,
  Clock,
} from 'lucide-react';
import {
  formatUptime,
  formatMemory,
  getWorkerStatusColor,
  getHealthStatusColor,
} from '@/lib/formatters';

interface ServerDiagnosticsProps {
  /** Optional className for custom styling */
  className?: string;
  /** Whether the diagnostics section should start expanded */
  defaultOpen?: boolean;
}

export function ServerDiagnostics({ className, defaultOpen = true }: ServerDiagnosticsProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  const [lastFetchTime, setLastFetchTime] = useState<number>(Date.now());

  // Fetch health data with auto-refresh when expanded
  const {
    data: health,
    isLoading,
    error,
    refetch,
    isFetching,
  } = useQuery({
    queryKey: ['server-health'],
    queryFn: async () => {
      const data = await apiClient.getEnhancedHealth();
      setLastFetchTime(Date.now());
      return data;
    },
    refetchInterval: isOpen ? 30000 : false, // Auto-refresh every 30s when expanded
    enabled: true, // Always enabled for initial load and proper loading states
    refetchOnMount: isOpen, // Refetch on mount if expanded
  });

  // Check if data is stale (> 2 minutes old)
  const [isStale, setIsStale] = useState(false);
  
  useEffect(() => {
    if (!isOpen) {
      setIsStale(false);
      return;
    }

    const checkStale = () => {
      const age = Date.now() - lastFetchTime;
      setIsStale(age > 120000); // 2 minutes
    };

    checkStale();
    const interval = setInterval(checkStale, 10000); // Check every 10 seconds

    return () => clearInterval(interval);
  }, [lastFetchTime, isOpen]);

  // Refetch when expanding
  useEffect(() => {
    if (isOpen) {
      refetch();
    }
  }, [isOpen, refetch]);

  const handleManualRefresh = () => {
    refetch();
  };

  // Status icon based on health status
  const StatusIcon = ({ status }: { status: string }) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle2 className="h-4 w-4 text-green-600 dark:text-green-400" />;
      case 'degraded':
        return <AlertTriangle className="h-4 w-4 text-yellow-600 dark:text-yellow-400" />;
      case 'unhealthy':
        return <XCircle className="h-4 w-4 text-red-600 dark:text-red-400" />;
      default:
        return <Server className="h-4 w-4 text-gray-600 dark:text-gray-400" />;
    }
  };

  return (
    <Card className={className}>
      <Collapsible open={isOpen} onOpenChange={setIsOpen}>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CollapsibleTrigger asChild>
              <Button variant="ghost" className="w-full justify-between p-0 hover:bg-transparent">
                <CardTitle className="flex items-center gap-2 text-lg">
                  <Server className="h-5 w-5" />
                  Server Diagnostics
                  {health && (
                    <Badge variant={health.status === 'healthy' ? 'default' : 'destructive'} className="ml-2">
                      {health.status}
                    </Badge>
                  )}
                </CardTitle>
                {isOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
              </Button>
            </CollapsibleTrigger>
          </div>
        </CardHeader>

        <CollapsibleContent>
          <CardContent className="space-y-4">
            {/* Stale data warning */}
            {isStale && !isFetching && (
              <Alert variant="default" className="border-yellow-500 dark:border-yellow-400">
                <AlertTriangle className="h-4 w-4 text-yellow-600 dark:text-yellow-400" />
                <AlertDescription>
                  Data may be stale. Last updated {Math.floor((Date.now() - lastFetchTime) / 60000)} minutes ago.
                </AlertDescription>
              </Alert>
            )}

            {/* Loading state */}
            {isLoading && (
              <div className="flex items-center justify-center py-8 text-muted-foreground">
                <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                Loading server diagnostics...
              </div>
            )}

            {/* Error state */}
            {error && (
              <Alert variant="destructive">
                <XCircle className="h-4 w-4" />
                <AlertDescription>
                  Failed to load server diagnostics. Please try again.
                </AlertDescription>
              </Alert>
            )}

            {/* Health data */}
            {health && !isLoading && (
              <div className="space-y-4">
                {/* Server Info */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-sm font-medium text-muted-foreground">Version</div>
                    <div className="text-lg font-semibold">
                      {health.version || health.api_version || health.videoannotator_version || 'Unknown'}
                    </div>
                  </div>
                  {health.uptime_seconds !== undefined && (
                    <div>
                      <div className="text-sm font-medium text-muted-foreground">Uptime</div>
                      <div className="flex items-center gap-2 text-lg">
                        <Clock className="h-4 w-4" />
                        {formatUptime(health.uptime_seconds)}
                      </div>
                    </div>
                  )}
                </div>

                <Separator />

                {/* GPU Status */}
                {health.gpu_status && (
                  <>
                    <div>
                      <div className="mb-2 flex items-center gap-2 text-sm font-medium">
                        <Cpu className="h-4 w-4" />
                        GPU Status
                      </div>
                      {health.gpu_status.available ? (
                        <div className="space-y-2 rounded-md border p-3">
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-muted-foreground">Device</span>
                            <span className="text-sm font-medium">{health.gpu_status.device_name || 'Unknown'}</span>
                          </div>
                          {health.gpu_status.cuda_version && (
                            <div className="flex items-center justify-between">
                              <span className="text-sm text-muted-foreground">CUDA Version</span>
                              <span className="text-sm font-medium">CUDA {health.gpu_status.cuda_version}</span>
                            </div>
                          )}
                          {health.gpu_status.memory_total !== undefined && health.gpu_status.memory_used !== undefined && (
                            <div className="flex items-center justify-between">
                              <span className="text-sm text-muted-foreground">Memory</span>
                              <span className="text-sm font-medium">
                                {(() => {
                                  const mem = formatMemory(health.gpu_status.memory_used, health.gpu_status.memory_total);
                                  return `${mem.used} / ${mem.total} (${mem.percentage}%)`;
                                })()}
                              </span>
                            </div>
                          )}
                        </div>
                      ) : (
                        <div className="rounded-md border border-dashed p-3 text-center text-sm text-muted-foreground">
                          GPU not available
                        </div>
                      )}
                    </div>
                    <Separator />
                  </>
                )}

                {/* Worker Info */}
                {health.worker_info && (
                  <>
                    <div>
                      <div className="mb-2 text-sm font-medium">Worker Status</div>
                      <div className="space-y-2 rounded-md border p-3">
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-muted-foreground">Status</span>
                          <Badge
                            variant="outline"
                            className={getWorkerStatusColor(health.worker_info.worker_status)}
                          >
                            {health.worker_info.worker_status}
                          </Badge>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-muted-foreground">Active Jobs</span>
                          <span className="text-sm font-medium">{health.worker_info.active_jobs}</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-muted-foreground">Queued Jobs</span>
                          <span className="text-sm font-medium">{health.worker_info.queued_jobs}</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-muted-foreground">Max Concurrent</span>
                          <span className="text-sm font-medium">{health.worker_info.max_concurrent_jobs}</span>
                        </div>
                      </div>
                    </div>
                    <Separator />
                  </>
                )}

                {/* Diagnostics */}
                {health.diagnostics && (
                  <div>
                    <div className="mb-2 text-sm font-medium">System Diagnostics</div>
                    <div className="space-y-2">
                      {/* Database */}
                      {health.diagnostics.database && (
                        <div className="rounded-md border p-3">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <Database className="h-4 w-4" />
                              <span className="text-sm font-medium">Database</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <StatusIcon status={health.diagnostics.database.status} />
                              <span className={`text-sm ${getHealthStatusColor(health.diagnostics.database.status)}`}>
                                {health.diagnostics.database.status}
                              </span>
                            </div>
                          </div>
                          {health.diagnostics.database.message && (
                            <div className="mt-2 text-xs text-muted-foreground">
                              {health.diagnostics.database.message}
                            </div>
                          )}
                        </div>
                      )}

                      {/* Storage */}
                      {health.diagnostics.storage && (
                        <div className="rounded-md border p-3">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <HardDrive className="h-4 w-4" />
                              <span className="text-sm font-medium">Storage</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <StatusIcon status={health.diagnostics.storage.status} />
                              <span className={`text-sm ${getHealthStatusColor(health.diagnostics.storage.status)}`}>
                                {health.diagnostics.storage.status}
                              </span>
                              {health.diagnostics.storage.disk_usage_percent !== undefined && (
                                <Badge variant="outline" className="ml-2">
                                  {health.diagnostics.storage.disk_usage_percent}% used
                                </Badge>
                              )}
                            </div>
                          </div>
                          {health.diagnostics.storage.message && (
                            <div className="mt-2 text-xs text-muted-foreground">
                              {health.diagnostics.storage.message}
                            </div>
                          )}
                        </div>
                      )}

                      {/* FFmpeg */}
                      {health.diagnostics.ffmpeg && (
                        <div className="flex items-center justify-between rounded-md border p-3">
                          <div className="flex items-center gap-2">
                            <Server className="h-4 w-4" />
                            <span className="text-sm font-medium">FFmpeg</span>
                          </div>
                          <div className="flex items-center gap-2">
                            {health.diagnostics.ffmpeg.available ? (
                              <>
                                <CheckCircle2 className="h-4 w-4 text-green-600 dark:text-green-400" />
                                {health.diagnostics.ffmpeg.version && (
                                  <span className="text-sm text-muted-foreground">
                                    v{health.diagnostics.ffmpeg.version}
                                  </span>
                                )}
                              </>
                            ) : (
                              <>
                                <XCircle className="h-4 w-4 text-red-600 dark:text-red-400" />
                                <span className="text-sm text-red-600 dark:text-red-400">Not available</span>
                              </>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Refresh button */}
                <div className="flex items-center justify-between pt-2">
                  <div className="text-xs text-muted-foreground">
                    {isFetching ? (
                      <span className="flex items-center gap-1">
                        <RefreshCw className="h-3 w-3 animate-spin" />
                        Refreshing...
                      </span>
                    ) : (
                      `Last updated: ${new Date(lastFetchTime).toLocaleTimeString()}`
                    )}
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleManualRefresh}
                    disabled={isFetching}
                    className="h-8"
                  >
                    <RefreshCw className={`mr-2 h-3 w-3 ${isFetching ? 'animate-spin' : ''}`} />
                    Refresh Now
                  </Button>
                </div>
              </div>
            )}
          </CardContent>
        </CollapsibleContent>
      </Collapsible>
    </Card>
  );
}
