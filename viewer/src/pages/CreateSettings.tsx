import { useEffect, useMemo, useState } from 'react';
import { formatDistanceToNow } from 'date-fns';
import {
  Settings,
  Key,
  Server as ServerIcon,
  HelpCircle,
  Loader2,
  RefreshCw,
  Trash2,
  Database
} from 'lucide-react';

import { TokenSetup } from '@/components/TokenSetup';
import { ServerDiagnostics } from '@/components/ServerDiagnostics';
import { GPUInfo } from '@/components/GPUInfo';
import { WorkerInfo } from '@/components/WorkerInfo';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { ErrorDisplay } from '@/components/ErrorDisplay';
import { parseApiError } from '@/lib/errorHandling';
import type { ParsedError } from '@/types/api';
import { Separator } from '@/components/ui/separator';
import { Skeleton } from '@/components/ui/skeleton';

import { useTokenStatus } from '@/hooks/useTokenStatus';
import {
  usePipelineCatalog,
  useRefreshPipelineCatalog,
  useClearPipelineCatalogCache,
  useVideoAnnotatorServerInfo
} from '@/hooks/usePipelineCatalog';

import vavIcon from '@/assets/v-a-v.icon.png';

const TOKEN_KEY = 'videoannotator_api_token';
const URL_KEY = 'videoannotator_api_url';

const featureLabels: Record<string, string> = {
  pipelineCatalog: 'Pipeline Catalog',
  pipelineSchemas: 'Parameter Schemas',
  pipelineHealth: 'Pipeline Health',
  jobSSE: 'Job SSE',
  artifactListing: 'Artifact Listing'
};

const CreateSettings = () => {
  const [serverUrl, setServerUrl] = useState(() => localStorage.getItem(URL_KEY) || 'http://localhost:18011');
  const [isRefreshingCatalog, setIsRefreshingCatalog] = useState(false);
  const [isClearingCache, setIsClearingCache] = useState(false);

  const tokenStatus = useTokenStatus();
  const { data: serverInfoData, isLoading: serverInfoLoading, error: serverInfoError } = useVideoAnnotatorServerInfo();
  const { data: catalogData, isLoading: catalogLoading, error: catalogError } = usePipelineCatalog();
  const refreshCatalog = useRefreshPipelineCatalog();
  const clearPipelineCache = useClearPipelineCatalogCache();

  useEffect(() => {
    const handleStorage = (event: StorageEvent) => {
      if (!event.key || (event.key !== URL_KEY && event.key !== TOKEN_KEY)) {
        return;
      }
      setServerUrl(localStorage.getItem(URL_KEY) || 'http://localhost:18011');
    };

    window.addEventListener('storage', handleStorage);
    return () => {
      window.removeEventListener('storage', handleStorage);
    };
  }, []);

  const serverDetails = useMemo(() => {
    if (catalogData?.server) return catalogData.server;
    if (serverInfoData) return serverInfoData;
    return null;
  }, [catalogData, serverInfoData]);

  const featureFlags = serverDetails?.features || {};
  const pipelineCount = catalogData?.catalog.pipelines.length ?? 0;

  const pipelineGroups = useMemo(() => {
    if (!catalogData?.catalog.pipelines?.length) {
      return [];
    }

    const groups = new Map<string, typeof catalogData.catalog.pipelines>();

    catalogData.catalog.pipelines.forEach((pipeline) => {
      const groupKey = pipeline.group || 'Other';
      const existing = groups.get(groupKey) ?? [];
      existing.push(pipeline);
      groups.set(groupKey, existing);
    });

    return Array.from(groups.entries())
      .map(([group, pipelines]) => ({
        group,
        pipelines: pipelines.slice().sort((a, b) => a.name.localeCompare(b.name))
      }))
      .sort((a, b) => a.group.localeCompare(b.group));
  }, [catalogData]);

  const catalogUpdatedAt = useMemo(() => {
    const timestamp = catalogData?.catalog.generatedAt || serverDetails?.lastUpdated;
    if (!timestamp) return null;

    try {
      return formatDistanceToNow(new Date(timestamp), { addSuffix: true });
    } catch (error) {
      console.warn('Unable to format catalog timestamp:', error);
      return null;
    }
  }, [catalogData, serverDetails]);

  const handleRefreshCatalog = async () => {
    setIsRefreshingCatalog(true);
    try {
      await refreshCatalog({ forceServerRefresh: true });
      tokenStatus.refresh();
      setServerUrl(localStorage.getItem(URL_KEY) || 'http://localhost:18011');
    } finally {
      setIsRefreshingCatalog(false);
    }
  };

  const handleClearCache = async () => {
    setIsClearingCache(true);
    try {
      await clearPipelineCache();
      await refreshCatalog({ forceServerRefresh: true });
    } finally {
      setIsClearingCache(false);
    }
  };

  const tokenBadgeVariant = tokenStatus.isValid ? 'secondary' : tokenStatus.error ? 'destructive' : 'outline';
  const tokenStatusLabel = tokenStatus.isLoading
    ? 'Checking…'
    : tokenStatus.isValid
      ? 'Token Valid'
      : tokenStatus.error
        ? 'Token Error'
        : 'Not Validated';

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <img src={vavIcon} alt="VideoAnnotator" className="h-8 w-8" />
        <div className="flex items-center gap-2">
          <Settings className="h-6 w-6" />
          <h2 className="text-2xl font-semibold">Settings</h2>
        </div>
      </div>

      <Tabs defaultValue="api" className="space-y-4">
        <TabsList>
          <TabsTrigger value="api" className="flex items-center gap-2">
            <Key className="h-4 w-4" />
            API Configuration
          </TabsTrigger>
          <TabsTrigger value="server" className="flex items-center gap-2">
            <ServerIcon className="h-4 w-4" />
            Server Info
          </TabsTrigger>
          <TabsTrigger value="help" className="flex items-center gap-2">
            <HelpCircle className="h-4 w-4" />
            Help
          </TabsTrigger>
        </TabsList>

        <TabsContent value="api">
          <TokenSetup />
        </TabsContent>

        <TabsContent value="server" className="space-y-4">
          {(serverInfoError || catalogError || (tokenStatus.error && !tokenStatus.isLoading)) && (
            <ErrorDisplay
              error={parseApiError(
                catalogError || serverInfoError || tokenStatus.error || 'Server diagnostics issue'
              )}
            />
          )}

          <Card>
            <CardHeader>
              <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                <div>
                  <CardTitle>Server Integration Status</CardTitle>
                  <CardDescription>Live VideoAnnotator connection and pipeline capabilities</CardDescription>
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleClearCache}
                    disabled={isClearingCache || catalogLoading}
                    className="gap-1"
                  >
                    {isClearingCache ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
                    Clear Cache
                  </Button>
                  <Button
                    size="sm"
                    onClick={handleRefreshCatalog}
                    disabled={isRefreshingCatalog}
                    className="gap-1"
                  >
                    {isRefreshingCatalog ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <RefreshCw className="h-4 w-4" />
                    )}
                    Refresh
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <p className="font-medium text-sm">Server URL</p>
                  {catalogLoading && serverInfoLoading ? (
                    <Skeleton className="mt-1 h-4 w-40" />
                  ) : (
                    <p className="text-sm text-muted-foreground break-all">{serverUrl}</p>
                  )}
                </div>
                <div>
                  <p className="font-medium text-sm">Token Status</p>
                  <div className="mt-1 flex items-center gap-2">
                    <Badge variant={tokenBadgeVariant}>{tokenStatusLabel}</Badge>
                    {tokenStatus.user && !tokenStatus.isLoading && (
                      <span className="text-xs text-muted-foreground">{tokenStatus.user}</span>
                    )}
                  </div>
                </div>
                <div>
                  <p className="font-medium text-sm">Server Version</p>
                  {serverInfoLoading && !serverDetails ? (
                    <Skeleton className="mt-1 h-4 w-24" />
                  ) : (
                    <p className="text-sm text-muted-foreground">
                      {serverDetails?.version || 'Unknown'}
                    </p>
                  )}
                </div>
                <div>
                  <p className="font-medium text-sm">Pipelines Detected</p>
                  {catalogLoading ? (
                    <Skeleton className="mt-1 h-4 w-16" />
                  ) : (
                    <div className="mt-1 flex items-center gap-2">
                      <Badge variant="secondary" className="gap-1">
                        <Database className="h-3.5 w-3.5" /> {pipelineCount}
                      </Badge>
                      {catalogUpdatedAt && (
                        <span className="text-xs text-muted-foreground">Updated {catalogUpdatedAt}</span>
                      )}
                    </div>
                  )}
                </div>
              </div>

              <div>
                <p className="font-medium text-sm mb-2">Server Features</p>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(featureLabels).map(([key, label]) => {
                    const enabled = (featureFlags as Record<string, boolean | undefined>)[key] ?? false;
                    return (
                      <Badge key={key} variant={enabled ? 'secondary' : 'outline'} className="text-xs">
                        {label}
                      </Badge>
                    );
                  })}
                </div>
              </div>

              <Separator />

              <div>
                <p className="font-medium text-sm mb-3">Pipeline Catalog Overview</p>
                {catalogLoading ? (
                  <div className="space-y-2">
                    <Skeleton className="h-5 w-32" />
                    <Skeleton className="h-20 w-full" />
                    <Skeleton className="h-20 w-full" />
                  </div>
                ) : pipelineGroups.length === 0 ? (
                  <p className="text-sm text-muted-foreground">No pipelines detected. Refresh once the server is online.</p>
                ) : (
                  <div className="space-y-4">
                    {pipelineGroups.map(({ group, pipelines }) => (
                      <div key={group} className="space-y-2">
                        <div className="flex items-center justify-between">
                          <h4 className="text-sm font-semibold">{group}</h4>
                          <Badge variant="outline" className="text-xs">
                            {pipelines.length} pipelines
                          </Badge>
                        </div>
                        <div className="space-y-1.5">
                          {pipelines.map((pipeline) => (
                            <div
                              key={pipeline.id}
                              className="rounded-md border bg-muted/40 p-3"
                            >
                              <div className="flex flex-wrap items-start justify-between gap-2">
                                <div>
                                  <p className="text-sm font-medium">{pipeline.name}</p>
                                  {pipeline.description && (
                                    <p className="text-xs text-muted-foreground">
                                      {pipeline.description}
                                    </p>
                                  )}
                                </div>
                                <div className="flex flex-wrap items-center gap-2">
                                  {pipeline.version && (
                                    <Badge variant="secondary" className="text-[10px]">
                                      v{pipeline.version}
                                    </Badge>
                                  )}
                                  {pipeline.model && (
                                    <Badge variant="outline" className="text-[10px]">
                                      {pipeline.model}
                                    </Badge>
                                  )}
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Server Diagnostics */}
          <ServerDiagnostics className="mt-6" defaultOpen={false} />

          {/* GPU Information */}
          <GPUInfo />

          {/* Worker Information */}
          <WorkerInfo />

          <Card>
            <CardHeader>
              <CardTitle>Server Debugging Tools</CardTitle>
              <CardDescription>
                Use these tools to test your server connection and debug API issues
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="font-medium">Server URL</p>
                    <p className="text-muted-foreground break-all">{serverUrl}</p>
                  </div>
                  <div>
                    <p className="font-medium">Token</p>
                    <p className="text-muted-foreground break-all">
                      {localStorage.getItem(TOKEN_KEY) ? 'Configured' : 'Not configured'}
                    </p>
                  </div>
                </div>

                <div className="p-3 bg-muted rounded-md">
                  <p className="font-mono text-sm">python scripts/test_api_quick.py</p>
                  <p className="text-xs text-muted-foreground mt-1">
                    Comprehensive API endpoint testing
                  </p>
                </div>

                <div className="p-3 bg-muted rounded-md">
                  <p className="font-mono text-sm">VideoAnnotatorDebug.runAllTests()</p>
                  <p className="text-xs text-muted-foreground mt-1">
                    Browser console debugging suite (paste in browser dev tools)
                  </p>
                </div>

                <div className="pt-3 border-t">
                  <h4 className="font-medium text-sm mb-2">API Documentation</h4>
                  <Button variant="link" className="pl-0 h-auto" asChild>
                    <a
                      href={`${serverUrl}/docs`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm"
                    >
                      📖 Open Interactive API Docs ({serverUrl}/docs)
                    </a>
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="help" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Getting Help</CardTitle>
              <CardDescription>
                Resources and support for VideoAnnotator configuration
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-4">
                <div>
                  <h4 className="font-medium">Common Issues</h4>
                  <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground mt-2">
                    <li>Token authentication errors: Check your token is valid and not expired</li>
                    <li>Connection refused: Ensure VideoAnnotator server is running</li>
                    <li>404 errors: Server may not have all endpoints implemented yet</li>
                    <li>CORS issues: Check server allows requests from this domain</li>
                  </ul>
                </div>

                <div>
                  <h4 className="font-medium">Getting Your Token</h4>
                  <ol className="list-decimal list-inside space-y-1 text-sm text-muted-foreground mt-2">
                    <li>Contact your VideoAnnotator server administrator</li>
                    <li>
                      For development, use the default token:
                      <code className="bg-muted px-1 rounded">dev-token</code>
                    </li>
                    <li>Check server documentation for token generation instructions</li>
                  </ol>
                </div>

                <div className="pt-4 border-t">
                  <Button variant="link" className="pl-0" asChild>
                    <a
                      href="https://github.com/InfantLab/VideoAnnotator/blob/master/docs/CLIENT_TOKEN_SETUP_GUIDE.md"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      📖 View Complete Token Setup Guide
                    </a>
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default CreateSettings;
