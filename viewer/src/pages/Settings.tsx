import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { PageHelp } from '@/components/PageHelp';
import { ExternalLink, KeyRound, Loader2, RefreshCw, Server as ServerIcon, Settings, Sparkles } from 'lucide-react';

import { TokenSetup } from '@/components/TokenSetup';
import { HuggingFaceTokenCard } from '@/components/HuggingFaceTokenSetup';
import { ModelServers } from '@/components/settings/ModelServers';
import { ServerFacts, ServerLocationsCard } from '@/components/settings/ServerFacts';
import { ThemeChoice } from '@/components/settings/ThemeChoice';
import { ServerDiagnostics } from '@/components/ServerDiagnostics';
import { VideosAndResultsCard } from '@/components/VideosAndResultsCard';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { RestartRequiredBanner } from '@/components/RestartRequiredBanner';

import { useCurrentUser } from '@/hooks/useCurrentUser';
import { useConnection } from '@/hooks/useConnection';
import { getApiBaseUrl } from '@/api/client';
import { API_TOKEN_STORAGE_KEY, API_URL_STORAGE_KEY, describeApiUrl } from '@/lib/apiConnection';
import { usePipelineCatalog, useRefreshPipelineCatalog } from '@/hooks/usePipelineCatalog';

import vavIcon from '@/assets/v-a-v.icon.png';

const CreateSettings = () => {
  // The address the viewer actually calls, also when it's kept relative (UAT #16).
  const [serverUrl, setServerUrl] = useState(() => describeApiUrl(getApiBaseUrl()));
  const [refreshing, setRefreshing] = useState(false);

  const connection = useConnection();
  const { isAdmin } = useCurrentUser();
  const { data: catalogData } = usePipelineCatalog();
  const refreshCatalog = useRefreshPipelineCatalog();

  useEffect(() => {
    const handleStorage = (event: StorageEvent) => {
      if (event.key === API_URL_STORAGE_KEY || event.key === API_TOKEN_STORAGE_KEY) {
        setServerUrl(describeApiUrl(getApiBaseUrl()));
      }
    };
    window.addEventListener('storage', handleStorage);
    return () => window.removeEventListener('storage', handleStorage);
  }, []);

  const refresh = async () => {
    setRefreshing(true);
    try {
      await refreshCatalog({ forceServerRefresh: true });
    } finally {
      setRefreshing(false);
    }
  };

  return (
    <div className="container mx-auto px-6 py-8 max-w-5xl space-y-6">
      <div className="flex items-center gap-3">
        <img src={vavIcon} alt="VideoAnnotator" className="h-8 w-8" />
        <div className="flex items-center gap-2">
          <Settings className="h-6 w-6" />
          <h2 className="text-2xl font-semibold">Settings</h2>
        </div>
      </div>
      <PageHelp page="settings">
        How this viewer is connected, where Ask AI's models run, and what the server runs on. What the words mean
        is on <Link to="/help" className="underline">Help</Link>.
      </PageHelp>
      <ThemeChoice />

      <Tabs defaultValue="connection" className="space-y-4">
        <TabsList className="h-auto flex-wrap">
          <TabsTrigger value="connection" className="flex items-center gap-2">
            <KeyRound className="h-4 w-4" aria-hidden="true" />
            Connection
          </TabsTrigger>
          <TabsTrigger value="ask" className="flex items-center gap-2">
            <Sparkles className="h-4 w-4" aria-hidden="true" />
            Ask AI
          </TabsTrigger>
          <TabsTrigger value="server" className="flex items-center gap-2">
            <ServerIcon className="h-4 w-4" aria-hidden="true" />
            This server
          </TabsTrigger>
        </TabsList>

        <TabsContent value="connection" className="space-y-4">
          <TokenSetup connection={connection} />
          <HuggingFaceTokenCard isAdmin={isAdmin} />
        </TabsContent>

        <TabsContent value="ask" className="space-y-4">
          <ModelServers isAdmin={isAdmin} />
        </TabsContent>

        <TabsContent value="server" className="space-y-4">
          <RestartRequiredBanner restartRequired={catalogData?.restartRequired ?? false} />
          <ServerFacts
            address={serverUrl}
            version={connection.version ?? (catalogData?.server?.version !== 'unknown' ? catalogData?.server?.version : undefined)}
            pipelines={catalogData?.catalog.pipelines}
          />
          <VideosAndResultsCard />
          <ServerLocationsCard />

          <details className="rounded-lg border p-4 text-sm" data-testid="technical-details">
            <summary className="cursor-pointer font-medium">Technical details</summary>
            <div className="mt-4 space-y-4">
              <div className="flex flex-wrap items-center gap-2">
                <Button size="sm" variant="outline" onClick={() => void refresh()} disabled={refreshing} className="gap-1">
                  {refreshing ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
                  Ask the server for its pipelines again
                </Button>
                <Button size="sm" variant="link" asChild>
                  <a href={`${serverUrl}/docs`} target="_blank" rel="noopener noreferrer" className="gap-1">
                    The server's API, for programmers <ExternalLink className="h-3 w-3" />
                  </a>
                </Button>
              </div>
              <ServerDiagnostics defaultOpen={false} />
            </div>
          </details>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default CreateSettings;
