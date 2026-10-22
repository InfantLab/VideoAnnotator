// Settings → Model servers (VideoAnnotator spec 033): the servers Ask AI and
// runs can send frames to, adding and checking one, and the remote switch.
// A key typed here goes to the server and is never shown again.
import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Loader2, Plus, Server, Trash2 } from 'lucide-react';
import { apiClient } from '@/api/client';
import { APIError, apiErrorEnvelope } from '@/api/handleError';
import { EndpointBadge } from '@/components/ask/EndpointBadge';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { userMessage } from '@/lib/userMessage';
import { KIND_LABELS, KIND_SHORT, keyText, MODEL_SERVERS_QUERY_KEY } from '@/lib/modelServers';
import type { ModelServer, ModelServerCheck, ModelServerKind } from '@/types/ask';



type KeyMode = 'none' | 'key' | 'env';

const errorText = (error: unknown): string => {
  const hint = apiErrorEnvelope(error).hint;
  return hint ? `${userMessage(error)} ${hint}` : userMessage(error);
};

function ServerRow({ server, onRemoved }: { server: ModelServer; onRemoved: () => void }) {
  const [confirming, setConfirming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const remove = async () => {
    try {
      await apiClient.removeModelServer(server.id);
      onRemoved();
    } catch (e) {
      setError(errorText(e));
    }
  };
  return (
    <li className="space-y-1 rounded border p-3" data-testid="model-server">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <span className="font-medium">{server.name}</span>{' '}
          <span className="text-xs text-muted-foreground">{KIND_SHORT[server.kind]}</span>
          <div className="font-mono text-xs text-muted-foreground">{server.base_url}</div>
        </div>
        <EndpointBadge
          endpoint={{ host: server.endpoint.host, class: server.endpoint.class, confirmed: server.endpoint.confirmed }}
          model={server.name}
          showNote={false}
        />
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
        <span>
          {keyText(server)}
          {server.endpoint.before_switch && ' · Confirmed before remote servers needed switching on'}
          {server.endpoint.class === 'remote' && !server.endpoint.sendable && !server.endpoint.confirmed && (
            <> · frames go there only after someone confirms it in Ask AI</>
          )}
        </span>
        {server.removable &&
          (confirming ? (
            <span className="flex gap-1">
              <Button size="sm" variant="destructive" onClick={() => void remove()}>
                Remove {server.name}
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setConfirming(false)}>
                Keep
              </Button>
            </span>
          ) : (
            <Button size="sm" variant="ghost" className="gap-1" onClick={() => setConfirming(true)}>
              <Trash2 className="h-3.5 w-3.5" /> Remove
            </Button>
          ))}
      </div>
      {error && <p className="text-xs text-destructive">{error}</p>}
    </li>
  );
}

function AddServer({ onAdded }: { onAdded: () => void }) {
  const [name, setName] = useState('');
  const [kind, setKind] = useState<ModelServerKind>('openai_compatible');
  const [baseUrl, setBaseUrl] = useState('');
  const [keyMode, setKeyMode] = useState<KeyMode>('none');
  const [key, setKey] = useState('');
  const [envName, setEnvName] = useState('');
  const [check, setCheck] = useState<ModelServerCheck | null>(null);
  const [anyway, setAnyway] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const changed = () => {
    setCheck(null);
    setAnyway(false);
    setError(null);
  };

  const checking = useMutation({
    mutationFn: () =>
      apiClient.checkModelServer({
        kind,
        base_url: baseUrl.trim(),
        api_key: keyMode === 'key' && key ? key : null,
      }),
    onSuccess: (result) => {
      setCheck(result);
      setError(null);
    },
    onError: (e) => setError(errorText(e)),
  });

  const adding = useMutation({
    mutationFn: () =>
      apiClient.addModelServer({
        name: name.trim(),
        kind,
        base_url: baseUrl.trim(),
        api_key: keyMode === 'key' && key ? key : null,
        api_key_env: keyMode === 'env' && envName ? envName.trim() : null,
      }),
    onSuccess: () => {
      setKey('');
      setName('');
      setBaseUrl('');
      setEnvName('');
      setKeyMode('none');
      changed();
      onAdded();
    },
    onError: (e) => {
      setKey('');
      setError(errorText(e));
    },
  });

  const passed = check?.outcome === 'models';
  const canAdd = Boolean(name.trim() && baseUrl.trim()) && (passed || anyway) && !adding.isPending;

  return (
    <form
      className="space-y-3 rounded border p-3"
      onSubmit={(e) => {
        e.preventDefault();
        if (canAdd) adding.mutate();
      }}
      data-testid="add-model-server"
    >
      <div className="grid gap-3 md:grid-cols-2">
        <div className="space-y-1">
          <Label htmlFor="server-name">Name</Label>
          <Input id="server-name" value={name} placeholder="lab-gpu" onChange={(e) => setName(e.target.value)} />
        </div>
        <div className="space-y-1">
          <Label htmlFor="server-kind">Kind</Label>
          <Select
            value={kind}
            onValueChange={(v) => {
              setKind(v as ModelServerKind);
              changed();
            }}
          >
            <SelectTrigger id="server-kind">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {(Object.keys(KIND_LABELS) as ModelServerKind[]).map((k) => (
                <SelectItem key={k} value={k}>
                  {KIND_LABELS[k]}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>
      <div className="space-y-1">
        <Label htmlFor="server-address">Address</Label>
        <Input
          id="server-address"
          value={baseUrl}
          placeholder={kind === 'ollama' ? 'http://192.168.1.20:11434' : 'http://127.0.0.1:8080/v1'}
          onChange={(e) => {
            setBaseUrl(e.target.value);
            changed();
          }}
        />
      </div>
      <div className="space-y-2">
        <Label>Key</Label>
        <RadioGroup value={keyMode} onValueChange={(v) => setKeyMode(v as KeyMode)} className="flex flex-wrap gap-4">
          <label className="flex items-center gap-2 text-sm">
            <RadioGroupItem value="none" /> None
          </label>
          <label className="flex items-center gap-2 text-sm">
            <RadioGroupItem value="key" /> A key
          </label>
          <label className="flex items-center gap-2 text-sm">
            <RadioGroupItem value="env" /> Use an environment variable
          </label>
        </RadioGroup>
        {keyMode === 'key' && (
          <Input
            type="password"
            autoComplete="off"
            aria-label="Key"
            value={key}
            onChange={(e) => {
              setKey(e.target.value);
              changed();
            }}
          />
        )}
        {keyMode === 'env' && (
          <Input
            aria-label="Environment variable"
            placeholder="LAB_GPU_KEY"
            value={envName}
            onChange={(e) => setEnvName(e.target.value)}
          />
        )}
      </div>

      {check && (
        <div role="status" className={`space-y-1 text-sm ${passed ? 'text-green-700 dark:text-green-400' : 'text-amber-700 dark:text-amber-400'}`}>
          <p>{check.message}</p>
          {check.hint && <p className="text-muted-foreground">{check.hint}</p>}
          <div className="flex flex-wrap gap-2">
            {check.suggest_base_url && (
              <Button
                type="button"
                size="sm"
                variant="outline"
                onClick={() => {
                  setBaseUrl(check.suggest_base_url ?? '');
                  changed();
                }}
              >
                Use {check.suggest_base_url}
              </Button>
            )}
            {check.suggest_kind && check.suggest_kind !== kind && (
              <Button
                type="button"
                size="sm"
                variant="outline"
                onClick={() => {
                  setKind(check.suggest_kind as ModelServerKind);
                  changed();
                }}
              >
                Use the {KIND_SHORT[check.suggest_kind as ModelServerKind]} kind
              </Button>
            )}
            {!passed && (
              <Button type="button" size="sm" variant="ghost" onClick={() => setAnyway(true)} disabled={anyway}>
                {anyway ? 'Will add anyway' : 'Add anyway'}
              </Button>
            )}
          </div>
        </div>
      )}
      {error && <p className="text-sm text-destructive">{error}</p>}

      <div className="flex gap-2">
        <Button type="button" variant="outline" disabled={!baseUrl.trim() || checking.isPending} onClick={() => checking.mutate()}>
          {checking.isPending && <Loader2 className="mr-1 h-4 w-4 animate-spin" />}
          Check
        </Button>
        <Button type="submit" disabled={!canAdd}>
          Add
        </Button>
      </div>
    </form>
  );
}

function RemoteSwitch({ allowed, fixed, warning, isAdmin }: { allowed: boolean; fixed: boolean; warning: string; isAdmin: boolean | 'unknown' }) {
  const queryClient = useQueryClient();
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const change = useMutation({
    mutationFn: (on: boolean) => apiClient.setRemoteModelServers(on),
    onSuccess: () => {
      setError(null);
      void queryClient.invalidateQueries({ queryKey: MODEL_SERVERS_QUERY_KEY });
    },
    onError: (e) =>
      setError(
        e instanceof APIError && (e.status === 401 || e.status === 403)
          ? 'Only an administrator of this server can change this.'
          : errorText(e),
      ),
  });
  return (
    <div className="space-y-1 rounded border p-3" data-testid="remote-switch">
      <div className="flex items-center justify-between gap-3">
        <p className="text-sm">
          {allowed
            ? 'Remote model servers are on: servers outside this computer and your local network can be used, each once someone confirms it.'
            : 'Remote model servers are off: only servers on this computer or your local network can be used.'}
        </p>
        {isAdmin !== false && (
          <Switch
            aria-label="Remote model servers"
            checked={allowed}
            disabled={fixed || change.isPending}
            onCheckedChange={(on) => (on ? setAsking(true) : change.mutate(false))}
          />
        )}
      </div>
      {fixed && <p className="text-xs text-muted-foreground">Set by this server's configuration.</p>}
      {error && <p className="text-xs text-destructive">{error}</p>}
      <AlertDialog open={asking} onOpenChange={setAsking}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Switch on remote model servers?</AlertDialogTitle>
            <AlertDialogDescription>{warning}</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Keep them off</AlertDialogCancel>
            <AlertDialogAction onClick={() => change.mutate(true)}>Switch on</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

/** The model servers card on Settings. */
export function ModelServers({ isAdmin }: { isAdmin: boolean | 'unknown' }) {
  const queryClient = useQueryClient();
  const [adding, setAdding] = useState(false);
  const servers = useQuery({ queryKey: MODEL_SERVERS_QUERY_KEY, queryFn: () => apiClient.listModelServers(), retry: false });
  const refresh = () => {
    void queryClient.invalidateQueries({ queryKey: MODEL_SERVERS_QUERY_KEY });
    void queryClient.invalidateQueries({ queryKey: ['videoannotator', 'vlm', 'models'] });
  };

  return (
    <Card data-testid="model-servers">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Server className="h-5 w-5" /> Model servers
        </CardTitle>
        <CardDescription>
          Where Ask AI and runs send frames to be answered. This server's own Ollama is always listed; add the
          llama.cpp, vLLM or LM Studio server your lab already runs.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {servers.isLoading && <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />}
        {servers.isError && <p className="text-sm text-destructive">Couldn't list the model servers: {userMessage(servers.error)}</p>}
        {servers.data && (
          <>
            <RemoteSwitch
              allowed={servers.data.remote.allowed}
              fixed={servers.data.remote.fixed_by_environment}
              warning={servers.data.remote.warning ?? ''}
              isAdmin={isAdmin}
            />
            <ul className="space-y-2">
              {servers.data.servers.map((server) => (
                <ServerRow key={server.id} server={server} onRemoved={refresh} />
              ))}
            </ul>
            {adding ? (
              <AddServer
                onAdded={() => {
                  setAdding(false);
                  refresh();
                }}
              />
            ) : (
              <Button variant="outline" size="sm" className="gap-1" onClick={() => setAdding(true)}>
                <Plus className="h-4 w-4" /> Add a model server
              </Button>
            )}
            <p className="text-xs text-muted-foreground">
              Keys entered here are kept on the server in <code className="rounded bg-muted px-1">{servers.data.keys_file}</code>,
              beside its database, and never shown again; a copy of that folder carries them. A key from an environment
              variable is never stored.
            </p>
          </>
        )}
      </CardContent>
    </Card>
  );
}
