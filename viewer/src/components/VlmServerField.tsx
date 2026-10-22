import type { ReactNode } from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/api/client';
import { VlmModelPicker } from '@/components/VlmModelPicker';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { KIND_SHORT, MODEL_SERVERS_QUERY_KEY, serverForConfig } from '@/lib/modelServers';
import { DEFAULT_SERVER_ID, type ModelServer } from '@/types/ask';

const OTHER = '__other__';

function useListedServers(): ModelServer[] {
  const { data } = useQuery({
    queryKey: MODEL_SERVERS_QUERY_KEY,
    queryFn: () => apiClient.listModelServers(),
    retry: false,
  });
  return data?.servers ?? [];
}

interface VlmServerFieldProps {
  fieldId: string;
  baseUrl: unknown;
  onChoose: (server: ModelServer) => void;
  /** Shown instead while no other model server is listed. */
  fallback: ReactNode;
}

/**
 * vlm_annotation's `base_url` as a choice of the listed model servers (spec
 * 033): choosing one sets the address and the kind together. Any other
 * address stays as it is, and can be edited under "Edit as JSON".
 */
export function VlmServerField({ fieldId, baseUrl, onChoose, fallback }: VlmServerFieldProps) {
  const servers = useListedServers();
  if (servers.length < 2) return <>{fallback}</>;
  const current = serverForConfig(servers, baseUrl);
  return (
    <div className="space-y-1" data-testid="vlm-server-field">
      <Label htmlFor={fieldId} className="text-sm">
        Model server
      </Label>
      <Select
        value={current?.id ?? OTHER}
        onValueChange={(id) => {
          const chosen = servers.find((s) => s.id === id);
          if (chosen) onChoose(chosen);
        }}
      >
        <SelectTrigger id={fieldId}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {servers.map((server) => (
            <SelectItem key={server.id} value={server.id}>
              {server.name}
              {server.id !== DEFAULT_SERVER_ID && <span className="text-muted-foreground"> · {KIND_SHORT[server.kind]}</span>}
            </SelectItem>
          ))}
          {!current && (
            <SelectItem value={OTHER} disabled>
              {String(baseUrl)} (not in Settings → Model servers)
            </SelectItem>
          )}
        </SelectContent>
      </Select>
    </div>
  );
}

interface VlmModelFieldProps {
  baseUrl: unknown;
  value: string;
  onChange: (model: string) => void;
  fieldId: string;
}

/** The model picker for the server a vlm_annotation config points at. */
export function VlmModelField({ baseUrl, ...picker }: VlmModelFieldProps) {
  const server = serverForConfig(useListedServers(), baseUrl);
  return <VlmModelPicker {...picker} server={server?.id} serverName={server?.name} />;
}
