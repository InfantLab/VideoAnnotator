import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { KIND_SHORT } from '@/lib/modelServers';
import type { ModelServer } from '@/types/ask';

interface ServerPickerProps {
  servers: ModelServer[];
  value: string;
  onChange: (id: string) => void;
  fieldId: string;
}

/** Which model server answers (spec 033): this server's Ollama unless another is chosen. */
export function ServerPicker({ servers, value, onChange, fieldId }: ServerPickerProps) {
  if (servers.length < 2) return null;
  return (
    <div className="space-y-1" data-testid="server-picker">
      <Label htmlFor={fieldId} className="text-sm">
        Model server
      </Label>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger id={fieldId}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {servers.map((server) => (
            <SelectItem key={server.id} value={server.id}>
              {server.name}
              {server.id !== 'default' && <span className="text-muted-foreground"> · {KIND_SHORT[server.kind]}</span>}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
