import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select';
import { CheckCircle, AlertTriangle, Loader2 } from 'lucide-react';
import { useVlmModels } from '@/hooks/useVlmModels';
import { handleAPIError } from '@/api/handleError';

interface VlmModelPickerProps {
  value: string;
  onChange: (model: string) => void;
  fieldId: string;
}

/**
 * Model picker for vlm_annotation's `model` field (VideoAnnotator spec 009
 * US2), plus an Ollama-reachability badge sourced from the same query (no
 * separate call needed — spec's viewer-handoff #3). Falls back to the
 * original free-text input whenever there's nothing live to pick from: the
 * server is unreachable, or reachable but has zero models pulled — don't
 * hard-block configuration just because Ollama isn't running yet.
 */
export const VlmModelPicker = ({ value, onChange, fieldId }: VlmModelPickerProps) => {
  const { data, isLoading, isError, error } = useVlmModels();

  const reachabilityBadge = isLoading ? (
    <Badge variant="secondary" className="gap-1">
      <Loader2 className="h-3 w-3 animate-spin" />
      Checking Ollama…
    </Badge>
  ) : isError ? (
    <Badge variant="destructive" className="gap-1">
      <AlertTriangle className="h-3 w-3" />
      Ollama unreachable
    </Badge>
  ) : data && data.models.length === 0 ? (
    <Badge variant="secondary" className="gap-1">
      <AlertTriangle className="h-3 w-3" />
      No models pulled
    </Badge>
  ) : (
    <Badge variant="default" className="gap-1">
      <CheckCircle className="h-3 w-3" />
      {data?.models.length} model{data?.models.length === 1 ? '' : 's'} available
    </Badge>
  );

  const showDropdown = !isLoading && !isError && (data?.models.length ?? 0) > 0;

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between gap-2">
        <Label htmlFor={fieldId} className="text-sm">
          Model
        </Label>
        {reachabilityBadge}
      </div>

      {showDropdown ? (
        <Select value={value || undefined} onValueChange={onChange}>
          <SelectTrigger id={fieldId}>
            <SelectValue placeholder="Select a model" />
          </SelectTrigger>
          <SelectContent>
            {data!.models.map((model) => (
              <SelectItem key={model} value={model}>
                {model}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      ) : (
        <>
          <Input
            id={fieldId}
            value={value}
            placeholder="e.g. qwen3.5:9b"
            onChange={(event) => onChange(event.target.value)}
          />
          <p className="text-xs text-muted-foreground">
            {isError
              ? `Can't reach the configured Ollama server (${handleAPIError(error)}). Type a model name, or start Ollama and reload.`
              : data && data.models.length === 0
                ? `Ollama is reachable at ${data.baseUrl} but nothing is pulled yet — run "ollama pull <model>", or type a model name to pull later.`
                : 'Type the exact model name as it appears in "ollama list".'}
          </p>
        </>
      )}
    </div>
  );
};
