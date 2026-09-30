// Load or save a pipeline preset from the job wizard (VideoAnnotator spec 007,
// viewer-handoff item 4). A preset is the wizard's own selection + config,
// saved on the server and shared with everyone on it.

import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Bookmark, Loader2, Save } from 'lucide-react';
import { apiClient } from '@/api/client';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import type { Preset } from '@/types/presets';

const PRESETS_KEY = ['presets'] as const;

interface PresetBarProps {
  selectedPipelines: string[];
  config: Record<string, unknown>;
  onApply: (preset: Preset) => void;
}

export const PresetBar = ({ selectedPipelines, config, onApply }: PresetBarProps) => {
  const queryClient = useQueryClient();
  const [chosenId, setChosenId] = useState('');
  const [saving, setSaving] = useState(false);
  const [name, setName] = useState('');
  const [message, setMessage] = useState<string | null>(null);

  // Servers before v1.5.0 have no presets endpoint: then the bar just doesn't show.
  const { data, isError } = useQuery({
    queryKey: PRESETS_KEY,
    queryFn: () => apiClient.listPresets(),
    retry: false,
    staleTime: 30_000,
  });
  const presets = data?.presets ?? [];

  const save = useMutation({
    mutationFn: () =>
      apiClient.createPreset({
        name: name.trim(),
        selected_pipelines: selectedPipelines,
        // Only the selected pipelines' settings belong to this preset.
        config: Object.fromEntries(Object.entries(config).filter(([id]) => selectedPipelines.includes(id))),
      }),
    onSuccess: (preset) => {
      queryClient.invalidateQueries({ queryKey: PRESETS_KEY });
      setSaving(false);
      setName('');
      setChosenId(preset.id);
      setMessage(`Saved “${preset.name}”.`);
    },
    onError: (error: Error) => setMessage(`Couldn't save the preset: ${error.message}`),
  });

  if (isError) return null;

  const chosen = presets.find((p) => p.id === chosenId);

  return (
    <div className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-card p-3 text-sm">
      <Bookmark className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
      {presets.length > 0 ? (
        <>
          <select
            aria-label="Preset"
            className="h-8 rounded-md border border-input bg-background px-2 text-sm"
            value={chosenId}
            onChange={(e) => setChosenId(e.target.value)}
          >
            <option value="">Choose a preset…</option>
            {presets.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name} ({p.selected_pipelines.length} pipeline{p.selected_pipelines.length === 1 ? '' : 's'})
              </option>
            ))}
          </select>
          <Button
            type="button"
            size="sm"
            variant="outline"
            className="h-8"
            disabled={!chosen}
            onClick={() => {
              if (!chosen) return;
              onApply(chosen);
              setMessage(`Applied “${chosen.name}”.`);
            }}
          >
            Load
          </Button>
        </>
      ) : (
        <span className="text-muted-foreground">No saved presets yet.</span>
      )}

      <span className="mx-1 hidden h-5 w-px bg-border sm:inline-block" aria-hidden="true" />

      {saving ? (
        <form
          className="flex items-center gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            if (name.trim()) save.mutate();
          }}
        >
          <Input
            autoFocus
            aria-label="Preset name"
            placeholder="Preset name"
            className="h-8 w-48"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          <Button type="submit" size="sm" className="h-8" disabled={!name.trim() || save.isPending}>
            {save.isPending ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : null}
            Save
          </Button>
          <Button type="button" size="sm" variant="ghost" className="h-8" onClick={() => setSaving(false)}>
            Cancel
          </Button>
        </form>
      ) : (
        <Button
          type="button"
          size="sm"
          variant="outline"
          className="h-8"
          disabled={selectedPipelines.length === 0}
          onClick={() => {
            setMessage(null);
            setSaving(true);
          }}
        >
          <Save className="mr-1 h-3 w-3" />
          Save selection as preset
        </Button>
      )}

      {message && <span className="basis-full text-xs text-muted-foreground">{message}</span>}
    </div>
  );
};
