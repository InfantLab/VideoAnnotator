import { useState } from 'react';
import { Bookmark } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { useWorkflowActions } from '@/hooks/useWorkflows';
import { notify } from '@/lib/notify';
import { userMessage } from '@/lib/userMessage';
import type { ExportChoices, SettingsSource } from '@/types/videos';

export interface SavedSettings {
  selected: string[];
  config: Record<string, unknown>;
}

/**
 * Save as workflow (spec 029 FR-014): a project's or run's settings under a
 * name, with how its results are offered, to reuse and share.
 */
export function SaveAsWorkflow({
  source,
  settings,
  size = 'sm',
}: {
  source: SettingsSource & { name: string };
  /** The settings to save, read when the dialog opens. */
  settings: () => Promise<SavedSettings>;
  size?: 'sm' | 'default';
}) {
  const { create } = useWorkflowActions();
  const [open, setOpen] = useState(false);
  const [name, setName] = useState('');
  const [table, setTable] = useState<'csv' | 'parquet'>('csv');
  const [elan, setElan] = useState(true);
  const [methods, setMethods] = useState(true);
  const [problem, setProblem] = useState<string | null>(null);

  const save = async () => {
    setProblem(null);
    try {
      const { selected, config } = await settings();
      const choices: ExportChoices = { table, elan, methods };
      const workflow = await create.mutateAsync({
        name: name.trim(),
        selected_pipelines: selected,
        config: Object.fromEntries(Object.entries(config).filter(([id]) => selected.includes(id))),
        export_choices: choices,
        source,
      });
      setOpen(false);
      notify({
        title: `Workflow ${workflow.name} saved`,
        body: 'Choose it in New run, or apply it to a project.',
        action: { label: 'Workflows', to: '/pipelines#workflows' },
        tone: 'success',
      });
    } catch (error) {
      setProblem(userMessage(error));
    }
  };

  return (
    <>
      <Button
        variant="outline"
        size={size}
        onClick={() => {
          setName(source.name);
          setOpen(true);
        }}
      >
        <Bookmark className="mr-1 h-4 w-4" />
        Save as workflow
      </Button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Save as workflow</DialogTitle>
            <DialogDescription>
              The pipelines and settings of {source.name}, under a name, to start runs from or share as a file.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-1">
              <Label htmlFor="workflow-name">Name</Label>
              <Input id="workflow-name" value={name} onChange={(e) => setName(e.target.value)} autoFocus />
            </div>
            <fieldset className="space-y-2 text-sm">
              <legend className="mb-1 font-medium">Results it offers first</legend>
              <label className="flex items-center gap-2">
                Results table as
                <select aria-label="Table format" value={table} onChange={(e) => setTable(e.target.value as 'csv' | 'parquet')} className="h-8 rounded-md border border-input bg-background px-2">
                  <option value="csv">CSV</option>
                  <option value="parquet">Parquet</option>
                </select>
              </label>
              <label className="flex items-center gap-2">
                <Checkbox checked={elan} onCheckedChange={(on) => setElan(on === true)} aria-label="ELAN files" />
                ELAN files
              </label>
              <label className="flex items-center gap-2">
                <Checkbox checked={methods} onCheckedChange={(on) => setMethods(on === true)} aria-label="Methods paragraph" />
                Methods paragraph
              </label>
            </fieldset>
            {problem && (
              <p className="text-sm text-destructive" role="alert">
                {problem}
              </p>
            )}
          </div>
          <DialogFooter>
            <Button onClick={save} disabled={!name.trim() || create.isPending}>
              Save workflow
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
