import { useCallback, useState } from 'react';

import { WhatToFind } from '@/components/run/WhatToFind';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { usePipelineCatalog } from '@/hooks/usePipelineCatalog';
import { useWorkflowActions } from '@/hooks/useWorkflows';
import { notify } from '@/lib/notify';
import { configFor } from '@/lib/pipelineDefaults';
import { userMessage } from '@/lib/userMessage';
import type { Preset } from '@/types/presets';

/** New workflow, or Edit one: a name and New run's What to find, without videos (UAT #71). */
export function WorkflowEditor({ workflow, open, onOpenChange }: { workflow?: Preset; open: boolean; onOpenChange: (open: boolean) => void }) {
  const catalog = usePipelineCatalog({ enabled: open });
  const { create, update } = useWorkflowActions();
  const [name, setName] = useState(workflow?.name ?? '');
  const [selected, setSelected] = useState<string[]>(workflow?.selected_pipelines ?? []);
  const [config, setConfig] = useState<Record<string, unknown>>(workflow?.config ?? {});
  const [valid, setValid] = useState(true);
  const onConfig = useCallback((updater: (prev: Record<string, unknown>) => Record<string, unknown>) => setConfig(updater), []);
  const saving = create.isPending || update.isPending;

  const save = () => {
    const body = { name: name.trim(), selected_pipelines: selected, config: configFor(selected, config) ?? {} };
    const done = {
      onSuccess: () => {
        notify({ title: workflow ? `Workflow ${body.name} saved` : `Workflow ${body.name} made`, tone: 'success' as const });
        onOpenChange(false);
      },
      onError: (e: unknown) => notify({ title: "Couldn't save the workflow", body: userMessage(e), tone: 'error' as const }),
    };
    if (workflow) update.mutate({ id: workflow.id, body }, done);
    else create.mutate(body, done);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[90vh] max-w-3xl overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{workflow ? `Edit ${workflow.name}` : 'New workflow'}</DialogTitle>
        </DialogHeader>
        <div className="space-y-1">
          <Label htmlFor="workflow-name">Name</Label>
          <Input id="workflow-name" value={name} onChange={(e) => setName(e.target.value)} placeholder="Peekaboo, standard" autoFocus />
        </div>
        <WhatToFind
          pipelines={catalog.data?.catalog.pipelines ?? []}
          isLoading={catalog.isLoading}
          error={catalog.error}
          restartRequired={catalog.data?.restartRequired ?? false}
          selected={selected}
          onSelect={setSelected}
          config={config}
          onConfig={onConfig}
          source={workflow ? `this workflow, as saved` : 'the defaults'}
          onValidity={setValid}
        />
        <DialogFooter>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button disabled={!name.trim() || selected.length === 0 || !valid || saving} onClick={save}>
            {workflow ? 'Save' : 'Make workflow'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
