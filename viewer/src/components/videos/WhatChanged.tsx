import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { GitCompare } from 'lucide-react';

import { apiClient } from '@/api/client';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { usePipelineNames } from '@/hooks/usePipelineNames';
import { settingsDiff } from '@/lib/settingsDiff';
import type { RunSummary } from '@/types/videos';
import { formatDay } from '@/lib/when';

const label = (run: RunSummary) =>
  `${run.name}${run.started_at ? ` · ${formatDay(run.started_at)}` : ''}`;

/** The settings a run ran with: its first video's. */
function useRunConfig(run: RunSummary | undefined) {
  return useQuery({
    queryKey: ['run-config', run?.id],
    enabled: !!run,
    retry: false,
    queryFn: async () => {
      const job = await apiClient.getJob(run!.job_ids[0]);
      return { pipelines: job.selected_pipelines ?? [], config: (job.config ?? {}) as Record<string, unknown> };
    },
  });
}

/**
 * What changed? between two runs (spec 029 FR-013): "Faces · min_face_size:
 * 30 → 40", and pipelines one ran that the other didn't.
 */
export function WhatChanged({ runs, initial }: { runs: RunSummary[]; initial?: [string, string] }) {
  const [open, setOpen] = useState(false);
  const [a, setA] = useState(initial?.[1] ?? runs[1]?.id ?? '');
  const [b, setB] = useState(initial?.[0] ?? runs[0]?.id ?? '');
  const { nameOf } = usePipelineNames(open);
  const left = useRunConfig(open ? runs.find((r) => r.id === a) : undefined);
  const right = useRunConfig(open ? runs.find((r) => r.id === b) : undefined);
  if (runs.length < 2) return null;

  const ready = left.data && right.data;
  const differences = ready ? settingsDiff({ settings: left.data!.config }, { settings: right.data!.config }) : [];
  const added = ready ? right.data!.pipelines.filter((p) => !left.data!.pipelines.includes(p)) : [];
  const dropped = ready ? left.data!.pipelines.filter((p) => !right.data!.pipelines.includes(p)) : [];
  const pick = (value: string, onChange: (v: string) => void, name: string) => (
    <select aria-label={name} value={value} onChange={(e) => onChange(e.target.value)} className="h-9 max-w-[16rem] rounded-md border border-input bg-background px-2 text-sm">
      {runs.map((r) => (
        <option key={r.id} value={r.id}>
          {label(r)}
        </option>
      ))}
    </select>
  );

  return (
    <>
      <Button variant="outline" size="sm" onClick={() => setOpen(true)}>
        <GitCompare className="mr-1 h-4 w-4" />
        What changed?
      </Button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="max-w-xl">
          <DialogHeader>
            <DialogTitle>What changed?</DialogTitle>
            <DialogDescription>The settings two runs ran with, where they differ.</DialogDescription>
          </DialogHeader>
          <div className="flex flex-wrap items-center gap-2 text-sm">
            {pick(a, setA, 'Earlier run')}
            <span>→</span>
            {pick(b, setB, 'Later run')}
          </div>
          <ul className="space-y-1 text-sm" data-testid="settings-changes">
            {!ready && <li className="text-muted-foreground">Reading their settings…</li>}
            {added.map((p) => <li key={`+${p}`}>{nameOf(p)}: added</li>)}
            {dropped.map((p) => <li key={`-${p}`}>{nameOf(p)}: left out</li>)}
            {differences.map((d) => {
              const [pipeline, ...rest] = d.key.split('.');
              return (
                <li key={d.key}>
                  {nameOf(pipeline)} · {rest.join('.') || pipeline}: {d.a} → {d.b}
                </li>
              );
            })}
            {ready && !differences.length && !added.length && !dropped.length && <li>Nothing: they ran with the same settings.</li>}
          </ul>
        </DialogContent>
      </Dialog>
    </>
  );
}
