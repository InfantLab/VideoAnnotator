import { Download } from 'lucide-react';

import { apiClient } from '@/api/client';
import { Button } from '@/components/ui/button';
import { runPipelineKey, type ProvenanceByRun, type RunFileRef } from '@/hooks/useRunFiles';
import type { Lane } from '@/lib/lanes';
import { settingsDiff } from '@/lib/settingsDiff';
import { NOT_RECORDED } from '@/lib/table/rows';

interface AboutPanelProps {
  lanes: Lane[];
  provenance: ProvenanceByRun;
  files: RunFileRef[];
  /** `${run}/${pipeline}` of compared runs. */
  compared: Set<string>;
}

type Record_ = Record<string, unknown> | null | undefined;

const isDict = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);

function models(record: Record_): string {
  const list = record?.models;
  if (!Array.isArray(list) || !list.length) return NOT_RECORDED;
  return list.map((m) => (isDict(m) ? String(m.name ?? '?') + (m.revision ? ` (${String(m.revision).slice(0, 8)})` : '') : '?')).join(' + ');
}

function settingsLines(record: Record_): [string, string][] {
  const settings = record?.settings;
  if (!isDict(settings)) return [];
  const out: [string, string][] = [];
  const walk = (v: unknown, prefix: string) => {
    if (isDict(v) && Object.keys(v).length) for (const [k, x] of Object.entries(v)) walk(x, prefix ? `${prefix}.${k}` : k);
    else out.push([prefix, typeof v === 'string' ? v : JSON.stringify(v)]);
  };
  walk(settings, '');
  return out;
}

async function save(file: RunFileRef) {
  const blob = await apiClient.downloadResultFile(file.url, file.name);
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = file.name;
  a.click();
  URL.revokeObjectURL(url);
}

/** Each lane's provenance (spec 017) and raw files; what differs between compared runs (spec 028 FR-012, US5). */
export function AboutPanel({ lanes, provenance, files, compared }: AboutPanelProps) {
  const shown = lanes.filter((l) => l.pipeline && l.status !== 'failed');
  const seen = new Set<string>();
  return (
    <div className="space-y-3 pb-3" data-testid="about-panel">
      {shown.map((lane) => {
        const key = runPipelineKey(lane.run?.id ?? '', lane.pipeline!);
        const isCompare = compared.has(key);
        // One entry per pipeline and run: Faces and Emotion from one file say the same.
        if (seen.has(key)) return null;
        seen.add(key);
        const record = provenance[key] as Record_;
        const own = files.filter((f) => f.jobId === lane.run?.id && f.pipeline === lane.pipeline);
        const main = isCompare
          ? lanes.find((l) => l.pipeline === lane.pipeline && !compared.has(runPipelineKey(l.run?.id ?? '', l.pipeline!)))
          : undefined;
        const diff = main ? settingsDiff(provenance[runPipelineKey(main.run?.id ?? '', main.pipeline!)] as Record_, record) : null;
        const settings = settingsLines(record);
        return (
          <section key={key} className="rounded border p-2" data-testid="about-entry">
            <h3 className="font-medium">
              {lane.pipeline}
              {lane.run && <span className="font-normal text-muted-foreground"> · {lane.run.label}</span>}
            </h3>
            <dl className="mt-1 grid grid-cols-[6rem_1fr] gap-x-2 gap-y-0.5">
              <dt className="text-muted-foreground">Model</dt>
              <dd className="break-words">{record ? models(record) : NOT_RECORDED}</dd>
              <dt className="text-muted-foreground">VideoAnnotator</dt>
              <dd>{record?.videoannotator_version ? String(record.videoannotator_version) : NOT_RECORDED}</dd>
              <dt className="text-muted-foreground">Settings</dt>
              <dd>
                {!record ? (
                  NOT_RECORDED
                ) : settings.length ? (
                  <details>
                    <summary className="cursor-pointer">{settings.length} recorded</summary>
                    <ul className="mt-1 font-mono text-[10px]">
                      {settings.map(([k, v]) => (
                        <li key={k} className="break-all">
                          {k}: {v}
                        </li>
                      ))}
                    </ul>
                  </details>
                ) : (
                  'none'
                )}
              </dd>
            </dl>
            {diff && main && (
              <div className="mt-2" data-testid="settings-diff">
                <p className="font-medium">Differs from {main.run?.label}</p>
                {diff.length === 0 ? (
                  <p className="text-muted-foreground">Same recorded settings.</p>
                ) : (
                  <table className="mt-1 w-full font-mono text-[10px]">
                    <thead>
                      <tr className="text-left text-muted-foreground">
                        <th className="font-normal">setting</th>
                        <th className="font-normal">{main.run?.label}</th>
                        <th className="font-normal">{lane.run?.label}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {diff.map((d) => (
                        <tr key={d.key}>
                          <td className="break-all pr-1">{d.key}</td>
                          <td className="break-all pr-1">{d.a}</td>
                          <td className="break-all">{d.b}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            )}
            {own.length > 0 && (
              <ul className="mt-2 space-y-0.5">
                {own.map((f) => (
                  <li key={f.name} className="flex items-center gap-1">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-5 w-5"
                      aria-label={`Download ${f.name}`}
                      title="Download"
                      onClick={() => void save(f)}
                    >
                      <Download className="h-3 w-3" />
                    </Button>
                    <span className="truncate font-mono text-[10px]">{f.name}</span>
                  </li>
                ))}
              </ul>
            )}
            {!own.length && lane.source.length > 0 && (
              <p className="mt-2 truncate font-mono text-[10px] text-muted-foreground">{lane.source.join(', ')}</p>
            )}
          </section>
        );
      })}
      {shown.length === 0 && <p className="text-muted-foreground">No results to describe yet.</p>}
    </div>
  );
}
